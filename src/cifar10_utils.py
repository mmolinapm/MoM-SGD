"""Utility definitions extracted verbatim from the historical CIFAR engine."""
import csv,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision import datasets,transforms
from src.optimizers import gradient_norm
def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

def gradient(net, x, y):
    loss = nn.functional.cross_entropy(net(x), y)
    g = torch.cat([v.reshape(-1) for v in torch.autograd.grad(loss, tuple(net.parameters()))])
    return g, loss.detach()

def normalize(g):
    n = gradient_norm(g)
    return g.clone() if n == 0 else g/n

def evaluate(net, loader, device):
    sync(device)
    start = time.perf_counter()
    prior = net.training
    net.eval()
    loss = correct = count = 0
    try:
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                z = net(x)
                loss += nn.functional.cross_entropy(z, y, reduction='sum').item()
                correct += (z.argmax(1) == y).sum().item()
                count += len(y)
    finally:
        net.train(prior)
    sync(device)
    assert np.isfinite(loss)
    return loss/count, correct/count, time.perf_counter()-start

def atomic_save(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)

def dump(path, value):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(tmp, path)

def table(path, rows):
    if not rows:
        return
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)

def batch(training, indices, device):
    if isinstance(training,CachedCIFAR):
        return training.get_batch(indices,device)
    pairs = [training[int(j)] for j in indices]
    return torch.stack([x for x, _ in pairs]).to(device), torch.tensor([y for _, y in pairs], device=device)

class CachedCIFAR:
    """Exact float32 sample tensors, shared through private file-backed CPU mappings."""
    def __init__(self, images, targets):
        assert images.dtype == torch.float32 and images.is_contiguous()
        self.images = images
        self.targets = torch.as_tensor(targets,dtype=torch.int64)
        assert len(images) == len(self.targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.images[index], int(self.targets[index])

    def get_batch(self, indices, device):
        indices = torch.as_tensor(indices,dtype=torch.int64,device='cpu')
        return (self.images.index_select(0,indices).to(device),
                self.targets.index_select(0,indices).to(device))

class CachedEvaluationBatches:
    """Sequential 512-example views; preserve original evaluation grouping and order."""
    def __init__(self, dataset, batch_size=512):
        self.dataset, self.batch_size = dataset,batch_size

    def __iter__(self):
        for start in range(0,len(self.dataset),self.batch_size):
            yield (self.dataset.images[start:start+self.batch_size],
                   self.dataset.targets[start:start+self.batch_size])

    def __len__(self):
        return (len(self.dataset)+self.batch_size-1)//self.batch_size

def file_sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):
            h.update(block)
    return h.hexdigest()

def cache_cifar(dataset, folder, split):
    """Apply the original callable once per sample; never fuse/change arithmetic."""
    start = time.perf_counter()
    folder.mkdir(parents=True,exist_ok=True)
    path, manifest_path = folder/(split+'.f32'), folder/(split+'.json')
    raw = hashlib.sha256()
    for first in range(0,len(dataset),512):
        raw.update(dataset.data[first:first+512].tobytes())
    key = dict(version=1, samples=len(dataset), shape=[len(dataset),3,32,32], dtype='float32',
               raw_sha256=raw.hexdigest(), targets_sha256=hashlib.sha256(np.asarray(dataset.targets,dtype=np.int64).tobytes()).hexdigest(),
               torch=torch.__version__, torchvision=__import__('torchvision').__version__,
               preprocessing='PIL; ToTensor; type(default float32); Normalize(.491,.482,.447)/(.247,.243,.262)')
    created = False
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['key'] != key or not path.exists() or path.stat().st_size != len(dataset)*3072*4:
            raise ValueError(f'Cache metadata differs: {manifest_path}; use a fresh --cache-dir')
        if file_sha256(path) != manifest['tensor_sha256']:
            raise ValueError(f'Cache integrity failed: {path}')
    else:
        if path.exists():
            raise ValueError(f'Uncommitted cache file: {path}; use a fresh --cache-dir')
        tmp = path.with_suffix('.f32.tmp')
        mapping = np.memmap(tmp,mode='w+',dtype=np.float32,shape=key['shape'])
        try:
            for i in range(len(dataset)):
                image,target = dataset[i]
                assert image.dtype == torch.float32 and image.shape == (3,32,32)
                assert int(target) == dataset.targets[i]
                mapping[i] = image.numpy()
            mapping.flush()
        finally:
            mapping._mmap.close()
        os.replace(tmp,path)
        manifest = dict(key=key,tensor_sha256=file_sha256(path))
        dump(manifest_path,manifest)
        created = True
    images = torch.from_file(str(path),shared=False,size=len(dataset)*3072,dtype=torch.float32).reshape(key['shape'])
    cached = CachedCIFAR(images,dataset.targets)
    return cached,dict(created=created,seconds=time.perf_counter()-start,bytes=path.stat().st_size,
                       tensor_sha256=manifest['tensor_sha256'],path=str(path.resolve()))

def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['torch'].cpu())
    if s['cuda']:
        torch.cuda.set_rng_state_all([v.cpu() for v in s['cuda']])
