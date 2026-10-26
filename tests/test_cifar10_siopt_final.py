"""SIOPT scientific invariants; fixtures are synthetic and never scientific results."""
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

SOURCE=Path(__file__).resolve().parents[1]/'experiments/cifar10_siopt_final.py'
spec=importlib.util.spec_from_file_location('siopt_tests',SOURCE)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
torch.set_num_threads(2)


def test_protocol_schedule_and_lr_boundaries():
    assert list(p.METHODS)==['sgd','normalized','clipped_0.25','clipped_0.5','clipped_2','clipped_4','clipped_8','mom_K5_m20']
    assert [p.lr_at(i) for i in (1,4999,5000,9999,10000,15000)]==[.1,.1,.05,.05,.025,.025]
    for i in (0,15001):
        with pytest.raises(ValueError):p.lr_at(i)
    assert p.evaluation_iterations()==list(range(0,15001,500))
    indices=p.schedule(42)
    assert indices.shape==(15000,100) and indices.dtype==torch.int32
    rng=torch.Generator().manual_seed(42)
    for epoch in indices.reshape(30,50000):
        assert torch.equal(epoch.long(),torch.randperm(50000,generator=rng))
        assert epoch.unique().numel()==50000
    torch.rand(100);assert torch.equal(indices,p.schedule(42))
    assert not torch.equal(indices[:2],p.schedule(43,steps=2))


@pytest.mark.parametrize('architecture',['cnn'])
def test_architecture_gradient_and_all_eight_updates(architecture,monkeypatch):
    torch.manual_seed(42);initial=p.model(architecture);n=p.PARAMETERS[architecture]
    assert sum(v.numel() for v in initial.parameters())==n
    assert not any(isinstance(v,(torch.nn.Dropout,torch.nn.BatchNorm2d)) for v in initial.modules())
    assert all(v.bias is None for v in initial.modules() if isinstance(v,(torch.nn.Linear,torch.nn.Conv2d)))
    if architecture=='cnn':
        assert [v.weight.shape for v in initial.modules() if isinstance(v,torch.nn.Conv2d)]==[torch.Size([32,3,3,3]),torch.Size([64,32,3,3]),torch.Size([128,64,3,3])]
    x,y=torch.randn(100,3,32,32),torch.arange(100)%10
    assert initial(x).shape==(100,10)
    full,_=p.e.gradient(initial,x,y)
    blocks=torch.stack([p.e.gradient(initial,a,b)[0] for a,b in zip(x.split(20),y.split(20))])
    torch.testing.assert_close(blocks.mean(0),full,rtol=5e-4,atol=5e-7)
    canonical=p.e.gradient;calls=[]
    def gradient(net,a,b):calls.append(len(a));return canonical(net,a,b)
    monkeypatch.setattr(p.e,'gradient',gradient)
    for name,specification in p.METHODS.items():
        net=copy.deepcopy(initial);reference=copy.deepcopy(initial);calls.clear()
        assert all(torch.equal(a,b) and a.data_ptr()!=b.data_ptr() for a,b in zip(initial.parameters(),net.parameters()))
        expected=p.coordinatewise_median(blocks) if specification['K'] else p.e.normalize(full) if name=='normalized' else p.clip_gradient(full,specification['gamma']) if specification['gamma'] is not None else full
        offset=0
        for parameter in reference.parameters():
            size=parameter.numel();p.sgd_step_(parameter,expected[offset:offset+size].view_as(parameter),.05);offset+=size
        row=p.update(net,x,y,architecture,name,5000)
        assert calls==([20]*5 if specification['K'] else [100])
        assert all(torch.equal(a,b) for a,b in zip(net.parameters(),reference.parameters()))
        assert row['learning_rate']==.05 and row['B']==100
        if specification['gamma'] is not None:assert row['effective_gradient_norm']<=specification['gamma']+1e-6
        if name=='normalized':assert abs(row['effective_gradient_norm']-1.)<1e-6
        if specification['K']:
            assert row['K']==5 and row['m']==20 and row['clipping_factor']==1.
            assert row['arithmetic_mean_gradient_norm']==p.gradient_norm(blocks.mean(0)).item()


@pytest.mark.parametrize('architecture',['cnn'])
def test_evaluation_preserves_all_state_and_weights_observations(architecture):
    torch.manual_seed(42);net=p.model(architecture)
    x,y=torch.randn(23,3,32,32),torch.arange(23)%10
    for parameter in net.parameters():parameter.grad=torch.randn_like(parameter)
    before=copy.deepcopy(net.state_dict());grads=[v.grad.clone() for v in net.parameters()];rng=p.e.rng_state()
    net.train();loss,accuracy,_=p.evaluate(net,[(x[:17],y[:17]),(x[17:],y[17:])],torch.device('cpu'))
    z=net(x)
    assert abs(loss-torch.nn.functional.cross_entropy(z,y).item())<1e-6
    assert accuracy==int((z.argmax(1)==y).sum())/23
    assert net.training and all(torch.equal(before[n],v) for n,v in net.state_dict().items())
    assert all(torch.equal(g,v.grad) for g,v in zip(grads,net.parameters()))
    assert torch.equal(rng['torch'],torch.get_rng_state()) and rng['python']==p.random.getstate()
    assert np.array_equal(rng['numpy'][1],np.random.get_state()[1])


def test_clipping_normalization_and_zero_edges(monkeypatch):
    torch.manual_seed(42);net=p.model('fcn4');x=torch.zeros(100,3,32,32);y=torch.zeros(100,dtype=torch.long)
    for scale in (0.,1.,10000.):
        raw=torch.ones(427264)*scale
        monkeypatch.setattr(p.e,'gradient',lambda *a:(raw,torch.tensor(0.)))
        for method in ['normalized']+[f'clipped_{g:g}' for g in p.CLIPPING_GAMMAS]:
            row=p.update(copy.deepcopy(net),x,y,'fcn4',method,1)
            if method=='normalized':assert abs(row['effective_gradient_norm']-(1. if scale else 0.))<1e-6
            else:
                gamma=p.METHODS[method]['gamma'];assert row['effective_gradient_norm']<=gamma+1e-6
                assert row['clipping_factor']==(min(1.,gamma/p.gradient_norm(raw).item()) if scale else 1.)
                assert row['clipping_active']==bool(p.gradient_norm(raw)>gamma)


WORKER=r"""
import importlib.util,json,random,sys
from pathlib import Path
import numpy as np,torch
source,architecture,fixture,bank_path,target,phase=sys.argv[1:]
spec=importlib.util.spec_from_file_location('siopt_fresh',source);p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
initial=p.model(architecture);initial.load_state_dict(torch.load(fixture,weights_only=True))
bank=torch.from_file(bank_path,shared=False,size=400*3072,dtype=torch.float32).reshape(400,3,32,32)
class Mapped(p.e.CachedCIFAR):
    def get_batch(self,indices,device):return super().get_batch(indices.long()%400,device)
data=Mapped(bank,torch.arange(400)%10)
loader=p.e.CachedEvaluationBatches(p.e.CachedCIFAR(bank[:23],torch.arange(23)%10),17)
indices=p.schedule(42,steps=4);update=p.update
trace={}
def step(net,x,y,arch,name,i):
    if phase=='crash' and i==3:raise RuntimeError('simulated interruption')
    row=update(net,x,y,arch,name,i);trace.setdefault(name,[]).append(indices[i-1].tolist());return row
p.update=step
out=Path(target);out.mkdir(exist_ok=True)
for method in p.METHODS:
    seed=937 if phase=='resume' else 42
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    try:p.train_one(out,architecture,method,initial,data,[loader,loader],indices,torch.device('cpu'),steps=4,resume=phase=='resume',checkpoint_every=1,evaluation_interval=2)
    except RuntimeError as exc:assert phase=='crash' and str(exc)=='simulated interruption'
    else:assert phase!='crash'
(out/(phase+'_trace.json')).write_text(json.dumps(trace))
"""


@pytest.mark.parametrize('architecture',['cnn'])
def test_fresh_process_exact_resume_all_methods_and_shared_indices(tmp_path,architecture):
    torch.manual_seed(42);fixture=tmp_path/'initial.pt';torch.save(p.model(architecture).state_dict(),fixture)
    bank=tmp_path/'bank.f32';torch.randn(400,3,32,32).numpy().tofile(bank)
    a,b=tmp_path/'continuous',tmp_path/'resumed'
    def run(out,phase):
        subprocess.run([sys.executable,'-c',WORKER,str(SOURCE),architecture,str(fixture),str(bank),str(out),phase],check=True,capture_output=True,text=True)
    run(a,'continuous');run(b,'crash');run(b,'resume')
    uninterrupted=json.loads((a/'continuous_trace.json').read_text());prefix=json.loads((b/'crash_trace.json').read_text());suffix=json.loads((b/'resume_trace.json').read_text())
    expected=p.schedule(42,steps=4).tolist()
    for method in p.METHODS:
        assert uninterrupted[method]==prefix[method]+suffix[method]==expected
        sa=torch.load(a/'checkpoints'/f'{method}.pt',weights_only=False);sb=torch.load(b/'checkpoints'/f'{method}.pt',weights_only=False)
        assert p.state_digest(sa['model'])==p.state_digest(sb['model'])
        assert sb['next_schedule_row']==4 and sb['next_learning_rate']==.1
        assert [r['iteration'] for r in sb['evaluations']]==[0,2,4]
        for category in ('diagnostics','evaluations'):
            for ra,rb in zip(sa[category],sb[category]):
                for key in ra:
                    if key.endswith('seconds'):continue
                    if isinstance(ra[key],float):np.testing.assert_allclose(ra[key],rb[key],rtol=0,atol=0,equal_nan=True)
                    else:assert ra[key]==rb[key]
        assert torch.equal(sa['rng']['torch'],sb['rng']['torch']) and sa['rng']['python']==sb['rng']['python']
        assert np.array_equal(sa['rng']['numpy'][1],sb['rng']['numpy'][1])


def test_numerical_failure_preserves_durable_checkpoint_and_rejects_changed_schedule(tmp_path,monkeypatch):
    torch.manual_seed(42);initial=p.model('fcn4');data=p.e.CachedCIFAR(torch.randn(100,3,32,32),torch.arange(100)%10)
    loader=p.e.CachedEvaluationBatches(p.e.CachedCIFAR(data.images[:3],data.targets[:3]))
    indices=torch.arange(100,dtype=torch.int32).reshape(1,100)
    monkeypatch.setattr(p.e,'gradient',lambda *a:(torch.full((427264,),float('nan')),torch.tensor(float('nan'))))
    result=p.train_one(tmp_path,'fcn4','sgd',initial,data,[loader,loader],indices,torch.device('cpu'),steps=1)
    assert result['status']['status']=='failed'
    checkpoint=torch.load(tmp_path/'checkpoints/sgd.pt',weights_only=False)
    assert checkpoint['iteration']==0 and p.state_digest(checkpoint['model'])==p.state_digest(initial.state_dict())
    assert (tmp_path/'progress/sgd_failure.pt').exists()
    with pytest.raises(ValueError,match='schedule mismatch'):
        p.train_one(tmp_path,'fcn4','sgd',initial,data,[loader,loader],indices.flip(1),torch.device('cpu'),steps=1,resume=True)


def test_export_phases_final_window_and_duplicate_free(tmp_path):
    results={}
    for method,specification in p.METHODS.items():
        d=[dict(architecture='fcn4',method=method,iteration=i,learning_rate=p.lr_at(i),B=100,
            effective_gradient_norm=1.,cosine_similarity=.5 if specification['K'] else float('nan'),clipping_active=specification['gamma'] is not None,
            clipping_factor=.5 if specification['gamma'] is not None else 1.,parameter_update_seconds=.001) for i in range(1,15001)]
        v=[dict(architecture='fcn4',method=method,iteration=i,training_loss=float(i),test_loss=float(i)+1,training_accuracy=.2,test_accuracy=.1,evaluation_seconds=.1) for i in p.evaluation_iterations()]
        results[method]=dict(diagnostics=d,evaluations=v,timing=dict(checkpoint_save_seconds=.1,progress_export_seconds=.2),wall_seconds=1.,status=dict(status='completed'))
    for _ in range(2):p.export_results(tmp_path,results,15000,False)
    with (tmp_path/'run_summary.csv').open() as f:rows=list(p.e.csv.DictReader(f))
    assert len(rows)==8 and all(float(r['final_window_training_loss'])==14500 for r in rows)
    with (tmp_path/'clipping_frequency_summary.csv').open() as f:phases=list(p.e.csv.DictReader(f))
    assert len(phases)==20
    assert [int(r['updates']) for r in phases[:4]]==[4999,5000,5001,15000]
    with (tmp_path/'training_loss_history.csv').open() as f:history=list(p.e.csv.DictReader(f))
    assert len(history)==len({(r['architecture'],r['method'],r['iteration']) for r in history})==8*31


def test_cli_refuses_overwrite_before_data_setup(tmp_path):
    folder=tmp_path/'cnn_seed42';folder.mkdir();marker=folder/'keep.txt';marker.write_text('existing result')
    result=subprocess.run([sys.executable,str(SOURCE),'preflight','--architecture','cnn','--output',str(tmp_path)],capture_output=True,text=True)
    assert result.returncode!=0 and 'Output exists' in result.stderr and marker.read_text()=='existing result'
