"""Final controlled CIFAR-10 FCN-4/CNN study. Full execution requires manual run mode."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
ENGINE_PATH=ROOT/'src/cifar10_utils.py'
from src import cifar10_utils as e
torch,np,nn=e.torch,e.np,e.nn
from src.mom import coordinatewise_median
from src.optimizers import clip_gradient,gradient_norm,sgd_step_

SEED,NUM_ITERATIONS,BATCH_SIZE,MOM_K,MOM_M=42,15000,100,5,20
EVALUATION_INTERVAL,CHECKPOINT_EVERY=500,250
CLIPPING_GAMMAS=(.25,.5,2.,4.,8.)
METHODS={'sgd':{},'normalized':{},**{f'clipped_{g:g}':{'gamma':g} for g in CLIPPING_GAMMAS},'mom_K5_m20':{'K':5}}
METHODS={n:dict(K=v.get('K',0),m=20 if v.get('K') else 0,gamma=v.get('gamma')) for n,v in METHODS.items()}
PARAMETERS={'fcn4':427264,'cnn':94304}

class NumericalFailure(RuntimeError):
    def __init__(self,message,payload=None):
        super().__init__(message);self.payload=payload or {}


def model(architecture):
    if architecture=='fcn4':
        net=nn.Sequential(nn.Flatten(),nn.Linear(3072,128,bias=False),nn.ReLU(),nn.Linear(128,128,bias=False),
            nn.ReLU(),nn.Linear(128,128,bias=False),nn.ReLU(),nn.Linear(128,10,bias=False))
    elif architecture=='cnn':
        net=nn.Sequential(nn.Conv2d(3,32,3,padding=1,bias=False),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1,bias=False),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1,bias=False),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),nn.Linear(128,10,bias=False))
    else:raise ValueError(architecture)
    assert sum(p.numel() for p in net.parameters())==PARAMETERS[architecture]
    return net


def lr_at(i):
    if not 1<=i<=NUM_ITERATIONS:raise ValueError(i)
    return .1 if i<5000 else .05 if i<10000 else .025


def evaluation_iterations(steps=NUM_ITERATIONS,interval=EVALUATION_INTERVAL):
    return sorted(set(range(0,steps+1,interval))|{steps})


def schedule(seed=SEED,steps=NUM_ITERATIONS):
    rng=torch.Generator().manual_seed(seed)
    return torch.cat([torch.randperm(50000,generator=rng).reshape(500,100)
        for _ in range((steps+499)//500)])[:steps].to(torch.int32)


def digest_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_digest(state):
    h=hashlib.sha256()
    for name,value in state.items():h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def update(net,x,y,architecture,method,i):
    assert x.shape==(100,3,32,32) and y.shape==(100,)
    e.sync(x.device);start=time.perf_counter();spec=METHODS[method];nan=float('nan')
    row=dict(architecture=architecture,method=method,iteration=i,learning_rate=lr_at(i),B=100,K=spec['K'],m=spec['m'],
        gamma=spec['gamma'] if spec['gamma'] is not None else nan,clipping_factor=1.,clipping_active=False,
        mean_block_gradient_norm=nan,median_block_gradient_norm=nan,mom_gradient_norm=nan,
        arithmetic_mean_gradient_norm=nan,cosine_similarity=nan)
    blocks=None
    if spec['K']:
        pairs=[e.gradient(net,a,b) for a,b in zip(x.split(20),y.split(20))]
        blocks=torch.stack([g for g,_ in pairs]);assert blocks.shape==(5,PARAMETERS[architecture])
        raw=blocks.mean(0);effective=coordinatewise_median(blocks);loss=torch.stack([loss for _,loss in pairs]).mean()
        norms=torch.stack([gradient_norm(g) for g in blocks]);rn,en=gradient_norm(raw),gradient_norm(effective)
        row.update(mean_block_gradient_norm=norms.mean().item(),median_block_gradient_norm=norms.median().item(),
            arithmetic_mean_gradient_norm=rn.item(),mom_gradient_norm=en.item(),
            cosine_similarity=(torch.dot(raw.double(),effective.double())/(rn*en)).item() if rn>0 and en>0 else nan)
    else:
        raw,loss=e.gradient(net,x,y)
        effective=e.normalize(raw) if method=='normalized' else clip_gradient(raw,spec['gamma']) if spec['gamma'] is not None else raw
        rn,en=gradient_norm(raw),gradient_norm(effective)
        if spec['gamma'] is not None:
            row.update(clipping_active=bool(rn>spec['gamma']),clipping_factor=min(1.,spec['gamma']/rn.item()) if rn>0 else 1.)
    row.update(minibatch_loss=loss.item(),raw_gradient_norm=rn.item(),effective_gradient_norm=en.item(),update_norm=lr_at(i)*en.item())
    if not (torch.isfinite(loss) and torch.isfinite(raw).all() and torch.isfinite(effective).all()
            and (blocks is None or torch.isfinite(blocks).all())):
        raise NumericalFailure(f'Nonfinite {architecture}/{method} update {i}',dict(row=row,
            raw_gradient=raw.detach().cpu(),effective_gradient=effective.detach().cpu(),
            block_gradients=blocks.detach().cpu() if blocks is not None else None))
    # Precheck candidate parameters before any mutation, preserving last valid state on overflow.
    offset=0
    for parameter in net.parameters():
        n=parameter.numel();g=effective[offset:offset+n].view_as(parameter);candidate=parameter.detach().clone()
        sgd_step_(candidate,g,lr_at(i))
        if not torch.isfinite(candidate).all():raise NumericalFailure('Nonfinite candidate parameters',dict(row=row))
        offset+=n
    offset=0
    for parameter in net.parameters():
        n=parameter.numel();sgd_step_(parameter,effective[offset:offset+n].view_as(parameter),lr_at(i));offset+=n
    e.sync(x.device);row['parameter_update_seconds']=time.perf_counter()-start
    return row


def evaluate(net,loader,device):
    rng=e.rng_state()
    try:return e.evaluate(net,loader,device)
    finally:e.restore_rng(rng)


def train_one(out,architecture,method,initial,training,loaders,indices,device,steps=NUM_ITERATIONS,resume=False,
              checkpoint_every=CHECKPOINT_EVERY,evaluation_interval=EVALUATION_INTERVAL):
    name=method;spec=METHODS[method];net=copy.deepcopy(initial)
    (out/'checkpoints').mkdir(exist_ok=True);(out/'progress').mkdir(exist_ok=True)
    path=out/'checkpoints'/(name+'.pt');timing_path=out/'checkpoints'/(name+'.runtime.json')
    protocol=dict(experiment='siopt_final_v1',architecture=architecture,method=method,K=spec['K'],m=spec['m'],B=100,gamma=spec['gamma'],steps=steps,
        initial_sha256=state_digest(initial.state_dict()),source_hashes=source_hashes(),
        learning_rate='0.1:1-4999;0.05:5000-9999;0.025:10000-15000',evaluation_interval=evaluation_interval,
        schedule_sha256=hashlib.sha256(indices.numpy().tobytes()).hexdigest())
    d,v,completed,prior_wall=[],[],0,0.
    timing=dict(checkpoint_save_seconds=0.,progress_export_seconds=0.)
    if resume and not path.exists() and (out/'progress'/(name+'_optimization.csv')).exists():
        raise FileNotFoundError('Progress exists without its checkpoint; refusing to restart completed updates')
    if resume and path.exists():
        state=torch.load(path,weights_only=False,map_location='cpu')
        if state['protocol']!=protocol:raise ValueError('Checkpoint configuration/schedule mismatch')
        net.load_state_dict(state['model']);e.restore_rng(state['rng'])
        d,v,completed,prior_wall=state['diagnostics'],state['evaluations'],state['iteration'],state['wall_seconds']
        timing=state['timing']
        assert state['next_schedule_row']==completed and state['next_learning_rate']==(lr_at(completed+1) if completed<NUM_ITERATIONS else None)
        assert [r['iteration'] for r in d]==list(range(1,completed+1))
        assert [r['iteration'] for r in v]==[i for i in evaluation_iterations(steps,evaluation_interval) if i<=completed]
        if timing_path.exists():
            recent=json.loads(timing_path.read_text())
            if recent['iteration']==completed and recent['protocol']==protocol:
                timing=recent['timing'];prior_wall=recent['wall_seconds']
    start=time.perf_counter()
    def assess(i):
        tr,tra,a=evaluate(net,loaders[0],device);te,tea,b=evaluate(net,loaders[1],device)
        v.append(dict(architecture=architecture,method=name,iteration=i,training_loss=tr,test_loss=te,training_accuracy=tra,test_accuracy=tea,evaluation_seconds=a+b))
        print(f'EVAL {name} {i}/{steps} train={tr:.9g} test={te:.9g}',flush=True)
    def save(i,status='running'):
        state=dict(schema_version=1,model=net.state_dict(),iteration=i,next_schedule_row=i,
            next_learning_rate=lr_at(i+1) if i<NUM_ITERATIONS else None,protocol=protocol,rng=e.rng_state(),
            diagnostics=d,evaluations=v,timing=dict(timing),wall_seconds=prior_wall+time.perf_counter()-start)
        before=time.perf_counter();e.atomic_save(path,state)
        timing['checkpoint_save_seconds']+=time.perf_counter()-before
        before=time.perf_counter()
        e.table(out/'progress'/(name+'_optimization.csv'),d);e.table(out/'progress'/(name+'_evaluation.csv'),v)
        timing['progress_export_seconds']+=time.perf_counter()-before
        wall=prior_wall+time.perf_counter()-start
        e.dump(timing_path,dict(iteration=i,protocol=protocol,timing=timing,wall_seconds=wall))
        e.dump(out/'progress'/(name+'.json'),dict(status=status,architecture=architecture,method=method,K=spec['K'],m=spec['m'],B=100,iteration=i,wall_seconds=wall,
            last_evaluation=v[-1] if v else None))
        print(f'CHECKPOINT {name} {i}/{steps} wall={wall:.1f}s',flush=True)
    iteration=completed
    try:
        if not v:assess(0);save(0)
        for iteration in range(completed+1,steps+1):
            e.sync(device);before=time.perf_counter()
            x,y=e.batch(training,indices[iteration-1],device);e.sync(device)
            data_seconds=time.perf_counter()-before
            digest=hashlib.sha256(indices[iteration-1].numpy().tobytes()).hexdigest()
            row=update(net,x,y,architecture,method,iteration);row['minibatch_index_sha256']=digest;row['data_preparation_seconds']=data_seconds
            d.append(row);completed=iteration
            if iteration in evaluation_iterations(steps,evaluation_interval):assess(iteration)
            if iteration%checkpoint_every==0 or iteration==steps:save(iteration,'completed' if iteration==steps else 'running')
    except (NumericalFailure,AssertionError) as exc:
        failure=dict(status='failed',method=name,K=spec['K'],attempted_iteration=iteration,last_completed_update=completed,
            message=str(exc),traceback=traceback.format_exc(),hyperparameters_unchanged=True)
        e.atomic_save(out/'progress'/(name+'_failure.pt'),dict(failure=failure,model=net.state_dict(),rng=e.rng_state(),
            indices=indices[min(max(iteration-1,0),len(indices)-1)],diagnostics=d,evaluations=v,timing=timing,wall_seconds=prior_wall+time.perf_counter()-start,detail=getattr(exc,'payload',{})))
        e.dump(out/'progress'/(name+'_failure.json'),failure)
        e.dump(out/'progress'/(name+'.json'),failure)
        print('FAILED '+json.dumps(failure),flush=True)
        return dict(diagnostics=d,evaluations=v,wall_seconds=prior_wall+time.perf_counter()-start,timing=timing,status=failure)
    return dict(diagnostics=d,evaluations=v,wall_seconds=prior_wall+time.perf_counter()-start,timing=timing,
                status=dict(status='completed',method=method,updates=completed))



def summary_rows(results,steps):
    summaries=[];runtimes=[];phases=[]
    for method,result in results.items():
        d,v=result['diagnostics'],result['evaluations'];spec=METHODS[method]
        assert [r['iteration'] for r in d]==list(range(1,len(d)+1))
        assert len({r['iteration'] for r in v})==len(v)
        assert all(r['method']==method and r['B']==100 and r['learning_rate']==lr_at(r['iteration']) for r in d)
        if result['status']['status']=='completed':
            assert len(d)==steps and [r['iteration'] for r in v]==evaluation_iterations(steps)
        final=v[-1] if v else {};window=[r for r in v if max(0,steps-1000)<=r['iteration']<=steps]
        finite_cos=[r for r in d if np.isfinite(r['cosine_similarity'])]
        maximum=max(d,key=lambda r:r['effective_gradient_norm']) if d else None
        median=float(np.median([r['effective_gradient_norm'] for r in d])) if d else float('nan')
        summaries.append(dict(architecture=final.get('architecture',''),method=method,status=result['status']['status'],updates=len(d),
            **{f'final_{k}':final.get(k,float('nan')) for k in ('training_loss','test_loss','training_accuracy','test_accuracy')},
            last_evaluation_iteration=final.get('iteration',-1),
            final_window_training_loss=float(np.mean([r['training_loss'] for r in window])) if window else float('nan'),
            final_window_test_loss=float(np.mean([r['test_loss'] for r in window])) if window else float('nan'),
            generalization_gap=final.get('test_loss',float('nan'))-final.get('training_loss',float('nan')),
            median_effective_gradient_norm=median,max_effective_gradient_norm=maximum['effective_gradient_norm'] if maximum else float('nan'),
            max_norm_iteration=maximum['iteration'] if maximum else -1,
            max_mom_gradient_norm=maximum['effective_gradient_norm'] if maximum and spec['K'] else float('nan'),
            min_cosine_similarity=min((r['cosine_similarity'] for r in finite_cos),default=float('nan')),
            min_cosine_iteration=min(finite_cos,key=lambda r:r['cosine_similarity'])['iteration'] if finite_cos else -1,
            median_cosine_similarity=float(np.median([r['cosine_similarity'] for r in finite_cos])) if finite_cos else float('nan'),
            large_gradient_relative_threshold=5*median,large_gradient_relative_count=sum(r['effective_gradient_norm']>5*median for r in d),
            large_gradient_absolute_threshold=10.,large_gradient_absolute_count=sum(r['effective_gradient_norm']>10 for r in d),
            clipping_frequency=float(np.mean([r['clipping_active'] for r in d])) if d and spec['gamma'] is not None else float('nan'),
            mean_clipping_factor=float(np.mean([r['clipping_factor'] for r in d])) if d else float('nan'),
            training_update_seconds=sum(r['parameter_update_seconds'] for r in d)))
        runtimes.append(dict(architecture=final.get('architecture',''),method=method,updates=len(d),status=result['status']['status'],
            training_update_seconds=sum(r['parameter_update_seconds'] for r in d),
            data_preparation_seconds=sum(r.get('data_preparation_seconds',0.) for r in d),
            evaluation_seconds=sum(r['evaluation_seconds'] for r in v),**result['timing'],total_method_wall_seconds=result['wall_seconds']))
        if spec['gamma'] is not None:
            for first,last in ((1,4999),(5000,9999),(10000,15000),(1,15000)):
                part=[r for r in d if first<=r['iteration']<=last]
                phases.append(dict(architecture=final.get('architecture',''),method=method,gamma=spec['gamma'],
                    phase_start=first,phase_end=last,updates=len(part),clipped_updates=sum(r['clipping_active'] for r in part),
                    clipping_frequency=float(np.mean([r['clipping_active'] for r in part])) if part else float('nan')))
    reference=next((r['training_update_seconds'] for r in runtimes if r['method']=='sgd'),0.)
    for r in runtimes:r['runtime_relative_to_sgd']=r['training_update_seconds']/reference if reference else float('nan')
    return summaries,runtimes,phases


def plot_results(out,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'figures';folder.mkdir(exist_ok=True)
    colors={n:plt.get_cmap('tab10')(i) for i,n in enumerate(METHODS)}
    def curves(ax,source,key,names=None):
        for name,result in results.items():
            if names is not None and name not in names:continue
            rows=result[source]
            if rows:ax.plot([r['iteration'] for r in rows],[r[key] for r in rows],label=name,color=colors[name],
                linestyle='--' if METHODS[name]['K'] else ':' if METHODS[name]['gamma'] is not None else '-',linewidth=1.)
        ax.set(xlabel='Parameter updates',ylabel=key.replace('_',' '));ax.legend(fontsize=7,ncol=2)
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(folder/f'{name}.{ext}',dpi=200)
        plt.close(fig)
    for number,key in ((1,'training_loss'),(2,'test_loss')):
        fig,ax=plt.subplots(figsize=(11,6));curves(ax,'evaluations',key);save(fig,f'figure_{number}')
    fig,axes=plt.subplots(2,1,figsize=(11,10))
    for ax,key in zip(axes,('training_accuracy','test_accuracy')):curves(ax,'evaluations',key)
    save(fig,'figure_3')
    fig,ax=plt.subplots(figsize=(11,6));curves(ax,'diagnostics','effective_gradient_norm');save(fig,'figure_4')
    fig,axes=plt.subplots(4,1,figsize=(12,17));_,_,phases=summary_rows(results,max((len(r['diagnostics']) for r in results.values()),default=0))
    clipped=[n for n in results if METHODS[n]['gamma'] is not None];positions=np.arange(len(clipped))
    for j,(first,last) in enumerate(((1,4999),(5000,9999),(10000,15000))):
        values=[next((r['clipping_frequency'] for r in phases if r['method']==n and r['phase_start']==first and r['phase_end']==last),float('nan')) for n in clipped]
        axes[0].bar(positions+(j-1)*.24,values,width=.24,label=f'{first}-{last}')
    axes[0].set_xticks(positions,clipped);axes[0].set_ylabel('Clipped fraction');axes[0].legend(fontsize=8)
    curves(axes[1],'diagnostics','mom_gradient_norm',['mom_K5_m20'])
    curves(axes[2],'diagnostics','arithmetic_mean_gradient_norm',['mom_K5_m20'])
    curves(axes[3],'diagnostics','cosine_similarity',['mom_K5_m20']);save(fig,'figure_5')
    fig,ax=plt.subplots(figsize=(12,6));ax.bar(list(results),[sum(r['parameter_update_seconds'] for r in result['diagnostics']) for result in results.values()],color=[colors[n] for n in results])
    ax.tick_params(axis='x',rotation=35);ax.set_ylabel('Training-update seconds');save(fig,'figure_6')


def export_results(out,results,steps,make_plots=True):
    before=time.perf_counter();summaries,runtimes,phases=summary_rows(results,steps)
    e.table(out/'optimization_diagnostics.csv',[r for result in results.values() for r in result['diagnostics']])
    evaluations=[r for result in results.values() for r in result['evaluations']]
    for file,fields in (('training_loss_history',['training_loss']),('test_loss_history',['test_loss']),('accuracy_history',['training_accuracy','test_accuracy'])):
        e.table(out/f'{file}.csv',[{k:r[k] for k in ['architecture','method','iteration']+fields} for r in evaluations])
    e.table(out/'runtime_summary.csv',runtimes);e.table(out/'run_summary.csv',summaries);e.table(out/'clipping_frequency_summary.csv',phases)
    csv_seconds=time.perf_counter()-before;before=time.perf_counter()
    if make_plots:plot_results(out,results)
    report=dict(csv_seconds=csv_seconds,figure_seconds=time.perf_counter()-before)
    e.dump(out/'reporting_runtime.json',report)
    return report


def cross_architecture(out,folders):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'cross_architecture_figures';folder.mkdir(exist_ok=True)
    available=[(a,p) for a,p in folders.items() if (p/'training_loss_history.csv').exists()]
    if len(available)!=2:return
    for filename,keys in (('training_loss_history',['training_loss']),('test_loss_history',['test_loss']),('accuracy_history',['training_accuracy','test_accuracy'])):
        fig,axes=plt.subplots(len(keys),2,figsize=(15,5*len(keys)),squeeze=False,sharex=True,sharey='row')
        for column,(architecture,path) in enumerate(available):
            with (path/f'{filename}.csv').open() as f:rows=list(e.csv.DictReader(f))
            for row_index,key in enumerate(keys):
                ax=axes[row_index,column]
                for j,method in enumerate(METHODS):
                    part=[r for r in rows if r['method']==method]
                    ax.plot([int(r['iteration']) for r in part],[float(r[key]) for r in part],label=method,color=plt.get_cmap('tab10')(j))
                ax.set(title=architecture.upper(),xlabel='Parameter updates',ylabel=key.replace('_',' '));ax.legend(fontsize=7,ncol=2)
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(folder/f'{filename}.{ext}',dpi=200)
        plt.close(fig)
    # Separate architecture panels prevent confusing absolute update costs between networks.
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    for ax,(architecture,path) in zip(axes,available):
        with (path/'runtime_summary.csv').open() as f:rows=list(e.csv.DictReader(f))
        ax.bar([r['method'] for r in rows],[float(r['training_update_seconds']) for r in rows])
        ax.set(title=architecture.upper(),ylabel='Training-update seconds');ax.tick_params(axis='x',rotation=45)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(folder/f'runtime.{ext}',dpi=200)
    plt.close(fig)


def peak_memory(device):
    peak=None
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ('peak','working','qpp','qp','qnpp','qnp','page','peakpage')]
        c=Counters();c.cb=ctypes.sizeof(c);ctypes.windll.kernel32.GetCurrentProcess.restype=wintypes.HANDLE
        fn=ctypes.windll.psapi.GetProcessMemoryInfo;fn.argtypes=[wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD]
        if fn(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(c),c.cb):peak=c.peak
    return dict(process_peak_working_set_bytes=peak,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)


def benchmark(out,architecture,initial,training,loaders,indices,device,setup_seconds):
    measured={};templates={};gather=[]
    for name in METHODS:
        values=[]
        for j in range(2):
            x,y=e.batch(training,indices[j*501],device);net=copy.deepcopy(initial)
            for _ in range(2):update(net,x,y,architecture,name,1)
            for _ in range(6):
                row=update(net,x,y,architecture,name,1);values.append(row['parameter_update_seconds'])
            templates[name]=row
        measured[name]=dict(mean_seconds=float(np.mean(values)),median_seconds=float(np.median(values)),raw_seconds=values)
        print('BENCHMARK',architecture,name,measured[name]['mean_seconds'],flush=True)
    for idx in indices[::499][:30]:
        e.sync(device);before=time.perf_counter();e.batch(training,idx,device);e.sync(device);gather.append(time.perf_counter()-before)
    # Small warmup then two measured complete train/test passes, all observations included.
    x,y=e.batch(training,indices[0],device);evaluate(initial,[(x,y)],device)
    ev=[]
    for _ in range(2):
        tr,tra,t=evaluate(initial,loaders[0],device);te,tea,u=evaluate(initial,loaders[1],device)
        ev.append(dict(training_loss=tr,test_loss=te,training_accuracy=tra,test_accuracy=tea,evaluation_seconds=t+u))
        print('FULL EVALUATION BENCHMARK',architecture,t+u,flush=True)
    io=out/'benchmark_io_synthetic';io.mkdir();synthetic={};costs={};io_samples={}
    for name in METHODS:
        d=[dict(templates[name],iteration=i,learning_rate=lr_at(i),minibatch_index_sha256='0'*64,data_preparation_seconds=float(np.mean(gather))) for i in range(1,15001)]
        v=[dict(ev[-1],architecture=architecture,method=name,iteration=i) for i in evaluation_iterations()]
        saves=[];exports=[];ages=(0,7500,15000)
        for age in ages:
            a=[];b=[]
            for _ in range(2):
                before=time.perf_counter();e.atomic_save(io/f'{name}.pt',dict(synthetic=True,model=initial.state_dict(),diagnostics=d[:age],evaluations=v,rng=e.rng_state()));a.append(time.perf_counter()-before)
                before=time.perf_counter();e.table(io/f'{name}.csv',d[:age]);b.append(time.perf_counter()-before)
            saves.append(a);exports.append(b)
        save_times=float(np.interp(range(0,15001,250),ages,[np.mean(x) for x in saves]).sum())
        export_times=float(np.interp(range(0,15001,250),ages,[np.mean(x) for x in exports]).sum())
        c=dict(training_update_seconds=15000*measured[name]['mean_seconds'],data_preparation_seconds=15000*float(np.mean(gather)),
            evaluation_seconds=31*float(np.mean([r['evaluation_seconds'] for r in ev])),checkpoint_save_seconds=save_times,progress_export_seconds=export_times)
        wall=sum(c.values());costs[name]=c
        synthetic[name]=dict(diagnostics=d,evaluations=v,timing=dict(checkpoint_save_seconds=save_times,progress_export_seconds=export_times),wall_seconds=wall,status=dict(status='completed'))
        io_samples[name]=dict(ages=ages,checkpoint_seconds=saves,csv_seconds=exports,checkpoint_bytes=(io/f'{name}.pt').stat().st_size)
    folder=out/'benchmark_export_synthetic';folder.mkdir();e.dump(folder/'SYNTHETIC_NOT_SCIENTIFIC.json',dict(synthetic=True,no_full_training=True))
    reporting=export_results(folder,synthetic,15000)
    # Real runs export aggregate CSVs after each method; figures once on final completion.
    aggregate_csv=8*reporting['csv_seconds'];figures=reporting['figure_seconds']
    for name,c in costs.items():c['aggregate_reporting_seconds_allocated']=(aggregate_csv+figures)/8;c['estimated_total_seconds']=sum(c.values())
    report=dict(architecture=architecture,methods=costs,update_measurements=measured,full_evaluation_measurements=ev,
        gather_seconds_per_batch=gather,checkpoint_measurements=io_samples,reporting_measurement=reporting,
        one_time_cache_setup_seconds=setup_seconds,estimated_architecture_seconds=setup_seconds+sum(c['estimated_total_seconds'] for c in costs.values()),
        schedule_bytes=indices.numel()*4,model_parameter_bytes=PARAMETERS[architecture]*4,
        estimated_checkpoint_disk_bytes=sum(v['checkpoint_bytes'] for v in io_samples.values()),
        measured_synthetic_export_disk_bytes=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()),
        **peak_memory(device),uncertainty='Two short rounds, each two warmups and six timed initial-model updates; two complete evaluations. Cache/thermal/CPU load and numerical trajectory change timings. Synthetic histories time I/O only; no full scientific training. Cross-architecture figures and small JSON/hash overhead are additional.')
    e.dump(out/'benchmark_summary.json',report)
    return report


def source_hashes():
    return {p.relative_to(ROOT).as_posix():e.file_sha256(p) for p in (Path(__file__).resolve(),ENGINE_PATH,ROOT/'src/mom.py',ROOT/'src/optimizers.py')}


def setup_data(cache_dir):
    transform=e.transforms.Compose([e.transforms.ToTensor(),lambda t:t.type(torch.get_default_dtype()),e.transforms.Normalize([.491,.482,.447],[.247,.243,.262])])
    a=e.datasets.CIFAR10(ROOT/'data',train=True,download=False,transform=transform);b=e.datasets.CIFAR10(ROOT/'data',train=False,download=False,transform=transform)
    train,train_info=e.cache_cifar(a,cache_dir,'train');test,test_info=e.cache_cifar(b,cache_dir,'test')
    assert len(train)==50000 and len(test)==10000
    return train,test,dict(train=train_info,test=test_info)


def validate(architecture,initial,training,testing,indices,device,out):
    assert sum(v.numel() for v in initial.parameters())==PARAMETERS[architecture]
    assert indices.shape==(15000,100) and torch.equal(indices,schedule(out['seed']))
    for epoch in indices.reshape(30,50000):assert torch.equal(epoch.sort().values,torch.arange(50000,dtype=torch.int32))
    nets=[copy.deepcopy(initial) for _ in METHODS]
    assert all(torch.equal(p,q) and p.data_ptr()!=q.data_ptr() for net in nets for p,q in zip(initial.parameters(),net.parameters()))
    x,y=e.batch(training,indices[0],device);assert x.shape==(100,3,32,32) and initial(x).shape==(100,10)
    blocks=torch.stack([e.gradient(initial,a,b)[0] for a,b in zip(x.split(20),y.split(20))])
    full,_=e.gradient(initial,x,y);torch.testing.assert_close(blocks.mean(0),full,rtol=5e-4,atol=5e-7)
    assert [lr_at(i) for i in (1,4999,5000,9999,10000,15000)]==[.1,.1,.05,.05,.025,.025]
    for gamma in CLIPPING_GAMMAS:assert gradient_norm(clip_gradient(full*1000,gamma))<=gamma+1e-6
    assert abs(gradient_norm(e.normalize(full)).item()-1.)<1e-6
    assert torch.equal(e.normalize(torch.zeros_like(full)),torch.zeros_like(full))
    rng=e.rng_state();before=state_digest(initial.state_dict());grads=[p.grad for p in initial.parameters()]
    loss,acc,_=evaluate(initial,[(x[:17],y[:17]),(x[17:23],y[17:23])],device)
    assert abs(loss-nn.functional.cross_entropy(initial(x[:23]),y[:23]).item())<1e-6
    assert before==state_digest(initial.state_dict()) and all(p.grad is g for p,g in zip(initial.parameters(),grads))
    assert torch.equal(rng['torch'],torch.get_rng_state()) and rng['python']==random.getstate() and np.array_equal(rng['numpy'][1],np.random.get_state()[1])
    return dict(status='passed',architecture=architecture,parameters=PARAMETERS[architecture],methods=list(METHODS),
        same_initial_parameters=True,schedule_sha256=digest_tensor(indices),epochs=30,dropped_observations=0,evaluation_checkpoints=31,
        batch_shape=[100,3,32,32],block_shape=list(blocks.shape),cache_counts=[len(training),len(testing)])


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('mode',choices=['preflight','run'])
    parser.add_argument('--architecture',choices=['cnn'],default='cnn')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight',type=Path)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--cache-dir',type=Path,default=ROOT/'data/cifar10_fixed_budget_float32_cache_v1')
    args=parser.parse_args();architectures=['fcn4','cnn'] if args.architecture=='both' else [args.architecture]
    if args.threads<1:parser.error('--threads must be positive')
    if args.mode=='preflight' and args.resume:parser.error('Use a new preflight output directory; preflight is not resumed')
    if args.mode=='run' and not args.resume and args.preflight is None:parser.error('--preflight is required')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(args.threads);torch.set_default_dtype(torch.float32);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    hashes=source_hashes();out=args.output.resolve();folders={a:out/f'{a}_seed{args.seed}' for a in architectures}
    # Refuse accidental overwrite before any data or training work.
    for architecture,folder in folders.items():
        if folder.exists() and not args.resume:raise FileExistsError(f'Output exists; use --resume: {folder}')
        if args.resume and not (folder/'config.json').exists():raise FileNotFoundError(f'No experiment config to resume: {folder}')
        if args.mode=='run' and not args.resume:
            pre=args.preflight/f'{architecture}_seed{args.seed}';prior=json.loads((pre/'config.json').read_text())
            assert prior['source_hashes']==hashes and prior['threads']==args.threads and prior['device']==args.device and prior['seed']==args.seed
            assert json.loads((pre/'validation.json').read_text())['status']=='passed'
            assert json.loads((pre/'smoke/status.json').read_text())['status']=='passed'
            assert (pre/'benchmark_summary.json').exists()
    start=time.perf_counter();training,testing,cache=setup_data(args.cache_dir);setup_seconds=time.perf_counter()-start
    device=torch.device(args.device);indices=schedule(args.seed);schedule_hash=digest_tensor(indices)
    loaders=[e.CachedEvaluationBatches(d,128) for d in (training,testing)]
    initials={};configs={}
    for architecture,folder in folders.items():
        # Reset RNG separately for each architecture; execution order cannot affect initialization.
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
        initial=model(architecture).to(device);initials[architecture]=initial
        identity=dict(experiment='siopt_final_v1',architecture=architecture,parameters=PARAMETERS[architecture],methods=METHODS,
            seed=args.seed,threads=args.threads,device=args.device,dtype='float32',source_hashes=hashes,
            torch=torch.__version__,torchvision=__import__('torchvision').__version__,numpy=np.__version__,
            schedule_sha256=schedule_hash,initial_sha256=state_digest(initial.state_dict()),
            cache_sha256={n:v['tensor_sha256'] for n,v in cache.items()})
        if args.resume:
            config=json.loads((folder/'config.json').read_text())
            if config['identity']!=identity:raise ValueError(f'Resume identity differs for {architecture}')
            saved=torch.load(folder/'minibatch_schedule.pt',weights_only=True)
            if not torch.equal(saved,indices):raise ValueError('Saved schedule differs')
            state=torch.load(folder/'initial_parameters.pt',weights_only=True,map_location='cpu')
            if state_digest(state)!=identity['initial_sha256']:raise ValueError('Initial parameter digest differs')
            initial.load_state_dict(state);configs[architecture]=config
        else:
            folder.mkdir(parents=True,exist_ok=False)
            e.atomic_save(folder/'minibatch_schedule.pt',indices);e.atomic_save(folder/'initial_parameters.pt',initial.state_dict())
            try:revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
            except (OSError,subprocess.SubprocessError):revision=None
            config=dict(identity=identity,mode=args.mode,source_hashes=hashes,seed=args.seed,threads=args.threads,device=args.device,
                architecture=repr(initial),parameters=PARAMETERS[architecture],methods=METHODS,num_iterations=15000,batch_size=100,
                learning_rate_boundaries={'1-4999':.1,'5000-9999':.05,'10000-15000':.025},
                evaluation_iterations=evaluation_iterations(),evaluation_batch_size=128,checkpoint_every=250,
                preprocessing={'mean':[.491,.482,.447],'std':[.247,.243,.262],'dtype':'float32','augmentation':False},
                cache=cache,hardware=platform.uname()._asdict(),python=sys.version,revision=revision,data_setup_seconds=setup_seconds,
                protocol='No momentum/weight decay/regularization; 30 full epochs, no discarded examples; shared ordered schedule; unsmoothed primary curves.',
                diagnostics='Strict >5 times within-method median effective norm and >10 absolute norm; diagnostic only; undefined cosine NaN.',
                interpretation='Single seed; report failures, generalization gaps and compute costs. Test loss is not an optimization certificate; no claim about infinite variance or theorem verification.',
                cnn_parameter_count_note='The exact requested bias-free layers contain 94304 parameters (864+18432+73728+1280), not 94368.')
            e.dump(folder/'config.json',config);(folder/'experiment_source.py').write_bytes(Path(__file__).read_bytes());configs[architecture]=config
    reports={}
    for architecture,folder in folders.items():
        initial=initials[architecture]
        if args.mode=='preflight':
            validation=validate(architecture,initial,training,testing,indices,device,configs[architecture]);e.dump(folder/'validation.json',validation)
            smoke=folder/'smoke';smoke.mkdir();e.dump(smoke/'SMOKE_NOT_SCIENTIFIC.json',dict(scientific_output=False,updates=2,evaluation_subset=[23,17]))
            mini=[e.CachedEvaluationBatches(e.CachedCIFAR(d.images[:n],d.targets[:n]),11) for d,n in ((training,23),(testing,17))]
            results={name:train_one(smoke,architecture,name,initial,training,mini,indices,device,steps=2,checkpoint_every=1) for name in METHODS}
            assert all(r['status']['status']=='completed' for r in results.values())
            export_results(smoke,results,2);e.dump(smoke/'status.json',dict(status='passed',methods=list(METHODS),scientific_output=False,updates=2))
            reports[architecture]=benchmark(folder,architecture,initial,training,loaders,indices,device,setup_seconds)
        else:
            before=time.perf_counter();results={};reporting=[]
            if not args.resume:
                report=json.loads((args.preflight/f'{architecture}_seed{args.seed}'/'benchmark_summary.json').read_text());e.dump(folder/'benchmark_summary.json',report)
            for method in METHODS:
                failure=folder/'progress'/f'{method}_failure.json'
                if args.resume and failure.exists():
                    failed=torch.load(failure.with_suffix('.pt'),weights_only=False)
                    results[method]=dict(diagnostics=failed['diagnostics'],evaluations=failed['evaluations'],wall_seconds=failed['wall_seconds'],timing=failed['timing'],status=failed['failure'])
                else:results[method]=train_one(folder,architecture,method,initial,training,loaders,indices,device,resume=args.resume)
                e.dump(folder/'status.json',{n:r['status'] for n,r in results.items()})
                reporting.append(export_results(folder,results,15000,make_plots=False))
            before_fig=time.perf_counter();plot_results(folder,results);figure_seconds=time.perf_counter()-before_fig
            complete=all(r['status']['status']=='completed' and len(r['diagnostics'])==15000 for r in results.values())
            e.dump(folder/'completion.json',dict(status='completed' if complete else 'completed_with_failures',all_eight_completed=complete,
                methods={n:r['status'] for n,r in results.items()},invocation_wall_seconds=time.perf_counter()-before,
                aggregate_reporting_seconds=sum(sum(r.values()) for r in reporting)+figure_seconds))
    if args.mode=='preflight':
        cross_architecture(out,{a:p/'smoke' for a,p in folders.items()})
        total=sum(r['estimated_architecture_seconds'] for r in reports.values())-(len(reports)-1)*setup_seconds
        e.dump(out/'benchmark_combined.json',dict(estimated_total_seconds=total,architectures={a:r['estimated_architecture_seconds'] for a,r in reports.items()},**peak_memory(device)))
        print('PREFLIGHT COMPLETE; NO FULL EXPERIMENT LAUNCHED. Estimated total hours:',total/3600,flush=True)
    else:cross_architecture(out,{a:out/f'{a}_seed{args.seed}' for a in ('fcn4','cnn')})


if __name__=='__main__':main()
