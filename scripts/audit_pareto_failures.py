"""Audit preserved Pareto records without training, gradients or datasets."""
import argparse,collections,csv,gzip,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FLAGS=('theta_nonfinite','gradient_nonfinite','loss_nonfinite','prediction_nonfinite','residual_nonfinite')
def export(path,rows):
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def make_reports(root,out):
 path=root/'results/pareto/plot_inputs.json.gz'
 manifest=json.loads((root/'MANIFEST.sha256.json').read_text(encoding='utf-8'))
 if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['results/pareto/plot_inputs.json.gz']:raise ValueError('Pareto input hash mismatch')
 with gzip.open(path,'rt',encoding='utf-8') as f:data=json.load(f)
 records=list(data['records'].values());failed=sorted((r for r in records if r['status']=='failed'),key=lambda r:(r['schedule'],r['reference_k'],r['seed_id']))
 assert len(records)==420 and len(failed)==24
 with (root/'results/pareto/failed_terminal_rows.csv').open(newline='',encoding='utf-8') as f:terminal={r['run_id']:r for r in csv.DictReader(f)}
 assert set(terminal)=={r['run_id'] for r in failed}
 out.mkdir(parents=True,exist_ok=False)
 per_run=[]
 for r in failed:
  row=terminal[r['run_id']]
  assert row['finite']=='False' and row['failure_reason']==r['failure_reason'] and int(row['iteration'])==r['failure_iteration']
  for flag in FLAGS:assert (row[flag]=='True')==r[flag]
  assert math.isnan(r['final_parameter_error']) and math.isnan(r['final_loss'])
  per_run.append({k:r[k] for k in ('run_id','schedule','regime_variant','method','reference_k','k','gamma','seed_id','data_seed','training_seed','failure_iteration','successful_updates','failure_reason','last_finite_parameter_error','maximum_parameter_error',*FLAGS,'trajectory_sha256')})
 export(out/'failed_runs.csv',per_run)
 groups=collections.defaultdict(list)
 for r in failed:groups[(r['schedule'],r['regime_variant'],r['method'],r['reference_k'],r['k'],r['gamma'],r['failure_reason'])].append(r)
 grouped=[]
 for key,rows in sorted(groups.items(),key=lambda x:str(x[0])):
  grouped.append(dict(zip(('schedule','regime_variant','method','reference_k','mom_k','clipping_threshold','failure_reason'),key),
    failed_count=len(rows),planned_seeds=5,completed_count=5-len(rows),seed_ids=';'.join(str(r['seed_id']) for r in rows),
    failure_iterations=';'.join(str(r['failure_iteration']) for r in rows),
    last_finite_error_min=min(r['last_finite_parameter_error'] for r in rows),last_finite_error_max=max(r['last_finite_parameter_error'] for r in rows),
    classification='numerical nonfiniteness; no mathematical divergence claim'))
 export(out/'failures_by_configuration.csv',grouped)
 denominators=[]
 for mode in ('equal_iterations','equal_samples'):
  with (root/f'results/pareto/reference/five_seed_aggregate_{mode}.csv').open(newline='',encoding='utf-8') as f:reference=list(csv.DictReader(f))
  for row in reference:
   matching=[r['analysis_summaries'][mode] for r in records if r['schedule']==row['schedule'] and r['method']==row['method'] and r['reference_k']==int(row['reference_k']) and (r['gamma'] is None if not row['gamma'] else r['gamma']==float(row['gamma']))]
   completed=[r for r in matching if r['status']=='completed' and r['finite'] is True]
   assert len(matching)==5 and len(completed)==int(row['n_completed'])
   for metric in ('final_parameter_error','final_loss','maximum_parameter_error','wall_clock_seconds','clipping_fraction'):
    values=[r[metric] for r in completed] if metric!='clipping_fraction' or row['method']=='clipped' else []
    assert all(isinstance(v,(int,float)) and math.isfinite(v) for v in values)
    assert len(values)==int(row['n_finite_'+metric])
   denominators.append({k:row[k] for k in ('comparison_mode','schedule','regime_variant','method','reference_k','gamma','n_seeds_total','n_completed','n_failed','n_physical_failed','completion_rate','n_finite_final_parameter_error','n_finite_final_loss','n_finite_maximum_parameter_error','n_finite_wall_clock_seconds','n_finite_clipping_fraction','statistics_condition')})
 export(out/'summary_denominators.csv',denominators)
 cells=collections.defaultdict(list)
 for r in records:cells[(r['schedule'],r['method'],r['reference_k'],r['gamma'])].append(r)
 intervals=[]
 for key,rows in sorted(cells.items(),key=lambda x:str(x[0])):
  assert len(rows)==5
  for metric in ('theta_error','loss'):
   previous=None;start=1
   for n in range(1,3002):
    count=None
    if n<=3000:
     count=0
     for r in rows:
      h=data['histories'][r['run_id']]
      if len(h)>=n:
       row=h[n-1];value=row[metric]
       if row['finite'] and value is not None and math.isfinite(value) and value>0:count+=1
    if count!=previous:
     if previous is not None:intervals.append(dict(schedule=key[0],method=key[1],reference_k=key[2],gamma=key[3],metric=metric,iteration_start=start,iteration_end=n-1,n_planned=5,n_used=previous,condition='available finite positive values at this iteration'))
     start=n;previous=count
 export(out/'trajectory_denominators.csv',intervals)
 return dict(terminal_records=len(records),failed_runs=len(failed),failure_groups=len(grouped),denominator_rows=len(denominators),trajectory_intervals=len(intervals))
def main():
 parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 print(json.dumps(make_reports(ROOT,args.output),indent=2))
if __name__=='__main__':main()
