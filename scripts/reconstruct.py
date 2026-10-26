"""Reconstruct frozen paper figures/tables; never train or download."""
import argparse,csv,gzip,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import pareto_d10_final as p
from experiments import cifar10_siopt_final_analysis as a
def read(path):
 with gzip.open(path,'rt',newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def csvout(path,rows):
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def cnn(out):
 for item in json.loads((ROOT/'provenance/cnn_inputs.json').read_text(encoding='utf-8-sig')):
  with gzip.open(ROOT/item['published_file'],'rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
  if digest!=item['original_sha256']:raise ValueError('Input hash mismatch: '+item['published_file'])
 ev,di,sources={},{},[]
 for seed in a.SEEDS:
  ev[seed],di[seed]={},{};folder=ROOT/f'results/cnn/seed{seed}'
  config=json.loads((folder/'config.json').read_text(encoding='utf-8-sig'));complete=json.loads((folder/'completion.json').read_text(encoding='utf-8-sig'))
  a.require(config['seed']==seed and config['parameters']==94304,'Seed/architecture mismatch')
  a.require(complete['all_eight_completed'],'Incomplete seed')
  for method in a.METHODS:
   ep=folder/f'{method}_evaluation.csv.gz';dp=folder/f'{method}_optimization.csv.gz'
   er,dr=read(ep),read(dp)
   a.require([int(x['iteration']) for x in er]==a.GRID,'Evaluation grid mismatch')
   a.require([int(x['iteration']) for x in dr]==list(range(1,15001)),'Update grid mismatch')
   for row in er+dr:a.require(row['architecture']=='cnn' and row['method']==method,'Mixed rows')
   for row in er:
    for metric in a.METRICS:
     value=float(row[metric]);a.require(math.isfinite(value) and value>=0 and ('accuracy' not in metric or value<=1),'Invalid metric')
   ev[seed][method],di[seed][method]=er,dr
   sources.append(dict(seed=seed,method=method,evaluation_file=ep.relative_to(ROOT).as_posix(),optimization_file=dp.relative_to(ROOT).as_posix()))
 curves,finals,summaries,gradient=a.aggregate(ev,di,sources)
 for name,metrics in (('aggregated_training_loss',('training_loss',)),('aggregated_test_loss',('test_loss',)),('aggregated_accuracy',('training_accuracy','test_accuracy'))):
  csvout(out/f'{name}.csv',[r for r in curves if r['metric'] in metrics])
 for name,rows in (('final_per_seed_metrics',finals),('final_median_iqr_summary',summaries),('gradient_diagnostics_summary',gradient)):csvout(out/f'{name}.csv',rows)
 for ref in (ROOT/'results/cnn/reference').glob('*.csv'):
  with ref.open(newline='',encoding='utf-8') as f:expected=list(csv.DictReader(f))
  with (out/ref.name).open(newline='',encoding='utf-8') as f:actual=list(csv.DictReader(f))
  assert len(actual)==len(expected)
  for x,y in zip(actual,expected):
   for key in x:
    if key.startswith('source_'):continue
    assert x[key]==y[key],(ref.name,key,x[key],y[key])
 a.figures(out,curves,finals,gradient)
def pareto(out):
 import matplotlib.pyplot as plt
 plt.rcdefaults()
 p.validate_settings()
 input_path=ROOT/'results/pareto/plot_inputs.json.gz'
 expected=json.loads((ROOT/'MANIFEST.sha256.json').read_text())['results/pareto/plot_inputs.json.gz']
 assert hashlib.sha256(input_path.read_bytes()).hexdigest()==expected,'Pareto input hash mismatch'
 with gzip.open(input_path,'rt',encoding='utf-8') as f:data=json.load(f)
 records,histories=data['records'],data['histories']
 records={p.run_id(config):records[p.run_id(config)] for config in p.configurations()}
 assert len(records)==420 and set(records)==set(histories)
 for mode in p.COMPARISON_MODES:
  rows=[r['analysis_summaries'][mode] for r in records.values()];csvout(out/f'{mode}.csv',rows)
  aggregates=p.aggregate(rows);csvout(out/f'five_seed_aggregate_{mode}.csv',aggregates)
  with (ROOT/f'results/pareto/reference/five_seed_aggregate_{mode}.csv').open(newline='',encoding='utf-8') as f:reference=list(csv.DictReader(f))
  assert len(reference)==len(aggregates)
  for actual,expected in zip(aggregates,reference):
   for key,value in actual.items():
    if isinstance(value,float):
     target=float(expected[key]);assert value==target or math.isnan(value) and math.isnan(target),(mode,key)
    else:assert ('' if value is None else str(value))==expected[key],(mode,key)
 p.plot_final(out,records,histories)
def main():
 parser=argparse.ArgumentParser(__doc__);parser.add_argument('experiment',choices=('cnn','pareto','all'))
 parser.add_argument('--output',type=Path,default=ROOT/'reconstructed');args=parser.parse_args()
 if args.output.exists():parser.error('Output exists; select a new directory')
 for name in (('cnn','pareto') if args.experiment=='all' else (args.experiment,)):
  out=args.output/name;out.mkdir(parents=True);globals()[name](out)
  print(name+': reference comparisons and figure generation passed',flush=True)
if __name__=='__main__':main()
