"""Non-scientific tests for three-seed analysis; no training functions called."""


import importlib.util


from pathlib import Path


import sys


import numpy as np


import pytest


FILE = Path(__file__).resolve().parents[1]/'experiments/cifar10_siopt_final_analysis.py'


spec = importlib.util.spec_from_file_location('three_seed_analysis_tests', FILE)


a = importlib.util.module_from_spec(spec)


spec.loader.exec_module(a)


def test_linear_quantiles_and_invalid_replicates():
    assert a.statistics([2,10,4]) == dict(median=4.,q1=3.,q3=7.,iqr=4.,minimum=2.,maximum=10.)
    assert a.statistics([7,7,7])['iqr'] == 0
    for values in ([1,2], [1,2,3,4], [1,np.nan,3], [1,np.inf,3]):
        with pytest.raises(ValueError): a.statistics(values)


@pytest.fixture
def synthetic():
    ev,di,sources = {},{},[]
    for j,seed in enumerate(a.SEEDS):
        ev[seed],di[seed] = {},{}
        for method in a.METHODS:
            # Seed order reverses at successive checkpoints: median is pointwise.
            ev[seed][method] = [dict(iteration=n,training_loss=1+abs(j-(i%3)),test_loss=2+abs(j-(i%3)),
                                    training_accuracy=.1*(j+1),test_accuracy=.15*(j+1)) for i,n in enumerate(a.GRID)]
            di[seed][method] = [dict(effective_gradient_norm=float(j+1),cosine_similarity=.2*(j+1),clipping_active=str(j==2))]*15000
            sources.append(dict(seed=seed,method=method,evaluation_file=f'synthetic/{seed}/{method}/evaluation.csv',optimization_file=f'synthetic/{seed}/{method}/optimization.csv'))
    return ev,di,sources


def test_seed_first_statistics_pointwise_medians_and_provenance(synthetic):
    curves,finals,summaries,gradient = a.aggregate(*synthetic)
    assert len(curves)==8*31*4 and len(finals)==24 and len(summaries)==8
    train=[r for r in curves if r['method']=='sgd' and r['metric']=='training_loss']
    assert [r['median'] for r in train[:3]] == [2.,2.,2.]
    assert [r['q3'] for r in train[:3]] == [2.5,2.,2.5]
    assert '2026' in train[0]['source_files'] and '123' in train[0]['source_files']
    row=next(r for r in gradient if r['method']=='clipped_8' and r['metric']=='clipping_activation_frequency')
    assert row['median']==0 and row['q3']==.5 and row['seed_2026']==1
    norm=next(r for r in gradient if r['method']=='sgd')
    assert norm['median']==2 and norm['q1']==1.5 and norm['q3']==2.5
    assert all(r['iqr']>=0 and r['minimum']<=r['median']<=r['maximum'] for r in curves+gradient)
    assert next(r for r in summaries if r['method']=='sgd')['median_final_training_loss']==train[-1]['median']


def test_undefined_cosines_counted_and_empty_windows_rejected(synthetic):
    ev,di,sources=synthetic
    di[42]['mom_K5_m20']=list(di[42]['mom_K5_m20'])
    di[42]['mom_K5_m20'][0]=dict(di[42]['mom_K5_m20'][0],cosine_similarity=float('nan'))
    gradient=a.aggregate(ev,di,sources)[3]
    r=next(r for r in gradient if r['metric']=='cosine_similarity')
    assert '499' in r['defined_counts']
    di[42]['mom_K5_m20'][:500]=[dict(cosine_similarity=float('nan'),effective_gradient_norm=1.)]*500
    with pytest.raises(ValueError,match='No defined'): a.aggregate(ev,di,sources)


def test_publication_figures_are_real_pdf_and_png(tmp_path,synthetic):
    curves,finals,_,gradient=a.aggregate(*synthetic)
    a.figures(tmp_path,curves,finals,gradient)
    pdf=list((tmp_path/'figures').glob('*.pdf'))
    png=list((tmp_path/'figures').glob('*.png'))
    assert len(pdf)==len(png)==5
    assert all(p.read_bytes().startswith(b'%PDF') for p in pdf)
    import struct
    for p in png:
        data=p.read_bytes()
        assert data[:8]==b'\x89PNG\r\n\x1a\n'
        location=data.index(b'pHYs')
        x,y,unit=struct.unpack('>IIB',data[location+4:location+13])
        assert unit==1 and min(x,y)*.0254 >= 300

