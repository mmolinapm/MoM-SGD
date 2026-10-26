"""Frozen three-seed CNN analysis. Never calls train_one or writes scientific inputs."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 2026)
METHODS = ('sgd', 'normalized', 'clipped_0.25', 'clipped_0.5', 'clipped_2', 'clipped_4', 'clipped_8', 'mom_K5_m20')
METRICS = ('training_loss', 'test_loss', 'training_accuracy', 'test_accuracy')
GRID = list(range(0, 15001, 500))
DEFAULT_OUTPUT = ROOT / 'results/cifar10_siopt_cnn_three_seeds_final'

def require(condition, message):
    if not condition:
        raise ValueError(message)

def statistics(values):
    a = np.asarray(values, dtype=float)
    require(a.shape == (3,) and np.isfinite(a).all(), 'Exactly three finite seed observations required')
    q1, median, q3 = np.quantile(a, [.25, .5, .75], method='linear')
    require(a.min() <= q1 <= median <= q3 <= a.max(), 'Invalid quantiles')
    return dict(median=float(median), q1=float(q1), q3=float(q3), iqr=float(q3-q1), minimum=float(a.min()), maximum=float(a.max()))

def aggregate(evaluations, diagnostics, sources):
    lookup = {(s['seed'],s['method']): s for s in sources}
    curves, finals, summaries, gradient = [], [], [], []
    def provenance(method, kind):
        return json.dumps([dict(seed=s, file=lookup[s,method][kind]) for s in SEEDS])
    for method in METHODS:
        for j, iteration in enumerate(GRID):
            for metric in METRICS:
                values = [float(evaluations[s][method][j][metric]) for s in SEEDS]
                curves.append(dict(method=method, iteration=iteration, metric=metric, **statistics(values), seeds='42;123;2026', source_files=provenance(method,'evaluation_file')))
        for s in SEEDS:
            finals.append(dict(seed=s, method=method, **{k:float(evaluations[s][method][-1][k]) for k in METRICS}, source_file=lookup[s,method]['evaluation_file']))
        row = dict(method=method)
        for metric in METRICS:
            st = statistics([float(evaluations[s][method][-1][metric]) for s in SEEDS])
            for key in ('median','iqr','q1','q3'):
                row[f'{key}_final_{metric}'] = st[key]
        row['source_files'] = provenance(method,'evaluation_file')
        summaries.append(row)
        keys = ['effective_gradient_norm']
        if method == 'mom_K5_m20': keys.append('cosine_similarity')
        if method.startswith('clipped_'): keys.append('clipping_activation_frequency')
        for first in range(1,15001,500):
            for metric in keys:
                values, counts = [], []
                for s in SEEDS:
                    part = diagnostics[s][method][first-1:first+499]
                    a = [float(r['clipping_active']=='True') if metric=='clipping_activation_frequency' else float(r[metric]) for r in part]
                    a = [v for v in a if math.isfinite(v)]
                    require(bool(a), 'No defined diagnostic values in seed/window')
                    counts.append(len(a))
                    values.append(float(np.mean(a)))
                gradient.append(dict(method=method, window_start=first, window_end=first+499, metric=metric,
                                     **statistics(values), **{f'seed_{s}':v for s,v in zip(SEEDS,values)},
                                     defined_counts=json.dumps(dict(zip(SEEDS,counts))), source_files=provenance(method,'optimization_file')))
    return curves, finals, summaries, gradient

def figures(out, curves, finals, gradient):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9, 'pdf.fonttype':42, 'ps.fonttype':42, 'axes.spines.top':False, 'axes.spines.right':False})
    folder = out/'figures'
    folder.mkdir()
    colors = dict(zip(METHODS, plt.get_cmap('tab10').colors))
    styles = {m:'--' if m=='mom_K5_m20' else ':' if m.startswith('clipped_') else '-.' if m=='normalized' else '-' for m in METHODS}
    def save(fig, name):
        fig.tight_layout()
        for ext in ('pdf','png'): fig.savefig(folder/f'{name}.{ext}', dpi=350, bbox_inches='tight')
        plt.close(fig)
    def plot(ax, rows, metric, xkey='iteration', names=METHODS):
        for method in names:
            a = [r for r in rows if r['method']==method and r['metric']==metric]
            if not a: continue
            x = [r[xkey] for r in a]
            ax.plot(x,[r['median'] for r in a],label=method,color=colors[method],linestyle=styles[method],linewidth=1.4)
            ax.fill_between(x,[r['q1'] for r in a],[r['q3'] for r in a],color=colors[method],alpha=.11,linewidth=0)
        ax.set(xlabel='Parameter updates', ylabel=metric.replace('_',' ').capitalize())
        ax.grid(alpha=.15)
        ax.legend(fontsize=7,ncol=2)
    for letter,metric in (('A','training_loss'),('B','test_loss')):
        fig,ax=plt.subplots(figsize=(7,4.4));plot(ax,curves,metric);save(fig,f'figure_{letter}_{metric}')
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),sharey=True)
    for ax,metric in zip(axes,('training_accuracy','test_accuracy')):
        plot(ax,curves,metric);ax.set_ylim(0,1)
    save(fig,'figure_C_accuracy')
    fig,axes=plt.subplots(3,1,figsize=(8,11))
    for ax,metric in zip(axes,('effective_gradient_norm','cosine_similarity','clipping_activation_frequency')):
        plot(ax,gradient,metric,'window_end')
        if metric=='clipping_activation_frequency': ax.set_ylim(0,1)
        if metric=='cosine_similarity': ax.set_ylim(-1,1)
    save(fig,'figure_D_gradient_diagnostics')
    fig,axes=plt.subplots(1,3,figsize=(13,4.5))
    for ax,metric in zip(axes,('training_loss','test_loss','test_accuracy')):
        for i,method in enumerate(METHODS):
            a=[next(r[metric] for r in finals if r['seed']==s and r['method']==method) for s in SEEDS]
            st=statistics(a)
            ax.vlines(i,st['q1'],st['q3'],color=colors[method],linewidth=5,alpha=.4)
            ax.scatter(i,st['median'],marker='_',s=180,color=colors[method])
            for offset,value,seed,marker in zip((-.17,0,.17),a,SEEDS,('o','s','^')):
                ax.scatter(i+offset,value,marker=marker,s=24,color=colors[method],edgecolors='black',linewidths=.4,label=f'Seed {seed}' if i==0 else None)
        ax.set_xticks(range(8),METHODS,rotation=60,ha='right')
        ax.set_ylabel('Final '+metric.replace('_',' '));ax.legend(fontsize=7);ax.grid(axis='y',alpha=.15)
    save(fig,'figure_E_final_performance')
