"""Classification, steering, spillover, and sparsity figures for this unit."""

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from transformers import AutoModel, AutoTokenizer

from forget.plot.plot import LAYER_CMAP_COLORS
from .core import CACHE, PLOTS, config, save_csv


RED, GREEN, GRAY = '#980000', '#38761d', '#333333'
CMAP = LinearSegmentedColormap.from_list('layers', LAYER_CMAP_COLORS)
RATINGS = ['axbench_concept_relevance', 'axbench_instruction_relevance', 'axbench_fluency', 'axbench_aggregate']


def style():
    sns.set_theme(style='ticks', context='paper')
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm',
        'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
        'figure.dpi': 160, 'savefig.dpi': 300, 'savefig.facecolor': 'white',
        'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})


def finish(fig, name, axes):
    for ax in np.asarray(axes).ravel():
        if ax.has_data() and not ax.images:
            sns.despine(ax=ax, offset=5, trim=True)
    fig.savefig(PLOTS / f'{name}.png', bbox_inches='tight', facecolor='white')
    fig.savefig(PLOTS / f'{name}.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def classification_figure(metrics):
    frame = metrics[metrics.split == 'test']
    fig, axes = plt.subplots(2, 2, figsize=(7, 4.8), layout='constrained')
    for ax, field, label in zip(axes.flat,
        ['auroc', 'balanced_accuracy', 'false_positive_rate', 'false_negative_rate'],
        ['AUROC', 'Balanced accuracy', 'False-positive rate', 'False-negative rate']):
        for _, group in frame.groupby('target'):
            curve = group.groupby('layer')[field].mean()
            ax.plot(curve.index, curve, color='#bbbbbb', lw=.7)
        curve = frame.groupby('layer')[field].mean()
        ax.plot(curve.index, curve, color=RED, lw=2)
        if field in ['auroc', 'balanced_accuracy']:
            ax.axhline(.5, color=GRAY, ls=':', lw=1)
        ax.set(xlabel='Layer', ylabel=label, ylim=(-.03, 1.03), xticks=[0, 15], yticks=[0, .5, 1])
    finish(fig, 'classification', axes)


def evaluation_summary(evaluation):
    frame = evaluation.assign(subset=np.where(evaluation.concept == evaluation.target, 'targeted', 'untargeted'))
    groups = frame.groupby(['label', 'subset'])
    return groups[RATINGS].mean().join(groups.size().rename('n')).reset_index()


def steering_figure(calibration, evaluation):
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5), layout='constrained')
    for ax, metric, label in zip(axes.flat,
                               ['axbench_concept_relevance', 'axbench_instruction_relevance', 'axbench_fluency'],
                               ['Refusal relevance', 'Instruction relevance', 'Fluency']):
        cells = calibration.groupby(['layer', 'scale'])[metric].mean().reset_index()
        for layer, group in cells.groupby('layer'):
            ax.plot(group.scale, group[metric], color=CMAP(layer / 15), lw=1)
        ax.set(xlabel='Strength', ylabel=label, ylim=(-.06, 2.06), xticks=[.5, 5], yticks=[0, 2])
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 15), cmap=CMAP),
                 ax=axes[0, :], label='Layer', ticks=[0, 15], shrink=.7)
    ax = axes[1, 1]
    summary = evaluation_summary(evaluation).set_index(['label', 'subset'])
    x = np.arange(3)
    for shift, condition, subset, color, label in [
            (-.3, 'baseline', 'targeted', '#aaaaaa', 'Baseline: targeted'),
            (-.1, 'intervention', 'targeted', RED, 'Targeted'),
            (.1, 'baseline', 'untargeted', '#dddddd', 'Baseline: untargeted'),
            (.3, 'intervention', 'untargeted', GRAY, 'Untargeted')]:
        ratings = summary.loc[(condition, subset), RATINGS[:3]].to_numpy(dtype=float)
        ax.bar(x + shift, ratings, width=.2, color=color, label=label)
    ax.set(xticks=x, xticklabels=['Refusal', 'Instruction', 'Fluency'], ylabel='AxBench rating', ylim=(0, 2.1), yticks=[0, 2])
    ax.legend(fontsize=6.5, loc='upper center', bbox_to_anchor=(.5, 1.02), ncol=2)
    ax.set_ylim(0, 2.7)
    finish(fig, 'steering', axes)


def matrix(ax, frame, value, title, concepts, vmax=2, cmap='Greys'):
    values = frame.groupby(['concept', 'target'])[value].mean().unstack().reindex(index=concepts, columns=concepts)
    image = ax.imshow(values, vmin=0, vmax=vmax, cmap=cmap, aspect='equal')
    ax.set(xticks=np.arange(10), xticklabels=[f'$c_{{{i+1}}}$' for i in range(10)],
        yticks=np.arange(10), yticklabels=[c.replace('_', ' ') for c in concepts], title=title,
        xlabel='Steering target', ylabel='Queried concept')
    ax.tick_params(axis='both', length=0, labelsize=8)
    return image


def spillover_figure(evaluation, predictions, concepts):
    frame = evaluation[evaluation.label == 'intervention']
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout='constrained')
    for ax, source, field, title, vmax in [(axes[0], frame, 'axbench_concept_relevance', 'Refusal relevance', 2),
            (axes[1], frame, 'axbench_instruction_relevance', 'Instruction relevance', 2),
            (axes[2], predictions, 'detector_positive', 'Concept detection', 1)]:
        im = matrix(ax, source, field, title, concepts, vmax=vmax)
        fig.colorbar(im, ax=ax, ticks=[0, vmax], fraction=.04, pad=.03)
    save_csv(frame.groupby(['concept', 'target']).size().rename('n').reset_index(), PLOTS / 'spillover_counts.csv')
    finish(fig, 'spillover', axes)


def sparsity_figure(metrics, predictions, bundles):
    test = metrics[metrics.split == 'test']
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5), layout='constrained')
    activation = predictions.groupby(['seed', 'layer', 'row_id']).activation_sparsity.first().groupby('layer').mean()
    axes[0, 0].plot(activation.index, activation, color=GRAY)
    axes[0, 0].set(xlabel='Layer', ylabel='Activation sparsity')
    detector = test.groupby('layer').coordinate_sparsity.mean()
    axes[0, 1].plot(detector.index, detector, color=RED, label='Concept direction')
    refusal = bundles.groupby('layer').refusal_sparsity.mean()
    axes[0, 1].plot(refusal.index, refusal, color=GRAY, label='Refusal direction')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].set(xlabel='Layer', ylabel='Direction sparsity')
    selective = test.groupby('layer').concept_sparsity.mean()
    axes[1, 0].plot(selective.index, selective, color=GREEN)
    axes[1, 0].set(xlabel='Layer', ylabel='Concept-response\nsparsity')
    im = axes[1, 1].scatter(test.concept_sparsity, test.false_positive_rate, c=test.layer, cmap=CMAP,
                             vmin=0, vmax=15, s=14, alpha=.5, linewidths=0)
    axes[1, 1].set(xlabel='Concept-response sparsity', ylabel='False-positive rate')
    fig.colorbar(im, ax=axes[1, 1], label='Layer', ticks=[0, 15])
    for ax in axes.flat:
        ax.set_ylim(-.03, 1.03)
        ax.set_yticks([0, 1])
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.set_xticks([0, 15])
    finish(fig, 'sparsity', axes)


@torch.no_grad()
def semantic_vectors(questions):
    path = CACHE / 'semantic_embeddings.pt'
    if path.exists():
        return torch.load(path, weights_only=True).numpy()
    name = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer, model = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval()
    vectors = []
    for start in range(0, len(questions), 32):
        batch = tokenizer(questions[start:start + 32], return_tensors='pt', padding=True, truncation=True, max_length=256)
        hidden = model(**batch).last_hidden_state
        mask = batch.attention_mask[:, :, None]
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        vectors.append(torch.nn.functional.normalize(pooled, dim=1))
    result = torch.cat(vectors)
    torch.save(result, path)
    return result.numpy()


def explanatory_tables(evaluation, predictions, selected, cfg):
    all_rows = pd.read_csv(CACHE / 'rows.csv', keep_default_na=False)
    semantic_rows = all_rows[all_rows.split == 'validation'].reset_index(drop=True)
    embeddings = semantic_vectors(semantic_rows.question.tolist())
    concepts = sorted(semantic_rows.concept.unique())
    centroids = np.stack([embeddings[semantic_rows.concept == c].mean(0) for c in concepts])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    semantic = centroids @ centroids.T
    records = []
    evaluation = evaluation[evaluation.label == 'intervention'].copy()
    for seed in cfg['seeds']:
        layer = selected[str(seed)]['layer']
        bundle = torch.load(CACHE / f'seed_{seed}/layer_{layer:02d}/bundle.pt', weights_only=True)
        overlap = (bundle['detectors'] @ bundle['detectors'].T).numpy()
        for i, concept in enumerate(concepts):
            for j, target in enumerate(concepts):
                sub = evaluation[(evaluation.seed == seed) & (evaluation.concept == concept) & (evaluation.target == target)]
                cls = predictions[(predictions.seed == seed) & (predictions.concept == concept) & (predictions.target == target)]
                records.append(dict(seed=seed, layer=layer, concept=concept, target=target,
                    semantic_similarity=semantic[i, j], direction_overlap=overlap[i, j],
                    refusal=sub.axbench_concept_relevance.mean(), instruction=sub.axbench_instruction_relevance.mean(),
                    n=len(sub),
                    detector_positive=cls.detector_positive.mean(), prefill_gate_fraction=sub.prefill_gate_fraction.mean(),
                    decode_gate_fraction=sub.decode_gate_fraction.mean()))
    pairs = pd.DataFrame(records)
    save_csv(pairs, PLOTS / 'concept_pairs.csv')
    return pairs


def explanatory_figure(pairs):
    off = pairs[pairs.concept != pairs.target].groupby(['concept', 'target']).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(7, 5), layout='constrained')
    for ax, x, y, xlabel, ylabel in [
        (axes[0, 0], 'semantic_similarity', 'detector_positive', 'Semantic similarity', 'False-positive rate'),
        (axes[0, 1], 'semantic_similarity', 'refusal', 'Semantic similarity', 'Off-target refusal\nrelevance'),
        (axes[1, 0], 'direction_overlap', 'detector_positive', 'Direction alignment', 'False-positive rate'),
        (axes[1, 1], 'direction_overlap', 'refusal', 'Direction alignment', 'Off-target refusal\nrelevance')]:
        ax.scatter(off[x], off[y], s=22, color=RED, alpha=.7, linewidths=0)
        maximum = 2 if y == 'refusal' else 1
        ax.set(xlabel=xlabel, ylabel=ylabel, ylim=(-.03, maximum + .03), yticks=[0, maximum])
    finish(fig, 'semantics_geometry', axes)


def make_figures():
    from .core import hoyer
    style()
    PLOTS.mkdir(parents=True, exist_ok=True)
    cfg = config()
    metrics, predictions, bundles = [], [], []
    for seed in cfg['seeds']:
        for layer in cfg['layers']:
            directory = CACHE / f'seed_{seed}' / f'layer_{layer:02d}'
            metrics.append(pd.read_csv(directory / 'classification_metrics.csv'))
            predictions.append(pd.read_csv(directory / 'classification.csv'))
            bundle = torch.load(directory / 'bundle.pt', weights_only=True)
            bundles.append(dict(seed=seed, layer=layer, refusal_sparsity=hoyer(bundle['refusal']), amplitude=bundle['amplitude']))
    metrics, predictions, bundles = pd.concat(metrics), pd.concat(predictions), pd.DataFrame(bundles)
    evaluation = pd.read_csv(CACHE / 'evaluation_judged.csv')
    calibration = pd.read_csv(CACHE / 'calibration_judged.csv', keep_default_na=False)
    selected = json.loads((CACHE / 'selected.json').read_text())
    selected_predictions = pd.concat([predictions[(predictions.seed == s) &
        (predictions.layer == selected[str(s)]['layer']) & (predictions.split == 'test')] for s in cfg['seeds']])
    concepts = sorted(evaluation.concept.unique())
    save_csv(metrics, PLOTS / 'classification_metrics.csv')
    save_csv(bundles, PLOTS / 'direction_sparsity.csv')
    save_csv(calibration.groupby(['seed', 'layer', 'scale'])[RATINGS].mean().reset_index(), PLOTS / 'steering_calibration.csv')
    save_csv(evaluation_summary(evaluation), PLOTS / 'evaluation_summary.csv')
    classification_figure(metrics)
    steering_figure(calibration, evaluation)
    spillover_figure(evaluation, selected_predictions, concepts)
    sparsity_figure(metrics, predictions[predictions.split == 'test'], bundles)
    pairs = explanatory_tables(evaluation, selected_predictions, selected, cfg)
    explanatory_figure(pairs)


if __name__ == '__main__':
    make_figures()
