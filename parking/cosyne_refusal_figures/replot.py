"""Replot saved refusal results; writes only this unit's plots and cache."""

import ast
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[1]
STORE = ROOT / "parking/model_matrix/cache"
OUT = UNIT / "plots"
CACHE = UNIT / "cache"
MODELS = [("llama8b", "Llama-3.1-8B"), ("mistral7b", "Mistral-7B-v0.3"),
          ("qwen7b", "Qwen2.5-7B")]
METRICS = ["refusal", "retention", "fluency"]
COLORS = ["#980000", "midnightblue", "darkgreen"]
SOURCES = {}


def record(path):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def result(model, dataset, name):
    return record(STORE / f"{model}_{dataset}/results/prefill_logit/{name}.csv")


def save(fig, name):
    for ext in ("png", "pdf"):
        path = OUT / f"{name}.{ext}"
        if path.exists():
            path.unlink()
        fig.savefig(path, dpi=400, facecolor="white", transparent=False)
    plt.close(fig)


def layer_metrics():
    fig, axes = plt.subplots(3, 3, figsize=(7.5, 3.15))
    fig.subplots_adjust(left=.10, right=.91, bottom=.13, top=.91, wspace=.19, hspace=.24)
    tables = []
    for col, (model, label) in enumerate(MODELS):
        cols = ["label", "concept", "target", "question", "source_layer", "scale"] + [f"judge_{m}" for m in METRICS]
        d = pd.read_csv(result(model, "inhouse", "calibration_judged"), usecols=cols)
        d = d[d.label == "intervention"].copy()
        assert (d.concept == d.target).all()
        assert not d[cols].isna().any().any()
        assert not d.duplicated(["concept", "question", "source_layer", "scale"]).any()
        assert all(len(ast.literal_eval(s)) == 1 for s in d.source_layer.unique())
        d["layer"] = d.source_layer.map(lambda s: ast.literal_eval(s)[0])
        assert set(d.layer) == set(range(28 if model == "qwen7b" else 32))
        assert d.groupby(["layer", "scale"]).size().eq(100).all()
        assert set(d.scale) == set(np.arange(1, 11) * (10 if model == "qwen7b" else 1))
        rates = d.groupby(["layer", "scale"], as_index=False)[[f"judge_{m}" for m in METRICS]].mean()
        rates.insert(0, "model", model)
        rates["n"] = 100
        tables.append(rates)
        for row, metric in enumerate(METRICS):
            ax = axes[row, col]
            grid = rates.pivot(index="layer", columns="scale", values=f"judge_{metric}")
            assert not grid.isna().any().any()
            cm = LinearSegmentedColormap.from_list(metric, ["white", COLORS[row]])
            im = ax.imshow(grid.to_numpy(), origin="lower", aspect="auto", cmap=cm,
                           vmin=0, vmax=1, interpolation="none")
            ax.set_yticks([0, len(grid)-1], ["0", str(len(grid)-1)])
            ax.set_xticks([0, 9], [f"{grid.columns[0]:g}", f"{grid.columns[-1]:g}"])
            ax.tick_params(length=2, pad=1.5)
            if row == 0:
                ax.set_title(label, fontsize=10, fontweight="bold", pad=5)
            if row != 2:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Steering strength α", labelpad=1)
            if col == 0:
                ax.set_ylabel("Layer", labelpad=2)
                ax.text(-.34, .5, metric.capitalize(), transform=ax.transAxes,
                        rotation=90, ha="center", va="center", fontweight="bold", fontsize=10)
            if col == 2:
                pos = ax.get_position()
                cb = fig.colorbar(im, cax=fig.add_axes([.929, pos.y0, .012, pos.height]))
                cb.set_ticks([0, 1]); cb.ax.tick_params(length=2, pad=2)
                cb.set_label("Rate", labelpad=1, fontsize=9)
                cb.outline.set_visible(False)
    pd.concat(tables).to_csv(OUT / "layer_metrics.csv", index=False)
    save(fig, "layer_metrics")


def selectivity():
    fig, axes = plt.subplots(1, 4, figsize=(7.5, 1.85))
    fig.subplots_adjust(left=.13, right=.95, bottom=.24, top=.82, wspace=.12)
    labels = ["Bacteria", "Cats", "Chess", "Dogs", "Lasers", "Obama", "Paris", "People", "The Moon", "United States"]
    model_list = MODELS + [("phi4", "Phi-4")]
    matrices = []
    for i, ((model, title), ax) in enumerate(zip(model_list, axes)):
        cols = ["label", "concept", "target", "source_layer", "target_layer", "scale", "question", "judge_refusal"]
        d = pd.read_csv(result(model, "inhouse", "confusion_judged"), usecols=cols)
        d = d[d.label == "intervention"]
        assert d.scale.nunique() == 1 and d.source_layer.nunique() == 1
        assert not d.duplicated(["concept", "target", "question"]).any()
        concepts = list(d.concept.unique())
        assert [s.lower().replace('_', ' ') for s in concepts] == [s.lower() for s in labels]
        assert d.groupby(["concept", "target"]).size().eq(10).all()
        matrix = d.pivot_table(index="concept", columns="target", values="judge_refusal", aggfunc="mean").reindex(index=concepts, columns=concepts)
        im = ax.imshow(matrix, cmap="Greys", vmin=0, vmax=1, interpolation="none")
        ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
        ax.set_xticks(range(10), range(1, 11), fontsize=8)
        ax.set_yticks(range(10), [f"{k+1} {s}" for k, s in enumerate(labels)] if i == 0 else [""]*10, fontsize=8)
        ax.tick_params(length=0, pad=2)
        ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 10, 1), minor=True)
        ax.grid(which="minor", color="gray", linewidth=.35)
        ax.tick_params(which="minor", length=0)
        matrices.append(d.groupby(["concept", "target"], as_index=False).judge_refusal.mean().assign(model=model))
    fig.text(.55, .03, "Steering target (concept number)", ha="center", fontsize=9)
    cb = fig.colorbar(im, cax=fig.add_axes([.967, .24, .010, .58])); cb.set_ticks([0, 1]); cb.ax.tick_params(length=2, pad=1, labelsize=8)
    cb.outline.set_visible(False)
    pd.concat(matrices).to_csv(OUT / "selectivity.csv", index=False)
    save(fig, "selectivity")
    rows = []
    for model, _ in model_list:
        for dataset in ["inhouse", "mmlu", "rwku", "conceptvectors"]:
            d = pd.read_csv(result(model, dataset, "bars_judged"), usecols=["label", "concept", "target", "judge_refusal", "judge_retention"])
            d = d[d.label == "intervention"]
            target = d.concept == d.target
            rows.append(dict(model=model, dataset=dataset, target_refusal=d.loc[target, "judge_refusal"].mean(),
                             off_target_refusal=d.loc[~target, "judge_refusal"].mean(),
                             off_target_retention=d.loc[~target, "judge_retention"].mean()))
    pd.DataFrame(rows).to_csv(OUT / "evaluation_summary.csv", index=False)


def semantic_cloud():
    # Preserve the published UMAP, shells, highlighted pairs, and coordinates.
    source = record(ROOT / "figs/fig6_semantic_spillover/plots/figure1_paper.png")
    original = plt.imread(source)
    crop = [840, 215, 2420, 1520]
    x0, y0, x1, y1 = crop
    cloud = original[y0:y1, x0:x1]
    qpath = record(ROOT / "parking/semantic_spillover/plots/spillover_question_level.csv")
    ppath = record(ROOT / "parking/semantic_spillover/plots/spillover_pair_level_sentence_emb.csv")
    cached_q = pd.read_csv(qpath); cached_pairs = pd.read_csv(ppath)
    assert len(cached_q) == 2700 and len(cached_pairs) == 270
    assert (cached_q.concept != cached_q.target).all()
    question_frames, pair_frames, audit = [], [], []
    for model, _ in MODELS:
        columns = ["concept", "target", "question", "label", "judge_refusal"]
        keys = ["concept", "target", "question"]
        old = cached_q[cached_q.model == model]
        main = pd.read_csv(record(STORE / f"{model}_conceptvectors/results/main/confusion_judged.csv"), usecols=columns)
        main = main[(main.label == "intervention") & (main.concept != main.target)]
        old_match = old.merge(main, on=keys, validate="one_to_one", suffixes=("_cached", "_main"))
        assert len(old_match) == 900
        assert (old_match.judge_refusal_cached == old_match.judge_refusal_main).all()
        current = pd.read_csv(result(model, "conceptvectors", "confusion_judged"), usecols=columns)
        current = current[(current.label == "intervention") & (current.concept != current.target)]
        joined = old.drop(columns="judge_refusal").merge(current, on=keys, validate="one_to_one")
        assert len(joined) == 900 and not joined.judge_refusal.isna().any()
        question_frames.append(joined)
        rates = current.groupby(["target", "concept"], as_index=False).judge_refusal.mean().rename(columns={"concept":"source", "judge_refusal":"spillover"})
        pairs = cached_pairs[cached_pairs.model == model].drop(columns="spillover").merge(rates, on=["target", "source"], validate="one_to_one")
        assert len(pairs) == 90
        pair_frames.append(pairs)
        comparison = old.merge(current, on=keys, validate="one_to_one", suffixes=("_old", "_current"))
        audit.append({"model":model, "matched_rows":len(comparison), "changed_refusal_labels":int((comparison.judge_refusal_old != comparison.judge_refusal_current).sum())})
    q = pd.concat(question_frames, ignore_index=True)
    pairs = pd.concat(pair_frames, ignore_index=True)
    pairs.to_csv(OUT / "semantic_pairs.csv", index=False)
    median = float(q.sim_qst_to_target.median())
    q["near"] = q.sim_qst_to_target >= median
    stats = q.groupby(["model", "near"]).judge_refusal.agg(["mean", "count", "sem"]).reset_index()
    stats.to_csv(OUT / "semantic_near_far.csv", index=False)
    correlations = []
    for model, d in pairs.groupby("model"):
        rho, p = spearmanr(d.semantic_similarity, d.spillover)
        correlations.append(dict(model=model, n=len(d), spearman_rho=float(rho), p=float(p)))
    pd.DataFrame(correlations).to_csv(OUT / "semantic_correlations.csv", index=False)
    fig = plt.figure(figsize=(7.5, 3.0))
    ax = fig.add_axes([.005, .0, .51, .96]); ax.imshow(cloud); ax.axis("off")
    fig.text(.015, .96, "A", weight="bold", fontsize=11)
    fig.text(.515, .96, "B", weight="bold", fontsize=11)
    ax = fig.add_axes([.60, .18, .38, .63])
    for i, near in enumerate([False, True]):
        s = stats[stats.near == near].set_index("model").loc[[m[0] for m in MODELS]]
        ax.bar(np.arange(3)+(i-.5)*.34, s["mean"], width=.34, color="#980000" if near else "#777777",
               label="Near" if near else "Far", yerr=s["sem"], capsize=3,
               error_kw={"elinewidth":.8, "capthick":.8})
    ax.set_xticks(range(3), ["Llama-3.1\n8B", "Mistral\n7B-v0.3", "Qwen2.5\n7B"], fontsize=9)
    ax.set_ylim(0, .7); ax.set_yticks([0, .3, .6])
    ax.set_ylabel("Off-target refusal rate", fontsize=10, labelpad=4)
    ax.legend(loc="upper left", bbox_to_anchor=(0,1.26), ncol=2, fontsize=9)
    ax.spines["left"].set_position(("outward", 3)); ax.spines["bottom"].set_position(("outward", 3))
    save(fig, "semantic_cloud")
    (OUT / "semantic_provenance.json").write_text(json.dumps({"cloud_source":str(source.relative_to(ROOT)),
        "cloud_crop_xyxy":crop, "cloud_recomputed":False, "median_cosine":median,
        "near_far_rows_per_model":900, "pair_correlations":correlations,
        "rate_variant":"prefill_logit", "cached_original_rate_variant":"main", "rate_variant_audit":audit,
        "scope":"Cloud pools six datasets; near/far and pair tables use ten ConceptVectors concepts. External MiniLM embeddings, not LLM hidden states.",
        "limits":"Projection is illustrative; error bars are row-wise SEM, not cluster-adjusted confidence intervals."}, indent=2)+"\n")


def main():
    OUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"serif", "font.serif":["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset":"cm", "font.size":9, "axes.labelsize":9, "xtick.labelsize":8, "ytick.labelsize":8,
        "axes.spines.top":False, "axes.spines.right":False, "axes.linewidth":.65,
        "legend.frameon":False, "figure.dpi":120, "savefig.dpi":400})
    layer_metrics(); selectivity(); semantic_cloud()
    for path, digest in SOURCES.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest
    (OUT / "source_hashes.json").write_text(json.dumps(SOURCES, indent=2)+"\n")
    print(f"Wrote 3 figures and source summaries to {OUT}; all input hashes unchanged.")


if __name__ == "__main__":
    main()
