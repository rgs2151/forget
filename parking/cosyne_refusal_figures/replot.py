"""Replot saved refusal results; writes only this unit's plots and cache."""

import ast
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymupdf
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, ConnectionPatch
from matplotlib.transforms import Affine2D
from scipy.stats import spearmanr
from forget.plot.plot import custom_cmap

UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[1]
STORE = ROOT / "parking/model_matrix/cache"
OUT = UNIT / "plots"
CACHE = UNIT / "cache"
MODELS = [("llama8b", "Llama-3.1-8B"), ("mistral7b", "Mistral-7B-v0.3"),
          ("qwen7b", "Qwen2.5-7B")]
METRICS = ["refusal", "retention"]
SOURCES = {}
LAYOUT = {}


def record(path):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def result(model, dataset, name):
    return record(STORE / f"{model}_{dataset}/results/prefill_logit/{name}.csv")


def save(fig, name):
    fig.canvas.draw()
    text_sizes = sorted({t.get_fontsize() for t in fig.findobj(matplotlib.text.Text)
                         if t.get_visible() and t.get_text().strip()})
    LAYOUT[name] = {"width_inches":float(fig.get_figwidth()),
                    "height_inches":float(fig.get_figheight()),
                    "font_sizes_pt":text_sizes}
    for ext in ("png", "pdf"):
        path = OUT / f"{name}.{ext}"
        if path.exists():
            path.unlink()
        fig.savefig(path, dpi=400, facecolor="white", transparent=False)
    plt.close(fig)


def layer_metrics():
    record(ROOT / "forget/plot/plot.py")
    fig = plt.figure(figsize=(5.1, 2.7))
    axes = np.array([[fig.add_axes([x/5.1,y/2.7,.98/5.1,.98/2.7])
                      for x in [.34,2.00,3.66]] for y in [1.52,.30]])
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
        cmap = custom_cmap(d.layer.nunique())
        norm = Normalize(vmin=0, vmax=int(d.layer.max()))
        for row, metric in enumerate(METRICS):
            ax = axes[row, col]
            for layer, trace in rates.groupby("layer"):
                ax.plot(trace.scale, trace[f"judge_{metric}"], color=cmap(norm(layer)),
                        linewidth=.7, alpha=1.0)
            mx = float(rates.scale.max())
            ax.set_xlim(0, mx); ax.set_xticks([0, mx])
            ax.set_ylim(-.035, 1.035); ax.set_yticks([0, 1])
            ax.spines['left'].set_bounds(0, 1)
            ax.spines['bottom'].set_bounds(0, mx)
            ax.spines['left'].set_position(('outward',1.5))
            ax.spines['bottom'].set_position(('outward',1.5))
            ax.tick_params(length=2, pad=1.5)
            if row == 0:
                ax.set_title(label, fontsize=9, fontweight="bold", pad=4)
            if row == 0:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Scale $s$", labelpad=1)
            if col == 0:
                ax.set_ylabel(metric.capitalize(), labelpad=3, fontsize=8.5)
            else:
                ax.tick_params(labelleft=False)
            pos = ax.get_position()
            cb = fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),
                             cax=fig.add_axes([pos.x1+.045/5.1,pos.y0,.035/5.1,pos.height]))
            cb.set_ticks([0,int(d.layer.max())]);cb.ax.tick_params(length=1.5,pad=1)
            cb.set_label('Layer',rotation=270,labelpad=7,fontsize=8.5)
            cb.outline.set_visible(False)
    pd.concat(tables).to_csv(OUT / "layer_metrics.csv", index=False)
    save(fig, "layer_metrics")
    LAYOUT['layer_metrics'].update({'subplot_inches':[.98,.98],
                                    'row_gap_inches':.24,'colorbars':6})


def draw_selectivity_schematic(fig, original):
    # Matrix and bars have independent axes, with the matrix exactly the same
    # physical size and vertical alignment as each measured model matrix.
    size=1.10; left=.20; bottom=.30
    ax=fig.add_axes([left/7.5,bottom/1.65,size/7.5,size/1.65])
    ax.imshow(np.eye(5)*.60,cmap='Greys',vmin=0,vmax=1,interpolation='none')
    ax.set_xlim(-.5,4.5);ax.set_ylim(4.5,-.5)
    ax.set_xticks(np.arange(-.5,5),minor=True);ax.set_yticks(np.arange(-.5,5),minor=True)
    ax.grid(which='minor',color='#dddddd',lw=.45)
    ax.tick_params(which='both',length=0,labelbottom=False,labelleft=False)
    for spine in ax.spines.values():spine.set_visible(False)
    for k in range(5):
        icon=original[83+k*53:130+k*53,176:223].copy()
        red=(icon[:,:,0]>1.3*icon[:,:,1]) & (icon[:,:,0]>1.3*icon[:,:,2])
        icon[red,:3]=1.0
        for x,y in [(.035,bottom+size*(4.5-k)/5-.065),
                    (left+size*(k+.5)/5-.065,.14)]:
            ia=fig.add_axes([x/7.5,y/1.65,.13/7.5,.13/1.65])
            ia.imshow(icon);ia.axis('off')
    ax.add_patch(FancyBboxPatch((-.60,-.60),np.sqrt(32)+1.2,1.2,
        boxstyle='round,pad=0,rounding_size=.6',fc='none',ec='#980000',lw=1.1,
        transform=Affine2D().rotate_deg(45)+ax.transData,clip_on=False))
    ax.plot([.80,4.45,4.45,.80],[-.45,-.45,3.2,-.45],color='black',lw=1)
    ax.set_title('A  Targeted refusal',fontsize=9,fontweight='bold',pad=6)
    bars=fig.add_axes([1.42/7.5,bottom/1.65,.34/7.5,size/1.65])
    bars.bar([0,1],[.90,.10],width=.65,color=['#980000','black'])
    bars.set_xlim(-.6,1.6);bars.set_ylim(0,1)
    bars.set_xticks([0,1],['Target','Other'],rotation=55,ha='right',fontsize=8)
    bars.set_yticks([]);bars.tick_params(length=0,pad=2)
    for spine in bars.spines.values():spine.set_visible(False)
    fig.add_artist(ConnectionPatch(xyA=(4.5,3.3),coordsA=ax.transData,
        xyB=(-.35,.24),coordsB=bars.transData,arrowstyle='->',mutation_scale=7,
        color='#980000',lw=1,clip_on=False))
    arrow_x=1.42+.34*(1+.6)/2.2
    source_x=left+size*(2.4+.5)/5
    fig.add_artist(plt.Line2D([source_x/7.5,source_x/7.5,arrow_x/7.5],
        [(bottom+size)/1.65,1.425/1.65,1.425/1.65],transform=fig.transFigure,color='black',lw=.9))
    fig.add_artist(FancyArrowPatch((arrow_x/7.5,1.425/1.65),
        (arrow_x/7.5,(bottom+.11*size)/1.65),transform=fig.transFigure,
        arrowstyle='->',mutation_scale=7,color='black',lw=.9))


def selectivity():
    fig = plt.figure(figsize=(7.5, 1.65))
    original = plt.imread(record(UNIT / "inputs/Figure4_original.png"))
    draw_selectivity_schematic(fig, original)
    axes = [fig.add_axes([left/7.5,.30/1.65,1.10/7.5,1.10/1.65])
            for left in [2.53,3.70,4.87,6.04]]
    labels = ["bacteria", "cats", "chess", "dogs", "lasers", "obama", "paris", "people", "the moon", "USA"]
    model_list = MODELS + [("phi4", "Phi-4")]
    matrices = []
    for i, ((model, title), ax) in enumerate(zip(model_list, axes)):
        cols = ["label", "concept", "target", "source_layer", "target_layer", "scale", "question", "judge_refusal"]
        d = pd.read_csv(result(model, "inhouse", "confusion_judged"), usecols=cols)
        d = d[d.label == "intervention"]
        assert d.scale.nunique() == 1 and d.source_layer.nunique() == 1
        assert not d.duplicated(["concept", "target", "question"]).any()
        concepts = list(d.concept.unique())
        assert [s.lower().replace('_', ' ') for s in concepts] == labels[:-1]+['united states']
        assert d.groupby(["concept", "target"]).size().eq(10).all()
        matrix = d.pivot_table(index="concept", columns="target", values="judge_refusal", aggfunc="mean").reindex(index=concepts, columns=concepts)
        im = ax.imshow(matrix, cmap="Greys", vmin=0, vmax=1, interpolation="none")
        ax.set_title(chr(66+i)+'  '+title, fontsize=9, fontweight="bold", pad=6)
        ax.set_xticks([0,9], ['$c_1$','$c_{10}$'], fontsize=8)
        ax.set_yticks(range(10), [f"{s} ($c_{{{k+1}}}$)" for k, s in enumerate(labels)] if i == 0 else [""]*10, fontsize=8)
        ax.tick_params(length=0, pad=2)
        ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 10, 1), minor=True)
        ax.grid(which="minor", color="gray", linewidth=.35)
        ax.tick_params(which="minor", length=0)
        matrices.append(d.groupby(["concept", "target"], as_index=False).judge_refusal.mean().assign(model=model))
    axes[0].set_xlabel('Steered concepts',fontsize=8.5,labelpad=-1)
    cb = fig.colorbar(im, cax=fig.add_axes([7.22/7.5,.30/1.65,.045/7.5,1.10/1.65])); cb.set_ticks([0, 1]); cb.ax.tick_params(length=1.5,pad=1,labelsize=8)
    cb.outline.set_visible(False)
    pd.concat(matrices).to_csv(OUT / "selectivity.csv", index=False)
    save(fig, "selectivity")
    LAYOUT['selectivity']['matrix_inches']=[1.10,1.10]
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


def draw_original_cloud(fig):
    source = record(ROOT / "figs/fig6_semantic_spillover/plots/figure1_paper.pdf")
    pdf = pymupdf.open(source)
    page = pdf[0]
    clip = pymupdf.Rect(201.6, 51.6, 580.8, 280.0)
    # The last PDF artists are the four callout lines and six text boxes.
    # Remove only those artists, then redraw their labels at the final print size.
    stream = page.read_contents()
    marker = b'/A13 gs 1 J 1.6 w'
    assert stream.count(marker) == 1
    pdf.update_stream(page.get_contents()[0], stream.split(marker)[0] + b'Q\n')
    old_clip=pymupdf.Rect(201.6,51.6,580.8,364.8)
    cloud_paths=[]
    for drawing in page.get_drawings():
        fill=drawing['fill']
        if fill and old_clip.contains(drawing['rect']):
            is_point=all(abs(c-.2902)<.01 for c in fill) or all(abs(c-.1647)<.01 for c in fill)
            is_shell=fill[1]>fill[0]+.05 and fill[2]>fill[0]+.05
            if is_point or is_shell:cloud_paths.append(drawing['rect'])
    assert cloud_paths and all(clip.contains(box) for box in cloud_paths)
    labels = []
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines', []):
            for span in line['spans']:
                box = pymupdf.Rect(span['bbox'])
                if clip.contains(box):
                    labels.append((span, line['dir']))
                    page.add_redact_annot(box, fill=None)
    page.apply_redactions(images=0, graphics=0, text=0)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(5,5),clip=clip,alpha=False)
    ax=fig.add_axes([0,.06/2.15,3.2/5.2,(3.2*clip.height/clip.width)/2.15])
    ax.imshow(plt.imread(io.BytesIO(pix.tobytes('png'))),extent=(clip.x0,clip.x1,clip.y1,clip.y0))
    ax.set_xlim(clip.x0,clip.x1);ax.set_ylim(clip.y1,clip.y0);ax.axis('off')
    for span,direction in labels:
        x0,y0,x1,y1=span['bbox']; is_axis=span['text'].startswith('UMAP')
        label=span['text']
        # Endpoint-only tick labels avoid crowding at the smaller print size.
        if not is_axis:
            t=span['text'].replace('−','-')
            if y0>300 and x0<420:
                if x0>=280 and t!='15':continue
            elif y0>270 and x0>=420:
                if t=='121416':
                    # The source PDF groups these adjacent labels into one span.
                    label='16';x0,y0,x1,y1=515.0186,281.7320,525.0267,293.7110
                elif t!='0':continue
            elif x0>=520 and y0<280 and t not in ['0.0','15.0']:
                continue
        ax.text((x0+x1)/2,(y0+y1)/2,label,ha='center',va='center',
                rotation=np.degrees(np.arctan2(-direction[1],direction[0])),
                fontsize=8.5 if is_axis else 8,fontweight='bold' if is_axis else 'normal')
    for name,point,anchor,color in [
        ('Germany',(402.2668457,232.5444336),(250,64),'#B30000'),
        ('Austria',(400.8131714,236.9431152),(250,100),'#B30000'),
        ('Cannabis',(429.9030762,214.6152344),(510,64),'#6B6B6B'),
        ('Jason Momoa',(364.4332886,151.5898438),(510,100),'#6B6B6B')]:
        ax.plot([point[0],anchor[0]],[point[1],anchor[1]],color=color,lw=.8)
        ax.text(*anchor,name,fontsize=8,ha='center',va='center',
                bbox=dict(boxstyle='round,pad=.15',fc='white',ec=color,lw=.6))
    for x,label,color in [(250,'Near\nHigh similarity','#B30000'),(475,'Far\nLow similarity','#6B6B6B')]:
        ax.text(x,139,label,fontsize=8,ha='center',va='center',color='white',
                bbox=dict(boxstyle='round,pad=.15',fc=color,ec='white',lw=.4))
    # A compact orientation key replaces the empty projected base plane.
    for endpoint,label,label_position in [((295,257),'UMAP-1',(295,272)),
                                          ((272,223),'UMAP-2',(252,211))]:
        ax.annotate('',xy=endpoint,xytext=(230,245),
                    arrowprops=dict(arrowstyle='->',lw=.6,color='#555555'))
        ax.text(*label_position,label,ha='center',va='center',fontsize=8.5)
    pdf.close()
    return source, list(clip)


def semantic_cloud():
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
    fig = plt.figure(figsize=(5.2, 2.15))
    source, crop = draw_original_cloud(fig)
    fig.text(.005, .95, "A", weight="bold", fontsize=9)
    fig.text(3.35/5.2, .95, "B", weight="bold", fontsize=9)
    ax = fig.add_axes([3.72/5.2, .30/2.15, 1.40/5.2, 1.30/2.15])
    for i, near in enumerate([False, True]):
        s = stats[stats.near == near].set_index("model").loc[[m[0] for m in MODELS]]
        ax.bar(np.arange(3)+(i-.5)*.34, s["mean"], width=.34, color="#980000" if near else "#777777",
               label="Near: at/above median" if near else "Far: below median", yerr=s["sem"], capsize=2,
               error_kw={"elinewidth":.8, "capthick":.8})
    ax.set_xticks(range(3), ["Llama", "Mistral", "Qwen"], fontsize=8)
    ax.set_ylim(0, .7); ax.set_yticks([0, .3, .6])
    ax.set_ylabel("Off-target refusal\n(spillover)", fontsize=8.5, labelpad=3)
    ax.legend(loc="lower right",bbox_to_anchor=(1,1.02),ncol=1,fontsize=8,
              title='Semantic similarity',title_fontsize=8,
              handlelength=.9,handletextpad=.4,borderaxespad=0,labelspacing=.15)
    ax.spines["left"].set_position(("outward", 1.5)); ax.spines["bottom"].set_position(("outward", 1.5))
    save(fig, "semantic_cloud")
    (OUT / "semantic_provenance.json").write_text(json.dumps({"cloud_source":str(source.relative_to(ROOT)),
        "cloud_crop_pdf_points":crop, "cloud_recomputed":False,
        "cloud_annotations":"Original points/shells retained and checked inside the tighter crop; empty base trimmed, compact orientation key, separated pair labels, near/far definitions instead of cosine values.", "median_cosine":median,
        "near_far_rows_per_model":900, "pair_correlations":correlations,
        "rate_variant":"prefill_logit", "cached_original_rate_variant":"main", "rate_variant_audit":audit,
        "scope":"Cloud pools six datasets; near/far and pair tables use ten ConceptVectors concepts. External MiniLM embeddings, not LLM hidden states.",
        "limits":"Projection is illustrative; error bars are row-wise SEM, not cluster-adjusted confidence intervals."}, indent=2)+"\n")


def main():
    OUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
    for path in OUT.iterdir():
        if path.suffix in {'.png','.pdf','.csv','.json'}:
            path.unlink()
    plt.rcParams.update({"font.family":"serif", "font.serif":["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset":"cm", "font.size":8.5, "axes.labelsize":8.5, "xtick.labelsize":8, "ytick.labelsize":8,
        "axes.spines.top":False, "axes.spines.right":False, "axes.linewidth":.65,
        "legend.frameon":False, "figure.dpi":120, "savefig.dpi":400})
    record(UNIT / "inputs/refusal_example.png")
    layer_metrics(); selectivity(); semantic_cloud()
    for path, digest in SOURCES.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest
    (OUT / "source_hashes.json").write_text(json.dumps(SOURCES, indent=2)+"\n")
    (OUT / "layout.json").write_text(json.dumps(LAYOUT,indent=2)+"\n")
    print(f"Wrote 3 figures and source summaries to {OUT}; all input hashes unchanged.")


if __name__ == "__main__":
    main()
