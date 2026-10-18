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
import pymupdf
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, ConnectionPatch, PathPatch
from matplotlib.collections import PatchCollection
from matplotlib.path import Path as MplPath
from mpl_toolkits.mplot3d import proj3d
from matplotlib.transforms import Affine2D
from scipy.stats import spearmanr

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


def layer_colormap():
    """Blend three anchors in linear-light RGB to avoid dark transitions."""
    anchors = np.array([to_rgb(c) for c in ["#2476B0", "#980000", "#2CA02C"]])
    linear = np.where(anchors <= .04045, anchors/12.92,
                      ((anchors + .055)/1.055)**2.4)
    depth = np.linspace(0, 1, 4097)
    blended = np.column_stack([np.interp(depth, [0, .5, 1], channel)
                               for channel in linear.T])
    rgb = np.where(blended <= .0031308, 12.92*blended,
                   1.055*blended**(1/2.4) - .055)
    return LinearSegmentedColormap.from_list("layer_depth", rgb, N=len(depth))


def layer_metrics():
    width = 4.55
    fig = plt.figure(figsize=(width, 2.7))
    axes = np.array([[fig.add_axes([x/width,y/2.7,.98/width,.98/2.7])
                      for x in [.34,1.78,3.22]] for y in [1.52,.30]])
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
        cmap = layer_colormap()
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
                             cax=fig.add_axes([pos.x1+.04/width,pos.y0,.035/width,pos.height]))
            cb.set_ticks([0,int(d.layer.max())]);cb.ax.tick_params(length=1.5,pad=.5)
            cb.ax.minorticks_off()
            cb.set_label('Layer',rotation=270,labelpad=-1,fontsize=8.5)
            cb.outline.set_visible(False)
    pd.concat(tables).to_csv(OUT / "layer_metrics.csv", index=False)
    save(fig, "layer_metrics")
    LAYOUT['layer_metrics'].update({'subplot_inches':[.98,.98],
                                    'row_gap_inches':.24,'column_gap_inches':.46,'colorbars':6,
                                    'layer_palette':'continuous linear-light RGB interpolation; three anchors only',
                                    'layer_palette_anchors':['#2476B0','#980000','#2CA02C'],
                                    'layer_palette_positions':[0,.5,1]})


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
    cb.set_label('Refusal rate',rotation=270,labelpad=2,fontsize=8.5)
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
    record(ROOT / "figs/fig6_semantic_spillover/semantic_spillover_figure.ipynb")
    pdf = pymupdf.open(source)
    page = pdf[0]
    source_roi = pymupdf.Rect(201.6, 51.6, 580.8, 364.8)
    clip = pymupdf.Rect(201.6, 51.6, 500.0, 280.0)
    cloud_width = 3.2 * clip.width / 379.2
    # Rebuild the original projected vector geometry as native Matplotlib paths.
    # The notebook did not save 3D coordinates; never infer or refit them here.
    paths, faces, edges, widths = [], [], [], []
    counts = {'questions':0, 'centroids':0, 'shell_faces':0, 'highlight_markers':0}
    stroke_scale = 72 * cloud_width / clip.width
    for drawing in page.get_drawings():
        if not source_roi.contains(drawing['rect']):
            continue
        fill = drawing['fill']
        kind = None
        if fill:
            if all(abs(c-.2902)<.001 for c in fill):kind='questions'
            elif all(abs(c-.1647)<.001 for c in fill):kind='centroids'
            elif fill[1]>fill[0]+.05 and fill[2]>fill[0]+.05:kind='shell_faces'
            elif drawing['seqno'] in [57,59,61,63]:kind='highlight_markers'
        if kind is None:
            continue
        assert clip.contains(drawing['rect'])
        vertices, codes = [], []
        for item in drawing['items']:
            assert item[0] in ['l','c']
            start = tuple(item[1])
            if not vertices or vertices[-1] != start:
                vertices.append(start); codes.append(MplPath.MOVETO)
            if item[0]=='l':
                vertices.append(tuple(item[2]));codes.append(MplPath.LINETO)
            else:
                vertices.extend(tuple(pt) for pt in item[2:])
                codes.extend([MplPath.CURVE4]*3)
        vertices.append(vertices[0]);codes.append(MplPath.CLOSEPOLY)
        paths.append(PathPatch(MplPath(vertices,codes)))
        faces.append((*fill,drawing['fill_opacity']))
        edge=drawing['color']
        edges.append((*edge,drawing['stroke_opacity']) if edge else (0,0,0,0))
        widths.append((drawing['width'] or 0)*stroke_scale)
        counts[kind]+=1
    assert counts['questions']==5171 and counts['centroids']==311
    assert counts['highlight_markers']==4 and counts['shell_faces']>40000
    ax=fig.add_axes([0,.06/2.15,cloud_width/4.28,(cloud_width*clip.height/clip.width)/2.15])
    ax.add_collection(PatchCollection(paths,facecolors=faces,edgecolors=edges,
                                     linewidths=widths,antialiaseds=True))
    ax.set_xlim(clip.x0,clip.x1);ax.set_ylim(clip.y1,clip.y0)
    ax.set_aspect('equal');ax.axis('off')
    # Restore both example-pair links at the exact original projected endpoints.
    pair_links = [([(402.2668457,232.5444336),(400.8131714,236.9431152)],'#B30000'),
                  ([(364.4332886,151.5898438),(429.9030762,214.6152344)],'#6B6B6B')]
    for points,color in pair_links:
        xy=np.asarray(points)
        ax.plot(*xy.T,color='white',lw=3.2,solid_capstyle='round',zorder=3)
        ax.plot(*xy.T,color=color,lw=1.5,solid_capstyle='round',zorder=4)
        ax.plot(*xy.T,ls='none',marker='o',ms=2.0,mec='white',mew=.3,color=color,zorder=5)
    label_artists=[]
    for name,point,anchor,color in [
        ('Germany',(402.2668457,232.5444336),(250,64),'#B30000'),
        ('Austria',(400.8131714,236.9431152),(250,100),'#B30000'),
        ('Jason Momoa',(364.4332886,151.5898438),(415,64),'#6B6B6B'),
        ('Cannabis',(429.9030762,214.6152344),(450,112),'#6B6B6B')]:
        label_artists.append(ax.annotate(name,xy=point,xytext=anchor,
            fontsize=8,ha='center',va='center',
            bbox=dict(boxstyle='round,pad=.15',fc='white',ec=color,lw=.6),
            arrowprops=dict(arrowstyle='-',color=color,lw=.8,shrinkA=2,shrinkB=0)))
    for x,y,label,color in [(250,139,'Near','#B30000'),(460,150,'Far','#6B6B6B')]:
        label_artists.append(ax.text(x,y,label,fontsize=8,ha='center',va='center',color='white',
                bbox=dict(boxstyle='round,pad=.15',fc=color,ec='white',lw=.4)))
    # Project a true 3D orthogonal basis using the source camera and box aspect.
    camera_fig=plt.figure()
    camera=camera_fig.add_subplot(projection='3d')
    camera.set(xlim=(0,1),ylim=(0,1),zlim=(0,1))
    camera.view_init(elev=18,azim=-62);camera.set_box_aspect((1,1,.85))
    origin3=np.array([.5,.5,.5])
    basis=np.vstack([origin3,origin3+np.diag([.4,.55,.4])])
    projected=np.array(proj3d.proj_transform(*basis.T,camera.get_proj()))[:2].T
    directions=projected[1:]-projected[0]
    directions[:,1]*=-1
    directions*=78/np.linalg.norm(directions[0])
    plt.close(camera_fig)
    origin=np.array([237.,252.])
    for k,delta in enumerate(directions):
        end=origin+delta
        ax.annotate('',xy=end,xytext=origin,
                    arrowprops=dict(arrowstyle='->',lw=.7,color='#444444',shrinkA=0,shrinkB=0))
        # Place the labels parallel to their projected arrow shafts.
        normal=np.array([-delta[1],delta[0]])/np.linalg.norm(delta)
        offset=9 if k==0 else -9
        fraction=.85 if k==1 else .57
        position=origin+fraction*delta+offset*normal
        angle=np.degrees(np.arctan2(-delta[1],delta[0]))
        label_artists.append(ax.text(*position,f'UMAP-{k+1}',ha='center',va='center',
            rotation=angle,rotation_mode='anchor',fontsize=8.5))
    fig.canvas.draw()
    # Annotation extents include the leader; check the label boxes themselves.
    concept_boxes=[t.get_bbox_patch().get_window_extent() for t in label_artists[:6]]
    text_boxes=concept_boxes+[t.get_window_extent(fig.canvas.get_renderer()) for t in label_artists[6:]]
    overlaps=[(label_artists[i].get_text(),label_artists[j].get_text())
              for i,a in enumerate(text_boxes) for j,b in enumerate(text_boxes) if j>i and a.overlaps(b)]
    assert not overlaps, overlaps
    pdf.close()
    return source, list(clip), counts


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
    fig = plt.figure(figsize=(4.28, 2.15))
    source, crop, geometry_counts = draw_original_cloud(fig)
    fig.text(.005, .95, "A", weight="bold", fontsize=9)
    fig.text(2.49/4.28, .95, "B", weight="bold", fontsize=9)
    ax = fig.add_axes([2.80/4.28, .24/2.15, 1.40/4.28, 1.76/2.15])
    for i, near in enumerate([False, True]):
        s = stats[stats.near == near].set_index("model").loc[[m[0] for m in MODELS]]
        ax.bar(np.arange(3)+(i-.5)*.34, s["mean"], width=.34, color="#980000" if near else "#777777",
               label="Near" if near else "Far", yerr=s["sem"], capsize=2,
               error_kw={"elinewidth":.8, "capthick":.8})
    ax.set_xticks(range(3), ["Llama", "Mistral", "Qwen"], fontsize=8)
    ax.set_ylim(0, .72); ax.set_yticks([0, .3, .6])
    ax.set_ylabel("Off-target refusal (spillover)", fontsize=8.5, labelpad=3)
    ax.legend(loc="upper left",ncol=2,fontsize=8,
              columnspacing=.9,
              handlelength=.9,handletextpad=.4,borderaxespad=.3,labelspacing=.15)
    ax.spines["left"].set_position(("outward", 1.5)); ax.spines["bottom"].set_position(("outward", 1.5))
    save(fig, "semantic_cloud")
    LAYOUT['semantic_cloud'].update({'bar_subplot_inches':[1.40,1.76],
                                     'bar_ylim':[0,.72], 'legend':'upper left inside plot',
                                     'ylabel':'Off-target refusal (spillover)'})
    (OUT / "semantic_provenance.json").write_text(json.dumps({"cloud_source":str(source.relative_to(ROOT)),
        "cloud_crop_pdf_points":crop, "cloud_recomputed":False, "cloud_vector_paths":geometry_counts,
        "example_pair_links":[{"names":["Germany","Austria"],"projected_pdf_points":[[402.2668457,232.5444336],[400.8131714,236.9431152]]},{"names":["Jason Momoa","Cannabis"],"projected_pdf_points":[[364.4332886,151.5898438],[429.9030762,214.6152344]]}],
        "camera":{"elevation":18,"azimuth":-62,"box_aspect":[1,1,.85]},
        "cloud_annotations":"Native Matplotlib reconstruction of the source projected vector paths, preserving point/shell geometry and ordering. No frame or ticks. Three-axis key projected from the source camera; labels aligned with shafts. Jason Momoa/Cannabis/Far translated inward; both original example-pair connections restored at their exact projected endpoints. Near/Far labels only; definitions in the caption.", "median_cosine":median,
        "near_far_rows_per_model":900, "pair_correlations":correlations,
        "rate_variant":"prefill_logit", "cached_original_rate_variant":"main", "rate_variant_audit":audit,
        "scope":"Cloud pools six datasets; near/far and pair tables use ten ConceptVectors concepts. External MiniLM embeddings, not LLM hidden states.",
        "limits":"Projection is illustrative; error bars are row-wise SEM, not cluster-adjusted confidence intervals."}, indent=2)+"\n")


def main():
    OUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
    # Figures 1, 2, and 4 are frozen: preserve their artifacts byte-for-byte.
    SOURCES.update(json.loads((OUT/'source_hashes.json').read_text()))
    LAYOUT.update(json.loads((OUT/'layout.json').read_text()))
    for path in OUT.iterdir():
        if path.stem == 'layer_metrics':
            path.unlink()
    plt.rcParams.update({"font.family":"serif", "font.serif":["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset":"cm", "font.size":8.5, "axes.labelsize":8.5, "xtick.labelsize":8, "ytick.labelsize":8,
        "axes.spines.top":False, "axes.spines.right":False, "axes.linewidth":.65,
        "legend.frameon":False, "figure.dpi":120, "savefig.dpi":400})
    record(UNIT / "inputs/refusal_example.png")
    layer_metrics()
    for path, digest in SOURCES.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest
    (OUT / "source_hashes.json").write_text(json.dumps(SOURCES, indent=2)+"\n")
    (OUT / "layout.json").write_text(json.dumps(LAYOUT,indent=2)+"\n")
    print(f"Wrote Figure 3 to {OUT}; Figures 1/2/4 frozen; all input hashes unchanged.")


if __name__ == "__main__":
    main()
