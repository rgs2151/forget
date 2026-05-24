from .activations import (
    cached_concept_activations,
    collect_answer_activations_batched,
    flatten_token_rows,
    masked_mean_acts,
    pool_activation_dict,
)
from .baseline import generate_baseline
from .calibration import is_refusal_output, select_refusal_scale, select_scale
from .chat_templates import LLAMA3, QWEN, TEMPLATES, ChatTemplate, detect_template
from .gpu import GPUPool
from .intervention import (
    GatedSteering,
    Steering,
    load_or_empty_results,
    make_generation_jobs,
    run_generation_jobs,
    sample_per_concept,
)
from .model import load_llm
from .paths import Paths, cached_csv_rows, cached_pt
from .pipeline import CALIBRATION_SCALES, VECTOR_METHODS, default_intervention_layers, run
from .plots import (
    custom_cmap,
    make_all as make_plots,
    plot_calibration,
    plot_detection_roc,
    plot_heatmap,
    setup_style,
)
from .prompts import BASELINE_SYSTEM, refuse_system
from .scoring import add_acceptability_column, add_refusal_column, add_retention_column
from .vectors import (
    cached_diffed_vectors,
    cached_lda_vectors,
    cached_projected_vectors,
    diffed_vectors,
    lda_vectors,
    projected_vectors,
)
