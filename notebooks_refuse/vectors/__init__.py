from .activations import (
    build_answered_chats,
    build_answered_prompts,
    build_question_chats,
    build_question_prompts,
    collect_answer_activations_batched,
    collect_grouped_activations,
    masked_mean_acts,
    pool_activation_dict,
)
from .plots import (
    plot_forget_vector_similarity,
    plot_gated_separation,
    plot_vdetect_gate,
    plot_vdetect_similarity,
)
from .steering import diffed_vectors, lda_vectors, projected_vectors
