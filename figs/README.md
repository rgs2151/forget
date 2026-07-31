# Figures

| Unit | Outputs |
| --- | --- |
| `fig4_selective_refusal/` | Selective-refusal matrices and targeted/untargeted summaries |
| `fig5_layer_scale_sweeps/` | Main layer-scale calibration figure |
| `fig6_semantic_spillover/` | Semantic spillover paper figure |
| `fig7_model_size_scaling/` | Model-size figure, diagnostic, and result table |
| `supp_bars/` | Full supplementary bar summaries |
| `supp_confusion/` | Full supplementary evaluation matrices |
| `supp_calibration_optimal/` | Selected-layer calibration curves |
| `supp_calibration_layers/` | Full refusal, retention, and fluency layer sweeps |

Figure scripts read the protected model-matrix results and write only to their
own `cache/` and `plots/` folders. Each unit carries its own README and caption.
