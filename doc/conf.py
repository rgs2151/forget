import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "refuse"
author = "refuse"
copyright = "2026, refuse"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Heavy / GPU deps are mocked so the docs build stays light and runs no tests.
autodoc_mock_imports = [
    "torch", "torchvision", "transformers", "datasets", "accelerate",
    "huggingface_hub", "instructor", "pydantic", "matplotlib", "seaborn",
    "sklearn", "bert_score", "rouge_score", "optuna", "ipywidgets",
]
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

# The homepage is the repo readme, whose doc links are GitHub-relative (doc/*.md).
# In the built site those resolve via the sidebar toctree instead; silence the
# cosmetic missing-xref warning for them.
suppress_warnings = ["myst.xref_missing"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "refuse"
