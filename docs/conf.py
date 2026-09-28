import os
import sys


sys.path.insert(0, os.path.abspath(".."))


project = "dDTW"
author = "Johannes Zeitler"
copyright = "2026, Johannes Zeitler"
release = "0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autodoc_mock_imports = [
    "torch",
    "torch.cuda",
    "torch.autograd",
    "torch.nn",
    "torch.nn.functional",
    "torch.utils",
    "torch.utils.cpp_extension",
    "numba",
    "numpy",
]

autoclass_content = "both"
autodoc_default_options = {
    "exclude-members": "__init__",
}
autodoc_member_order = "bysource"
autosummary_generate = False
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "dDTW Documentation"
html_logo = os.path.join(html_static_path[0], "figures", "logo_ddtw.png")
