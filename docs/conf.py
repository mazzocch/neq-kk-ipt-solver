"""Sphinx configuration for neq-kk-ipt-solver.

Docstrings are NumPy-style, parsed via napoleon; narrative pages are
Markdown, parsed via myst-parser (so the README can be included verbatim as
the front page instead of duplicating it).
"""
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "neq-kk-ipt-solver"
copyright = "2026, Tommaso Maria Mazzocchi"
author = "Tommaso Maria Mazzocchi"

from neq_kk_ipt_solver import __version__ as release  # noqa: E402

version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

# -- Napoleon (NumPy-style docstrings) --------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_rtype = False
napoleon_use_ivar = True

# -- Autodoc -----------------------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# -- Intersphinx: link out to numpy/scipy/python types in signatures --------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- MyST (Markdown) ----------------------------------------------------------
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The included README has plain repo-relative links (examples/, LICENSE, ...)
# that resolve fine on GitHub but aren't pages inside the Sphinx doc tree --
# expected, not worth failing the build over.
suppress_warnings = ["myst.xref_missing"]

html_theme = "furo"
html_static_path = ["_static"]
html_title = f"{project} {version}"
