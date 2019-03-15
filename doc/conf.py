# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   A copy of the License is located at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   or in the "license" file accompanying this file. This file is distributed
#   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#   express or implied. See the License for the specific language governing
#   permissions and limitations under the License.
# ==============================================================================
# -*- coding: utf-8 -*-

from datetime import datetime
import os
import shutil
import sys
from os.path import abspath, dirname

# -- Mock graphing modules - workaround for optinal dependencies as well as for wrong behavior of some required ones
from unittest.mock import MagicMock


class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()


MOCK_MODULES = ['matplotlib', 'pylab', 'matplotlib.pyplot', 'pyrfr.regression', 'pybnn']
sys.modules.update((module_name, Mock()) for module_name in MOCK_MODULES)

# -- Project information -----------------------------------------------------

project = 'emukit'
copyright = '{}, Amazon.com'.format(datetime.now().year)

exec(open("../emukit/__version__.py").read())
version = __version__ # noqa: variable __version__ is defined in exec above
release = version


# -- Initial cleanup --------------------------------------------------------
# If you use autosummary, this ensures that any stale autogenerated files are
# cleaned up first.
if os.path.exists('generated'):
    print("cleaning up stale autogenerated files...")
    shutil.rmtree('generated')


# -- Docs configuration -----------------------------------------------------

# copy over examples folder for notebook docs
EXAMPLES_SRC = "../notebooks"
EXAMPLES_DST = "notebooks"
if os.path.isdir(EXAMPLES_DST):
    shutil.rmtree(EXAMPLES_DST)
shutil.copytree(EXAMPLES_SRC, EXAMPLES_DST)

# add necessary paths
emukit_source = os.path.abspath('../emukit')
sys.path.insert(0, emukit_source)
top_level = dirname(dirname(abspath(__file__)))
sys.path.insert(1, top_level)


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest',
              'sphinx.ext.intersphinx', 'sphinx.ext.todo',
              'sphinx.ext.coverage', 'sphinx.ext.autosummary',
              'sphinx.ext.napoleon', 'nbsphinx', 'sphinx.ext.mathjax',
              'sphinx.ext.ifconfig', 'sphinx_autodoc_typehints']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
# exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints', '._**']

# List of directories, relative to source directory, that shouldn't be searched
# for source files.
exclude_patterns = ['_build', '**.ipynb_checkpoints', '_templates', '**.cfg']

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'emukitdoc'


# -- Extension configuration -------------------------------------------------

# -- autodoc extension --
autoclass_content = "class"
autodoc_default_flags = ['show-inheritance', 'members', 'undoc-members', 'classes']
autodoc_member_order = 'bysource'

# -- autosummary extension --
autosummary_generate = True

# -- intersphinx extension --
# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'https://docs.python.org/': None}

# -- nbsphinx extension --
# Allow notebooks to have errors when generating docs
nbsphinx_allow_errors = True
nbsphinx_execute = 'never'