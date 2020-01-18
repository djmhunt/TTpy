PyHPDM!
================================

This code has been tested using ``Python 2.7``. Apart from the standard Python libraries it also depends on the `SciPy <http://www.scipy.org/>`_ libraries. For those installing Python for the first time I would recommend the `Anaconda Python distribution <https://store.continuum.io/cshop/anaconda/>`_.

The framework has until now either been run with a run script or live in a command-line (or `jupyter notebook <http://jupyter.org/>`_).

Running scripts
***************
Example running scripts can be found in ``./runScripts/``. Here, a number of scripts have been created as templates: ``runScript_sim.py`` for simulating the ``probSelect`` experiment and ``runScript_fit.py`` for fitting the data generated from ``runScript_sim.py``. A visual display of the interactions in one of these scripts will soon be created.

Documentation
*************
The documentation can be found in ``./doc/_build/html``, with the top level file being ``index.html``

To update the documentation you will need to install Sphinx and a set of extensions. The list of extensions can be found in ``./doc/conf.py``. To update the documentation follow the instruction in ``./doc/readme.md``

Contents:

.. toctree::
   :maxdepth: 4
   
   simulation
   dataFitting 
   data
   experimentGenerator
   experiment
   modelGenerator
   model
   fitAlgs
   outputting
   utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

