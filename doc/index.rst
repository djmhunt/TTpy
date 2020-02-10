pyHPDM!
================================
python Human Probabilistic Decision-Modelling (pyHPDM) is a framework for modelling and fitting the responses of people to probabilistic decision making tasks.

Prerequisites
*************
This code has been tested using ``Python 2.7``. Apart from the standard Python libraries it also depends on the `SciPy <http://www.scipy.org/>`_ librariesand a few others listed in ``requirements.txt``. For those installing Python for the first time I would recommend the `Anaconda Python distribution <https://store.continuum.io/cshop/anaconda/>`_.

Installation
************
For now this is just Python code that you download and use, not a package.

Usage
*****
The framework has until now either been run with a run script or live in a command-line (or `jupyter notebook <http://jupyter.org/>`_).

A task simulation can be simply created by running ``simulation.simulation()``. Equally, for fitting participant data, the function is ``dataFitting.data_fitting``. For now, no example data has been provided.

More complex example running scripts can be found in ``./runScripts/``. Here, a number of scripts have been created as templates: ``runScript_sim.py`` for simulating the ``probSelect`` task and ``runScript_fit.py`` for fitting the data generated from ``runScript_sim.py``. A visual display of the interactions in one of these scripts will soon be created.

A new method of passing in the fitting or simulation configuration is to use a YAML configuration file. This is done, for both simulations and data fitting, using the function ``start.run_script`` For example, to run the YAML configuration equivalent to the ``runScript_sim.py`` from a command line would be :``start.run_script('./runScripts/runScripts_sim.yaml')``.

Testing
*******
Testing is done using `pytest <https://pytest.org>`_.

License
*******
This project is licenced under the `MIT license <https://choosealicense.com/licenses/mit/>`_.

Documentation
*************
The documentation can be found on `readthedocs <https://pyhpdm.readthedocs.io>`_ or in ``./doc/_build/html``, with the top level file being ``index.html``

To update the documentation you will need to install Sphinx and a set of extensions. The list of extensions can be found in ``./doc/conf.py``. To update the documentation follow the instruction in ``./doc/readme.md``

Contents:

.. toctree::
   :maxdepth: 4
   
   simulation
   dataFitting 
   data
   taskGenerator
   tasks
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

