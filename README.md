## pyHPDM ##
python Human Probabilistic Decision-Modelling (pyHPDM) is a framework for modelling and fitting the responses of people to probabilistic decision making tasks.

### Prerequisites ###
This code has been tested using ``Python 2.7``. Apart from the standard Python libraries it also depends on the [SciPy](http://www.scipy.org/) libraries and a few others listed in ``requirements.txt``. For those installing Python for the first time I would recommend the [Anaconda Python distribution](https://store.continuum.io/cshop/anaconda/).

### Installation ###
For now this is just Python code that you download and use, not a package.

### Usage ###
The framework can been run with a run script, live in a terminal or Python command-line (or [jupyter notebook](http://jupyter.org/)).

A task simulation can be simply created by running ``simulation.simulation()``. Equally, for fitting participant data, the function is ``dataFitting.data_fitting``. For now, no example data has been provided.

More complex example running scripts can be found in ``./runScripts/``. Here, a number of scripts have been created as templates: ``runScript_sim.py`` for simulating the ``probSelect`` task and ``runScript.py`` for fitting the data generated from ``runScript_sim.py``. A visual display of the interactions in one of these scripts will soon be created.

A new method of passing in the fitting or simulation configuration is to use a YAML configuration file. This is done, for both simulations and data fitting, using the function ``start.run_script`` For example, to run the YAML configuration equivalent to the ``runScript_sim.py`` from a command line would be :``start.run_script('./runScripts/runScripts_sim.yaml')``.

### Documentation ###
[![Documentation Status](https://readthedocs.org/projects/pyhpdm/badge/?version=latest)](https://pyhpdm.readthedocs.io/en/latest/?badge=latest)

The documentation can be found on [readthedocs](https://pyhpdm.readthedocs.io) or in ``./doc/_build/html``, with the top level file being ``index.html``. Alternatively it can be found on [readthedocs.io](https://pyhpdm.readthedocs.io)

To update the documentation you will need to install Sphinx and a set of extensions. The list of extensions can be found in ``./doc/conf.py``. The list of extensions can be found in ``./doc/requirements.txt``. To update the documentation follow the instruction in ``./doc/readme.md``

Any questions can either be submitted as issues or e-mailed to d.hunt@gold.ac.uk

### Testing ###
Testing is done using [pytest](https://pytest.org)

### License ###
This project is licenced under the [MIT license](https://choosealicense.com/licenses/mit/).