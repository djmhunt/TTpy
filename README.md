## Tinker Taylor py ##
TTpy is a framework for modelling and fitting the responses of people to probabilistic decision making tasks.

Any questions can either be submitted as issues or e-mailed to d.hunt (at) gold.ac.uk

### Prerequisites ###
This code has been tested using ``Python 3.7``. Apart from the standard Python libraries it also depends on the [SciPy](http://www.scipy.org/) libraries and a few others listed in ``requirements.txt``. For those installing Python for the first time I would recommend the [Anaconda Python distribution](https://store.continuum.io/cshop/anaconda/).

### Installation ###
For now this is just Python code that you download and use, not a package.

### Usage ###
The framework can been run with a run script, live in a terminal or Python command-line (or [jupyter notebook](http://jupyter.org/)).

A task simulation can be simply created by running ``simulation.run()``. Equally, for fitting participant data, the function is ``dataFitting.run('./path/to/data/')``.

More complex example running scripts can be found in ``./runScripts/``. Here, a number of scripts have been created as templates: ``runScript_sim.py`` for simulating the ``probSelect`` task and ``runScript_fit.py`` for fitting the data generated from ``runScript_sim.py``, stored in the ``./tests/test_sim`` folder. The output of this fit can also be found ``./tests/test_fit`` folder.

A new method of passing in the fitting or simulation configuration is to use a YAML configuration file. This is done, for both simulations and data fitting, using the function ``start.run_config`` For example, to run the YAML configuration equivalent to the ``runScript_sim.py`` from a command line would be :``start.run_config('./runScripts/runScripts_sim.yaml')``. For the fitting example, the configuration equivalent to ``runScript_fit.py`` would be :``start.run_config('./runScripts/runScripts_fit.yaml')``.

### Documentation ###
[![Documentation Status](https://readthedocs.org/projects/ttpy/badge/?version=latest)](https://ttpy.readthedocs.io/en/latest/?badge=latest)

The documentation can be found on [readthedocs](https://ttpy.readthedocs.io) or in ``./doc/_build/html``, with the top level file being ``index.html``.

To update the documentation you will need to install Sphinx and a set of extensions. The list of extensions can be found in ``./doc/conf.py``. The list of extensions can be found in ``./doc/requirements.txt``. To update the documentation follow the instruction in ``./doc/readme.md``

### License ###
This project is licenced under the [MIT license](https://choosealicense.com/licenses/mit/).

### History ###
This repository first began as a clone of the original repository from [Bitbucket](https://bitbucket.org/djhunt/pyhpdm).

### Testing ###
Testing is done using [pytest](https://pytest.org). The tests can be found in ``./tests/``
