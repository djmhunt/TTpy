python Human Probabilistic Decision-Modelling (pyHPDM) is a framework for modelling and fitting the responses of people to probabilistic decision making tasks.

This code has been tested using ``Python 2.7``. Apart from the standard Python libraries it also depends on the `SciPy <http://www.scipy.org/>`_ librariesand a few others listed in ``requirements.txt``. For those installing Python for the first time I would recommend the `Anaconda Python distribution <https://store.continuum.io/cshop/anaconda/>`_.

Running scripts can be found in ``./runScripts/`` where a number of scripts have been created and are updated for each model. There are also two templates: ``runScript.py`` for fitting data, currently set up for fitting data to the ``Decks`` experiment, and ``runScript_sim.py`` for simulating an experiment. A visual display of the interactions in one of these scripts will soon be created.

Documentation
The documentation can be found in ``./doc/_build/html``, with the top level file being ``index.html``. Alternatively it can be found on `readthedocs.io <https://pyhpdm.readthedocs.io>`_

To update the documentation you will need to install Sphinx and a set of extensions. The list of extensions can be found in ``./doc/requirements.txt``. To update the documentation follow the instruction in ``./doc/readme.md``
