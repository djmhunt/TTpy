This code has been tested using ``Python 2.7.9``. Apart from the standard Python libraries it also depends on the [SciPy](http://www.scipy.org/) libraries. For those installing Python for the first time I would recommend the [Anaconda Python distribution](https://store.continuum.io/cshop/anaconda/).

A template running script can be found in ``./runScripts/runScript.py`` and a number of scripts have been created and are updated for each model. Currently they are set up for fitting data to the ``Decks`` experiment. A visual display of the interactions in one of these scripts will soon be created.

### Documentation ###
The documentation can be found in ``./doc/_build/html``, with the top level file being ``index.html``

To update the documentation you will need to install Sphinx and a set of extensions. The list of extensions can be found in ``./doc/conf.py``. To update the documentation follow the instruction in ``./doc/runNotes.txt``

### Call Graph ###

To see what is called you can:

```
#!python
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
with PyCallGraph(output=GraphvizOutput()):    
    execfile('<scriptname>.py') 

```