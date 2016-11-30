# Building the documentation

1. Install [Python 2.7](https://www.python.org/download/)

2. Install Python [setuptools](https://pypi.python.org/pypi/setuptools) (to get the easy_install script)

3. Install Sphinx: `easy_install -U Sphinx`

4. Install NumpyDoc: `easy_install numpydoc`

5. Open a command line in "./Code/doc"

6. Build the rst files automatically (OR add them by hand!): `sphinx-apidoc -o . ../ -f -d 4`

7. Clean up the generated `rst` files to include only what you want

8. Make the html documentation: `make.bat html`

# ReStructured Text

* [RST documentation](http://docutils.sourceforge.net/rst.html)
  - [Quickstart](http://docutils.sourceforge.net/docs/user/rst/quickstart.html)
  - [Quickref](http://docutils.sourceforge.net/docs/user/rst/quickref.html)
  - [Cheatsheet](http://docutils.sourceforge.net/docs/user/rst/cheatsheet.txt)
* [Sphinx rest](http://sphinx-doc.org/rest.html)

# Sphinx Markup Constructs

* [Sphinx markup description](http://sphinx-doc.org/markup/index.html)
* [NumpyDoc](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt)

# Sphinx

* [Sphinx](http://sphinx-doc.org/)
* [Sphinx autobuild](https://pypi.python.org/pypi/sphinx-autobuild)