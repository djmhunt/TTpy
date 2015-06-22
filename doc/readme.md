# Building the documentation

1. Install [Python 2.7](https://www.python.org/download/)

2. Install Python [setuptools](https://pypi.python.org/pypi/setuptools) (to get the easy_install script)

3. Install Sphinx: `easy_install -U Sphinx`

4. Open a command line in "./Code/doc"

5. Build the rst files automatically (OR add them by hand!): `sphinx-apidoc -o . ../ -f -d 4`

6. Clean up the generated `rst` files to inclde only what you want

7. Make the html documentation: `make.bat html`

# ReStructured Text

* http://docutils.sourceforge.net/rst.html
  - http://docutils.sourceforge.net/docs/user/rst/quickstart.html
  - http://docutils.sourceforge.net/docs/user/rst/quickref.html
  - http://docutils.sourceforge.net/docs/user/rst/cheatsheet.txt
* http://sphinx-doc.org/rest.html

# Sphinx Markup Constructs

* http://sphinx-doc.org/markup/index.html

# Sphinx

* http://sphinx-doc.org/
* https://pypi.python.org/pypi/sphinx-autobuild