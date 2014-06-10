NumPy based Continuous Time Markov Chain TREE algorithms

Required dependencies:
 * [Python 2.7+](http://www.python.org/)
 * [pip](https://pip.readthedocs.org/) (installation)
 * [git](http://git-scm.com/) (installation)
 * [numpy](http://www.numpy.org/)
 * [scipy](http://docs.scipy.org/doc/)
 * [Cython](http://cython.org) (manually write unsafe fast code for Python)
   - `$ pip install --user git+https://github.com/cython/cython`
 * [NetworkX](http://networkx.lanl.gov/) (graph data types and algorithms)
   - `$ pip install --user git+https://github.com/networkx/networkx`
 * [npmctree](https://github.com/argriffing/npmctree) (markov chain on trees)
   - `$ pip install --user git+https://github.com/argriffing/npmctree`

Optional dependencies:
 * [nose](https://nose.readthedocs.org/) (testing)
 * [coverage](http://nedbatchelder.com/code/coverage/) (test coverage)
   - `$ apt-get install python-coverage`


User
----

Install:

    $ pip install --user git+https://github.com/argriffing/npctmctree

Test:

    $ python -c "import npctmctree; npctmctree.test()"

Uninstall:

    $ pip uninstall npctmctree


Developer
---------

Install:

    $ git clone git@github.com:argriffing/npctmctree.git

Test:

    $ python runtests.py

Coverage:

    $ python-coverage run runtests.py
    $ python-coverage html
    $ chromium-browser htmlcov/index.html

Build docs locally:

    $ sh make-docs.sh
    $ chromium-browser /tmp/nxdocs/index.html

Subsequently update online docs:

    $ git checkout gh-pages
    $ cp /tmp/nxdocs/. ./ -R
    $ git add .
    $ git commit -am "update gh-pages"
    $ git push

