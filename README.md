# MA-VELP
Machine-learning Assisted Virtual Exfoliation via Liquid Phase

Python codes are located in the `src/mavelp` directory:
  1. data_tool.py : manages file reading and data management
  2. gui.py :  graphical user interface supports user interface and communication between various codes
  3. kernel_method.py : kernele ridge regression is implemented for MAVELP dataset and used for fitting, prediction and score, additionally it supports optimization 
  4. neural_network.py: neural network based regression is implemented to train NN for MAVELP dataset, addtionally it supports optimization of solvent composition

Data are located in the data directory:
  1. data.dat : data for MA-VELP
   
Usage: 
  - Access the live application at [https://nanohub.org/tools/mavelp](https://nanohub.org/tools/mavelp)
  - To install locally, see Installation (below)
  - Launch MA-VELP-App.ipynb to have access to the GUI
  
Future Release Note: 
  - We might move optimization to a separate code
  - Pip installation seems like a good option
  - Python test 

## Installation

#### Manual python install for testing
```
pip install -e .
```

#### Jupyter Notebook Testing
* clone or otherwise upload the repo to you nanoHUB workspace.  Be sure to checkout the appropriate branch.
* Run `make install` in `/src` directory
* Launch the Jupyter Notebook tool: https://nanohub.org/tools/jupyter
* open `MA-0VELP_App.ipynb` in `/bin`.

#### nanoHUB Testing (Workspace)
Install to `./bin` using makefile in `./src`
```
$ cd src
$ make distclean
$ make install
```

Test by launching the jupyter notebook tool on nanohub.  

## Versioning

Version numbers are based on the [SemVer](http://semver.org/) versioning convention. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).

Versions are incremented each time the devel branch is merged to master and/or anytime a development release is desired for testing. Increment versions using a dedicated git commit of `bumpversion` changes.

All `bumpversion` commands run from the top directory of this repo.

Show current version setting:
```
$ cat VERSION
$ 0.0.0
```
Increment a "build" release: `1.0.0-dev0 -> 1.0.0-dev1`:
```
$ cat VERSION
$ 0.0.0
$ bumpversion build
```

[Bump2version](https://github.com/c4urself/bump2version) is used to increment the version and apply tags.  The basic setup used here follows the guidlines illustrated [here](https://medium.com/@williamhayes/versioning-using-bumpversion-4d13c914e9b8).  All version bumps should happen on a clean working copy of the repository, after the last commit for that version has been pushed.  The push of the the `bump2version` changes will comprise the version.
**Relavant files**
```
.bumpversion.cfg
VERSION
src/gsaraman/__init.py
setup.py
```

**examples**
Create initial testing release for upcoming #.#.# :
```

```
