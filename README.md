# MA-VELP
Machine-learning Assisted Virtual Exfoliation via Liquid Phase

Python codes are located in the src directory:
  1. data_tool.py : manages file reading and data management
  2. gui.py :  graphical user interface supports user interface and communication between various codes
  3. kernel_method.py : kernele ridge regression is implemented for MAVELP dataset and used for fitting, prediction and score, additionally it supports optimization 
  4. neural_network.py: neural network based regression is implemented to train NN for MAVELP dataset, addtionally it supports optimization of solvent composition

Data are located in the data directory:
  1. data.dat : data for MA-VELP
  
 
Usage: 
  Simply copy src and data files to the working directory, you can then use MA-VELP
  MA-VELP-App.ipynb to have access to the GUI
  
  
Future Release Note: 
  - We might move optimization to a separate code
  - Pip installation seems like a good option
  - Python test 

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

[Bump2version](https://github.com/c4urself/bump2version) is used to increment the version and apply tags.  The basic setup follow tha guidlines illustrated [here](https://medium.com/@williamhayes/versioning-using-bumpversion-4d13c914e9b8).  All version bumps should happen on a clean working copy of the repository, after the last commit for that version has been pushed.  The push of the the `bump2version` changes will comprise the version.
**Relavant files**
```
.bumpversion.cfg
VERSION
src/gsaraman/__init.py
setup.py
```

**examples**
Create initial testing release:
```
bump
```

### Development Versions
For commits that are to be merged to master for further testing, a development version should be created.  
