# Python wrapper for [EPnP](http://cvlab.epfl.ch/EPnP/index.php)

Originally used to implement ORBSLAM2 with Python. 

This repo will consider supporting more general application scenarios.


## Prerequisites

* A compiler with C++11 support
* CMake >= 2.8.12
* Numpy
* OpenCV 4 (not opencv-python)


## Installation

```bash
pip install PyEPnP
```

**Or** just clone this repository and pip install. Note the `--recursive` option which is
needed for the pybind11 submodule:

```bash
git clone --recursive https://github.com/Cenbylin/Python-PyEPnP.git
cd ./Python-PyEPnP
pip install .
```

## Usage

```python
import PyEPnP as pnp
```

more examples are comming 

## License

PyDBoW3 is provided under a BSD-style license that can be found in the LICENSE
file. By using, distributing, or contributing to this project, you agree to the
terms and conditions of this license.