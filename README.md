# BGM FASt

[![Open Source Love](https://badges.frapsoft.com/os/mit/mit.svg?v=102)](https://github.com/EliseJ/astroABC/blob/master/LICENSE.txt)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/Marc1704/bgmfast-test/issues)

Python package author: Marc del Alcázar i Julià based on [Mor et al. 2018](https://ui.adsabs.harvard.edu/abs/2018A%26A...620A..79M/abstract)

The **BGM FASt Python package** is the implementation of the Besançon Galaxy Model Fast Approximate Simulations (BGM FASt) algorithm, intended for the computation of fast simulations of the standard Besançon Galaxy Model.

## Latest application

Recent results on the usage of BGM FASt together with the Approximate Bayesian Computation (ABC) to derive Galactic parameters can be found in the [corresponding directory](https://github.com/Marc1704/bgmfast-test/tree/main/latest_results).

## Installing

Install BGM FASt using pip
```
$ pip install git+https://github.com/Marc1704/bgmfast-test
```
or git clone the repository. If the first method is applied for the installation of the package, it may be necessary to uninstall it every time the package is upgraded. To do so, just use again pip
```
$ pip uninstall bgmfast
```
and then come back again with the installation.

### Dependencies

The following dependencies are required for the basic Python package:
* numpy
* scipy
* pyspark
* pandas
* astropy

In addition, to execute the examples are also required the following Python packages:
* os
* global
* mpi4py (for astroABC)
* multiprocessing (for astroABC)
* sklearn (for astroABC)

## Examples

Together with the Python package there are [some examples](https://github.com/Marc1704/bgmfast-test/tree/main/examples) on the usage of BGM FASt. The recommended example pipeline is the following one:
1.

## License

Copyright 2023 Marc del Alcázar i Julià

BGM FASt is free software made available under the MIT License. For details see the LICENSE.txt file.
