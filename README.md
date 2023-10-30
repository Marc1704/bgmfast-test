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

Once installed the BGM FASt package, it is important to move the astroABC package inside BGM FASt to the correct path to be able to import it from the example scripts. To do so, you can just run the move_astroabc.py script as follows:
```
python move_astroabc.py
```
If permission problems arise, try executing the same code as a sudo: 
```
sudo python move_astroabc.py
```

### Dependencies

The following dependencies are required to run the code and the examples:
* pandas
* numpy
* astropy
* pyspark
* scikit-learn


## Examples

Together with the Python package are given [some examples](https://github.com/Marc1704/bgmfast-test/tree/main/examples) on the usage of BGM FASt. The recommended example pipeline is the following one:

(0. If the MS file is provided in different files use join_ms_files.py to build a single MS file.)

1. set_catalog_for_bgmfast.py --> this script allows to adequate the format of the file obtained from the Gaia DR3 archive for BGM FASt.
2. set_ms_for_bgmfast.py --> this script is intended to adequate the format of the MS file for BGM FASt as well as computing the absolute magnitude and the PopBin.
3. bgmfast_single_run.py --> with this script you will be able to run BGM FASt for the first time and compute the distance with respect to the catalog data given a set of parameters. 
4. bgmfast_and_abc.py --> this script will let you iteratively run BGM FASt with ABC to derive Galactic parameters. 

Must be noted that this pipeline is just for testing the code, and proper modifictations of the input parameters must be done before obtaining scientific outcomes.

## License

Copyright 2023 Marc del Alcázar i Julià

BGM FASt is free software made available under the MIT License. For details see the LICENSE.txt file.
