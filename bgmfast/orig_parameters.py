'''
Define parameters for BGM FASt

This script is intended to define the parameters needed to make bgmfast module work.
'''


import numpy as np
import pandas as pd


class new_param:

    '''
    Define a new parameter
    '''

    def __init__(self, key, value, unit="-", info='No information available'):
        '''
        Initialize the parameter

        Input parameters
        ----------------
        key : str --> name of the parameter
        value : any --> value of the parameter. It can be an int, a float, a str, a list, ...
        unit : str --> unit in which the value of the parameter is expressed
        info : str --> additional information on the parameter
        '''

        self.key = key
        self.value = value
        self.unit = unit
        self.info = info

    def outdict(self):
        '''
        Recover the parameter information in a dictionary

        Output parameters
        -----------------
        param_dict : dict --> dictionary with all the information of the parameter defined in __init__ function
        '''

        param_dict = {self.key: self}

        return param_dict


def display_as_table(set_of_params):

    '''
    Display a set of parameters as a Pandas table

    Input parameters
    ----------------
    set_of_params : list --> list with the name of the parameters we want to display

    Output parameters
    -----------------
    df : pandas dataframe --> dataframe with all the information of the required parameters
    '''

    keys = set_of_params.keys()
    values = []
    units = []
    infos = []
    for key in keys:
        values.append(set_of_params[key].value)
        units.append(set_of_params[key].unit)
        infos.append(set_of_params[key].info)
    data = {'key': keys, 'value': values, 'unit': units, 'info': infos}

    df = pd.DataFrame(data=data)
    print(df)

    return df



# **********
# PARAMETERS
# **********

#Parameters to define the accumulators' dimensions for the HR diagram ("1" corresponds to -1<M_G'<5 and "2" to 5<M_G'<15)

nLonbins = new_param('nLonbins', 1, info='Number of bins in longitude of the complete sample')
nLatbins = new_param('nLatbins', 3, info='Number of bins in latitude of the complete sample')
nColorbins = new_param('nColorbins', 63, info='Number of bins in Bp-Rp color of the complete sample')
nGbins = new_param('nGbins', 64, info="Number of bins in M_G' magnitude of the complete sample")
nLonbins1 = new_param('nLonbins1', 1, info='Number of bins in longitude')
nLatbins1 = new_param('nLatbins1', 3, info='Number of bins in latitude')
nColorbins1 = new_param('nColorbins1', 63, info='Number of bins in Bp-Rp color')
nGbins1 = new_param('nGbins1', 24, info="Number of bins in M_G' magnitude")
nLonbins2 = new_param('nLonbins2', 1, info='Number of bins in longitude')
nLatbins2 = new_param('nLatbins2', 3, info='Number of bins in latitude')
nColorbins2 = new_param('nColorbins2', 1, info='Number of bins in Bp-Rp colour')
nGbins2 = new_param('nGbins2', 40, info="Number of bins in M_G' magnitude")

acc_parameters = {**nLonbins.outdict(), **nLatbins.outdict(), **nColorbins.outdict(), **nGbins.outdict(), **nLonbins1.outdict(), **nLatbins1.outdict(), **nColorbins1.outdict(), **nGbins1.outdict(), **nLonbins2.outdict(), **nLatbins2.outdict(), **nColorbins2.outdict(), **nGbins2.outdict()}


#Parameters for the binning

Xmin = new_param('Xmin', -0.42, 'mag', info='Minimum value for the binning in Bp-Rp range')
Xmax = new_param('Xmax', 2.73, 'mag', info='Maximum value for the binning Bp-Rp range')
Ymin = new_param('Ymin', -1, 'mag', info="Minimum value for the binning M_G' range")
Ymax = new_param('Ymax', 15, 'mag', info="Maximum value for the binning M_G' range")
Bmin = new_param('Bmin', -90., 'deg', info='Minimum value for the binning of latitude')
Bmax = new_param('Bmax', +90, 'deg', info='Maximum value for the binning of latitude')
Lmin = new_param('Lmin', 0, 'deg', info='Minimum value for the binning of longitude')
Lmax = new_param('Lmax', 360, 'deg', info='Maximum value for the binning of longitude')
blims = new_param('blims', [[30, 90], [10, 30], [0, 10]], info='Limits of the different absolute latitude ranges')
llims = new_param('llims', [[0, 360]], info='Limits of the different longitude ranges')
Ylims = new_param('Ylims', [[-1, 5], [5, 15]], info='Limits of the different Bp-Rp ranges')
Ylims_Xsteps = new_param('Ylims_Xsteps', [0.05, np.inf], info='Bp-Rp steps of the different Bp-Rp ranges')
Ylims_Ysteps = new_param('Ylims_Ysteps', [0.25, 0.25], info="M_G' steps of the different color Bp-Rp ranges")

binning_parameters = {**Xmin.outdict(), **Xmax.outdict(), **Ymin.outdict(), **Ymax.outdict(), **Bmin.outdict(), **Bmax.outdict(), **Lmin.outdict(), **Lmax.outdict(), **blims.outdict(), **llims.outdict(), **Ylims.outdict(), **Ylims_Xsteps.outdict(), **Ylims_Ysteps.outdict()}


#General parameters

x1 = new_param('x1', 0.015, 'M_Sun', info='Minimum mass to generate a star')
x4 = new_param('x4', 120, 'M_Sun', info='Maximum mass to generate a star')
tau_min_edges = new_param('tau_min_edges', [0, 0.15, 1, 2, 3, 5, 7], 'Gyr', info='Lower limits of the age subpopulations of the thin disc')
tau_max_edges = new_param('tau_max_edges', [0.15, 1, 2, 3, 5, 7, 10], 'Gyr', info='Upper limits of the age subpopulations of the thin disc')
ThickParamYoung = new_param('ThickParamYoung', 1, info='Weight of the young thick disc stars')
ThickParamOld = new_param('ThickParamOld', 1, info='Weight of the old thick disc stars')
HaloParam = new_param('HaloParam', 1, info='Weight of the halo stars')

general_parameters = {**x1.outdict(), **x4.outdict(), **tau_min_edges.outdict(), **tau_max_edges.outdict(), **ThickParamYoung.outdict(), **ThickParamOld.outdict(), **HaloParam.outdict()}


#Parameters for the Mother Simulation: MS-2306

x2_ms = new_param('x2_ms', 0.5, 'M_Sun', info='First mass limit of the IMF')
x3_ms = new_param('x3_ms', 1.53, 'M_Sun', info='Second mass limit of the IMF')
alpha1_ms = new_param('alpha1_ms', -0.001 + 1, info='First slope (alpha) of the IMF for the MS')
alpha2_ms = new_param('alpha2_ms', 1.7 + 1, info='Second slope (alpha) of the IMF for the MS')
alpha3_ms = new_param('alpha3_ms', 0.9 + 1, info='Third slope (alpha) of the IMF for the MS')
SigmaParam_ms = new_param('SigmaParam_ms', np.array([0.1, 2.5, 5.5, 9.0, 6.2 + 6.1, 2.9 + 5.0, 4.0 + 1.7 + 6.1]), 'M_Sun/pc^2', info='Surface density at the position of the Sun for the different age subpopulations of the thin disc for the MS')
midpopbin_ms = new_param('midpopbin_ms', np.array([6.2, 6.1, 2.9, 5.0]), 'M_Sun/pc^2', info='Surface density at the position of the Sun for the four subdivisions of the 5th and 6th age subpopulations of the thin disc for the MS (3-5 Gyr and 5-7 Gyr)')
lastpopbin_ms = new_param('lastpopbin_ms', np.array([4.0, 1.7, 6.1]), 'M_Sun/pc^2', info='Surface density at the position of the Sun for the three subdivisions of the last (7th) age subpopulation of the thin disc for the MS (7-10 Gyr)')

ms_parameters = {**x2_ms.outdict(), **x3_ms.outdict(), **alpha1_ms.outdict(), **alpha2_ms.outdict(), **alpha3_ms.outdict(), **SigmaParam_ms.outdict(), **midpopbin_ms.outdict(), **lastpopbin_ms.outdict()}


#Parameters for the BGM FASt simulation (pseudo-simulation)

x2_ps = new_param('x2_ps', x2_ms.value, 'M_Sun', info='First mass limit of the IMF')
x3_ps = new_param('x3_ps', x3_ms.value, 'M_Sun', info='Second mass limit of the IMF')

ps_parameters = {**x2_ps.outdict(), **x3_ps.outdict()}


#Parameters to check the stars coming from the Mother Simulation

tau_min = new_param('tau_min', 0, 'Gyr', info='Minimum age of a thin disc star')
tau_max = new_param('tau_max', 10, 'Gyr', info='Maximum age of a thin disc star')
mass_min = new_param('mass_min', 0.015, 'M_Sun', info='Minimum mass to generate a star')
mass_max = new_param('mass_max', 120, 'M_Sun', info='Maximum mass to generate a star')
l_min = new_param('l_min', 0, 'deg', info='Minimum galactic longitude')
l_max = new_param('l_max', 360, 'deg', info='Maximum galactic longitude')
b_min = new_param('b_min', -90, 'deg', info='Minimum galactic latitude')
b_max = new_param('b_max', +90, 'deg', info='Maximum galactic latitude')
r_min = new_param('r_min', 0, 'pc', info='Minimum distance')
r_max = new_param('r_max', 50000, 'pc', info='Maximum distance')

constraints_parameters = {**tau_min.outdict(), **tau_max.outdict(), **mass_min.outdict(), **mass_max.outdict(), **l_min.outdict(), **l_max.outdict(), **b_min.outdict(), **b_max.outdict(), **r_min.outdict(), **r_max.outdict()}


#Parameters for the import of the catalog file

sel_columns_catalog = new_param('sel_columns_catalog', ['G','BpRp','longitude','latitude', 'Mvarpi', 'parallax'])
Gmax_catalog = new_param('Gmax_catalog', 13.0, 'mag')

catalog_file_parameters = {**sel_columns_catalog.outdict(), **Gmax_catalog.outdict()}


#Parameters for the import of the Mother Simulation file

sel_columns_ms = new_param('sel_columns_ms', ['Gerr','BpRperr','PopBin','Age','MassOut','longitude','latitude','parallaxerr', 'Mvarpi'])
Gmax_ms = new_param('Gmax_ms', 13.0, 'mag')

ms_file_parameters = {**sel_columns_ms.outdict(), **Gmax_ms.outdict()}

















