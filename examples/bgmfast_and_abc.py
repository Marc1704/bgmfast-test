
'''
BGM FASt and ABC

This script is an example on how to run BGM FASt together with ABC to derive the approximate posterior distribution of the input parameters.
'''

# ***************
# GENERAL IMPORTS
# ***************

import astroabc

# ***************
# BGMFAST IMPORTS
# ***************

from bgmfast import parameters
from bgmfast.auxiliary_functions import *
from bgmfast.bgmfast_simulation_class import bgmfast_simulation

# *****************
# PARAMETERS IMPORT
# *****************

catalog_file_parameters = parameters.catalog_file_parameters
sel_columns_catalog = catalog_file_parameters['sel_columns_catalog'].value
Gmax_catalog = catalog_file_parameters['Gmax_catalog'].value

ms_file_parameters = parameters.ms_file_parameters
sel_columns_ms = ms_file_parameters['sel_columns_ms'].value
Gmax_ms = ms_file_parameters['Gmax_ms'].value

# ***********************************************
# NAME OF THE CATALOG AND MOTHER SIMULATION FILES
# ***********************************************

filename_catalog = "./input_data/catalog/Gaia_DR3_G13.csv"
filename_ms = "./input_data/ms/ms_G13_err.csv"

# ************************************
# NAME OF THE OUTPUT AND RESTART FILES
# ************************************

restart_file = "./bgmfast_and_abc_restart_file.txt"
output_file = "./bgmfast_and_abc_output_file.txt"

# ***************************
# SETTING BGM FAST SIMULATION
# ***************************

#Create a bgmfast_simulation class object
bgmfast_sim = bgmfast_simulation()

#Open Spark session and avoid INFO logs
sc, spark = bgmfast_sim.open_spark_session()
spark.sparkContext.setLogLevel("WARN")

#Set parameters for the BGM FASt simulation
bgmfast_sim.set_acc_parameters()
bgmfast_sim.set_binning_parameters()
bgmfast_sim.set_general_parameters()
bgmfast_sim.set_ms_parameters()
bgmfast_sim.set_ps_parameters()
bgmfast_sim.set_constraints_parameters()
bgmfast_sim.set_bgmfast_parameters()

#Generate catalog Hess diagram
bgmfast_sim.read_catalog(filename_catalog, sel_columns_catalog, Gmax_catalog)

#Retrive the generated Hess diagram
catalog_data = bgmfast_sim.generate_catalog_cmd()
MATRIXCatalog2CMD, MATRIXCatalog2CMD2 = bgmfast_sim.return_cmd()[1:3]

#Read the Mother Simulation
bgmfast_sim.read_ms(filename_ms, sel_columns_ms, Gmax_ms)

# ************************
# RUNNING BGMFAST WITH ABC
# ************************

#Define a general simulation (without fixed parameters)
model_sim = bgmfast_sim.run_simulation

#Define priors of the free parameters
ms_params = parameters.ms_parameters 
priors = [('normal', [ms_params['alpha1_ms'].value, 2]),
          ('normal', [ms_params['alpha2_ms'].value, 2]),
          ('normal', [ms_params['alpha3_ms'].value, 2]),
          ('normal', [ms_params['SigmaParam_ms'].value[0], 2]),
          ('normal', [ms_params['SigmaParam_ms'].value[1], 2]),
          ('normal', [ms_params['SigmaParam_ms'].value[2], 2]),
          ('normal', [ms_params['SigmaParam_ms'].value[3], 2]),
          ('normal', [ms_params['midpopbin_ms'].value[0], 2]),
          ('normal', [ms_params['midpopbin_ms'].value[1], 2]), 
          ('normal', [ms_params['midpopbin_ms'].value[2], 2]),
          ('normal', [ms_params['midpopbin_ms'].value[3], 2]),
          ('normal', [ms_params['lastpopbin_ms'].value[0], 2]),
          ('normal', [ms_params['lastpopbin_ms'].value[1], 2]),
          ('normal', [ms_params['lastpopbin_ms'].value[2], 2]), 
          ('normal', [ms_params['T_lastpopbin_ms'].value[0], 2]),
          ('normal', [ms_params['T_lastpopbin_ms'].value[1], 2])]

#Define specificities of the ABC process
prop = {"from_restart": False, 'dfunc': dist_metric_gdaf2, 'verbose': 1, 'adapt_t': True, 'pert_kernel': 2, 'restart': restart_file, 'outfile': output_file}

#Define an ABC class object
sampler = astroabc.ABC_class(16, 100, catalog_data, [2*10**6, 10**5], 100, priors, **prop)

#Start ABC
sampler.sample(model_sim)

print('Convergence achieved')

spark.stop()
