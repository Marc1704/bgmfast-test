'''
BGM FASt single run

This script is an example on how to use the bgmfast package to generate a BGM FASt simulation.
'''

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

# ***************************
# SETTING BGM FAST SIMULATION
# ***************************

#Create a bgmfast_simulation class object
bgmfast_sim = bgmfast_simulation()

#Open Spark session
sc, spark = bgmfast_sim.open_spark_session()

#Set parameters for the BGM FASt simulation
bgmfast_sim.set_acc_parameters()
bgmfast_sim.set_binning_parameters()
bgmfast_sim.set_general_parameters()
bgmfast_sim.set_ms_parameters()
bgmfast_sim.set_ps_parameters()
bgmfast_sim.set_constraints_parameters()

#Generate catalog Hess diagram
bgmfast_sim.read_catalog(filename_catalog, sel_columns_catalog, Gmax_catalog)

#Retrive the generated Hess diagram
catalog_data = bgmfast_sim.generate_catalog_cmd()
MATRIXCatalog2CMD, MATRIXCatalog2CMD2 = bgmfast_sim.return_cmd()[1:3]

#Read the Mother Simulation
bgmfast_sim.read_ms(filename_ms, sel_columns_ms, Gmax_ms)

# ************************************************
# RUNNING BGMFAST SIMULATION WITH FIXED PARAMETERS
# ************************************************

#Input parameters of BGM FASt: three slopes of the IMF and 11 SFH parameters (from the execution GV13-MS22 in the TFM that can be found in latest_results folder)
param = [-1.88, 2.76, 1.85, 0.84*0.15, 3.23*0.85, 6.06*1, 9.79*1, 6.74*1, 6.33*1, 2.51*1, 4.52*1, 3.92*1, 1.87*1, 6.44*1]

#Run the BGM FASt simulation for the given parameters
simulation_data = bgmfast_sim.run_simulation(param)

#Compute the distance between catalog and simulation Hess diagrams
print('Distance for the parameters derived in GV13-MS22:', dist_metric_gdaf2(catalog_data, simulation_data))

#End the Spark session
spark.stop()
