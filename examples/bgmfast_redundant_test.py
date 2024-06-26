'''
BGM FASt single run

This script is an example on how to use the bgmfast package to generate a BGM FASt simulation.
'''

# ***************
# BGMFAST IMPORTS
# ***************

from bgmfast import parameters, analysis_tools
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
filename_ms = "./input_data/ms/ms_G13_err.csv" #set parquet='generate'
#filename_ms = "./input_data/ms/ms_G13_err_reduced.parquet" #once generated, set parquet='open'

# ***************************
# SETTING BGM FAST SIMULATION
# ***************************

#Create a bgmfast_simulation class object
bgmfast_sim = bgmfast_simulation()

#Open Spark session
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

#Read the Mother Simulation
bgmfast_sim.read_ms(filename_ms, sel_columns_ms, Gmax_ms, parquet='generate', num_partitions=100)

# ************************************************
# RUNNING BGMFAST SIMULATION WITH FIXED PARAMETERS
# ************************************************

#Input parameters of BGM FASt: three slopes of the IMF and 11 SFH parameters (we put the values of the Mother Simulation, just for testing)
ms_params = parameters.ms_parameters 
param = [ms_params['alpha1_ms'].value,
         ms_params['alpha2_ms'].value,
         ms_params['alpha3_ms'].value,
         ms_params['SigmaParam_ms'].value[0],
         ms_params['SigmaParam_ms'].value[1],
         ms_params['SigmaParam_ms'].value[2],
         ms_params['SigmaParam_ms'].value[3],
         ms_params['midpopbin_ms'].value[0],
         ms_params['midpopbin_ms'].value[1],
         ms_params['midpopbin_ms'].value[2],
         ms_params['midpopbin_ms'].value[3],
         ms_params['lastpopbin_ms'].value[0],
         ms_params['lastpopbin_ms'].value[1],
         ms_params['lastpopbin_ms'].value[2],
         ms_params['T_SigmaParam_ms'].value[0],
         ms_params['T_SigmaParam_ms'].value[1], 
         ms_params['T_SigmaParam_ms'].value[2], 
         ms_params['T_SigmaParam_ms'].value[3]]

#Run the BGM FASt simulation for the given parameters
simulation_data = bgmfast_sim.run_simulation(param)

#Compute the distance between catalog and simulation Hess diagrams
print('Distance for the parameters in the Mother Simulation:', dist_metric_gdaf2(catalog_data, simulation_data))

#End the Spark session
spark.stop()

#Compute the distance between the catalog and the Mother Simulation Hess diagrams
comparison = analysis_tools.compare_hess_diagrams()
filename_ms = "./input_data/ms/ms_G13_err.csv"
colnames_ms = {'G': 'Gerr', 'color': 'GRperr', 'longitude': 'longitude', 'latitude': 'latitude', 'Mvarpi': 'Mvarpi', 'parallax': 'parallaxerr'}
ms_cmd, ms_data = comparison.generate_catalog_hess_diagram(filename_ms, colnames_ms, Gmax_ms)
distance = comparison.compute_distance(catalog_data, ms_data)
print('Distance between Gaia DR3 and the Mother Simulation:', distance)
