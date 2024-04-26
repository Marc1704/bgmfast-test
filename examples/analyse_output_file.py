import numpy as np
from bgmfast.analysis_tools import * 
from bgmfast.parameters import general_parameters, ms_parameters
from bgmfast import parameters
from bgmfast.bgmfast_simulation_class import bgmfast_simulation


# ********************
# GET FINAL PARAMETERS
# ********************

filename = './bgmfast_and_abc_output_file.txt'
params_keypos = {'datetime': [0, 'datetime'], 'alpha2': [1, 1], 'alpha3': [2, 1], 'sfh0': [3, 1/0.15], 'sfh1': [4, 1/0.85], 'sfh2': [5, 1], 'sfh3': [6, 1], 'sfh4': [7, 1], 'sfh5': [8, 1], 'sfh6': [9, 1], 'sfh7': [10, 1], 'sfh8': [11, 1], 'sfh9': [12, 1], 'sfh10': [13, 1], 'sfh9T': [14, 1], 'sfh10T': [15, 1], 'sfh11T': [16, 1], 'sfh12T': [17, 1], 'distance': [18, 'distance']}

analysis = output_file_analysis(filename, params_keypos, num_acc_sim=100)
final_parameters = final_params(analysis.data, num_acc_sim=100, show=True, step_range=[0, 100])


# *********
# BUILD SFH
# *********

sfh_params = [value for key, value in final_parameters.items() if 'T' not in key and 'sfh' in key]
T_sfh_params = [value for key, value in final_parameters.items() if 'T' in key and 'sfh' in key]

tau_ranges = general_parameters['tau_ranges'].value
T_tau_ranges = general_parameters['T_tau_ranges'].value

tau_values = [np.mean(tau_ranges[i][j]) for i in range(len(tau_ranges)) for j in range(len(tau_ranges[i]))] 
T_tau_values = [np.mean(T_tau_ranges[i]) for i in range(len(T_tau_ranges))] 

build_sfh([tau_values, T_tau_values], [sfh_params, T_sfh_params], show=True)


# ***************************
# BUILD IMF (SLOPES AND REAL)
# ***************************

x1 = general_parameters['x1'].value
x2 = ms_parameters['x2_ms'].value
x3 = ms_parameters['x3_ms'].value
x4 = general_parameters['x4'].value
alpha1 = ms_parameters['alpha1_ms'].value

imf_params = [[alpha1, alpha1, alpha1]]
imf_params.extend([value for key, value in final_parameters.items() if 'alpha' in key])
imf_ranges = [x1, x2, x3, x4]

build_imf(imf_ranges, imf_params, show=True)

build_real_imf(imf_ranges, imf_params, show=True)


# ******************
# DISTANCE EVOLUTION
# ******************

distances = analysis.distances

distance_evolution(distances, num_acc_sim=100, show=True)


# ********************
# PARAMETERS EVOLUTION
# ********************

for parameter in analysis.data.keys():
    parameter_evolution(parameter, analysis.data[parameter], num_acc_sim=100, show=True)
    
    
# ****************
# BUILD CORNERPLOT
# ****************

cornerplot(analysis.data, ranges='auto', num_acc_sim=100, show=True)


# ***************************
# BUILD ORIGINAL HESS DIAGRAM
# ***************************

hess_diagram_analysis = compare_hess_diagrams()

filename_catalog = "./input_data/catalog/Gaia_DR3_G13.csv"
filename_ms = "./input_data/ms/ms_G13_err.csv"

colnames_catalog = parameters.catalog_file_parameters['sel_columns_catalog'].value
colnames_ms = {'G':'Gerr', 'color': 'GRperr', 'longitude': 'longitude', 'latitude': 'latitude', 'Mvarpi': 'Mvarpi', 'parallax': 'parallaxerr'}

catalog_cmd, catalog_data = hess_diagram_analysis.generate_catalog_hess_diagram(filename_catalog, colnames=colnames_catalog)
ms_cmd, ms_data = hess_diagram_analysis.generate_catalog_hess_diagram(filename_ms, colnames=colnames_ms)

distance_cmd, difference_cmd, quocient_cmd = hess_diagram_analysis.generate_difference_hess_diagram(ms_cmd, catalog_cmd)

hess_diagram_analysis.build_hess_diagrams_plots(ms_cmd, catalog_cmd, distance_cmd, difference_cmd, quocient_cmd, titles=['MS', 'Gaia DR3', r'$\delta_P$(MS, Gaia DR3)', 'MS - Gaia DR3', 'MS/Gaia DR3'], limits='auto')

hess_diagram_analysis.compute_distance(catalog_data, ms_data)


# *******************************************
# BUILD FINAL HESS DIAGRAM AND MASS-AGE SPACE
# *******************************************

bgmfast_sim = bgmfast_simulation(logfile=False)

sc, spark = bgmfast_sim.open_spark_session()
spark.sparkContext.setLogLevel("WARN")

bgmfast_sim.set_acc_parameters()
bgmfast_sim.set_binning_parameters()
bgmfast_sim.set_general_parameters()
bgmfast_sim.set_ms_parameters()
bgmfast_sim.set_ps_parameters()
bgmfast_sim.set_constraints_parameters()
bgmfast_sim.set_bgmfast_parameters()

filename_ms = "./input_data/ms/ms_G13_err.csv" #set parquet='generate'
#filename_ms = "./input_data/ms/ms_G13_err_reduced.parquet" #once generated, set parquet='open'
sel_columns_ms = parameters.ms_file_parameters['sel_columns_ms'].value
Gmax_ms = parameters.ms_file_parameters['Gmax_ms'].value
bgmfast_sim.read_ms(filename_ms, sel_columns_ms, Gmax_ms, parquet='generate', num_partitions=100)

final_parameters_no_corr = [value for key, value in final_params(analysis.data_no_corr, num_acc_sim=100, show=True, step_range=[0, 100]).items()]
param = [1.0, 
         final_parameters_no_corr[0][1], 
         final_parameters_no_corr[1][1], 
         final_parameters_no_corr[2][1], 
         final_parameters_no_corr[3][1], 
         final_parameters_no_corr[4][1], 
         final_parameters_no_corr[5][1], 
         final_parameters_no_corr[6][1], 
         final_parameters_no_corr[7][1], 
         final_parameters_no_corr[8][1], 
         final_parameters_no_corr[9][1], 
         final_parameters_no_corr[10][1], 
         final_parameters_no_corr[11][1], 
         final_parameters_no_corr[12][1], 
         final_parameters_no_corr[13][1], 
         final_parameters_no_corr[14][1], 
         final_parameters_no_corr[14][1], 
         final_parameters_no_corr[14][1]]

simulation_data = bgmfast_sim.run_simulation(param)
simulation_cmd, simulation_data = bgmfast_sim.return_cmd()

hess_diagram_analysis = compare_hess_diagrams()

distance_cmd, difference_cmd, quocient_cmd = hess_diagram_analysis.generate_difference_hess_diagram(simulation_cmd, catalog_cmd)

hess_diagram_analysis.build_hess_diagrams_plots(simulation_cmd, catalog_cmd, distance_cmd, difference_cmd, quocient_cmd, titles=['BGM FASt', 'Gaia DR3', r'$\delta_P$(BGM FASt, Gaia DR3)', 'BGM FASt - Gaia DR3', 'BGM FASt/Gaia DR3'], limits=limits)

hess_diagram_analysis.compute_distance(catalog_data, simulation_data)

hess_diagram_analysis.build_mass_age_space(bgmfast_sim.smallacc.value, mass_range=[0, 120], mass_bins=20, show=True)


# *****************************
# BUILD ORIGINAL MASS-AGE SPACE
# *****************************

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

bgmfast_sim.run_simulation(param)
hess_diagram_analysis.build_mass_age_space(bgmfast_sim.smallacc.value, mass_range=[0, 120], mass_bins=20, show=True)
    
    
    
