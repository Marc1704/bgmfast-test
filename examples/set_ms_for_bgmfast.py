'''
Setting MS for BGM FASt

This script is an example on how to affect of errors the Mother Simulation data, compute the absolute magnitude and the PopBin, and set it for BGM FASt.
'''

# ***************
# BGMFAST IMPORTS
# ***************

from bgmfast.set_inputs_for_bgmfast import set_input_for_bgmfast
from bgmfast.Gaia_instrument_model_filter import Gaia_instrument_model

# ******************************************************
# NAME OF THE MOTHER SIMULATION FILE AND THE OUTPUT FILE
# ******************************************************

ms_in_name = './input_data/ms/ms_G13.csv'
ms_out_name= './input_data/ms/ms_G13_err.csv'

# **************************************
# AFFECT OF ERRORS THE MOTHER SIMULATION
# **************************************

#Create a Gaia instrument model class object
ms_file = Gaia_instrument_model(ms_in_name, fileformat='csv')

#Select the necessary columns from the Mother Simulation
sel_columns_ms = ['G', 'BP_RP', 'G_RP', 'Parallax', 'PopBin', 'Age', 'longitude', 'latitude', 'IniMass']
ms_file.select_columns(sel_columns_ms)

#Affect of errors the Mother Simulation
ms_file.affect_errors()

# ***************************************************************************
# COMPUTE THE ABSOLUTE MAGNITUDE AND THE POPBIN, AND SET THE FILE FOR BGMFAST
# ***************************************************************************

#Create a set input for bgmfast class object setting as input the Mother Simulation affected of errors
ms_file = set_input_for_bgmfast(ms_file.df, fileformat='pd')

#Select the necessary columns
sel_columns_ms_err = ['Gerr', 'BpRperr', 'GRperr', 'parallaxerr', 'PopBin', 'Age', 'longitude', 'latitude', 'IniMass']
ms_file.select_columns(sel_columns_ms_err)

#Change the name of some columns to set the file for BGM FASt
old_columns_names_ms = ['IniMass']
new_columns_names_ms = ['MassOut']
ms_file.change_column_name(old_columns_names_ms, new_columns_names_ms)

#Compute absolute magnitudes
ms_file.compute_absolute_magnitude(colnames={'G': 'Gerr', 'Parallax': 'parallaxerr'})

#Compute PopBins
ms_file.compute_popbin()

#Save the final Mother Simulation affected of errors, with the necessary magnitudes and with the proper columns names
sel_columns_ms_final = ['Gerr', 'BpRperr', 'GRperr', 'PopBin', 'Age', 'longitude', 'latitude', 'parallaxerr', 'MassOut', 'Mvarpi']
ms_file.save(ms_out_name, columns=sel_columns_ms_final)
