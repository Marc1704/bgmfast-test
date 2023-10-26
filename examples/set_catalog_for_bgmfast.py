'''
Setting catalog for BGM FASt

This script is an example on how to set the catalog file for BGM FASt.
'''

# ***************
# BGMFAST IMPORTS
# ***************

from bgmfast.set_inputs_for_bgmfast import set_input_for_bgmfast

# ********************************************
# NAME OF THE CATALOG FILE AND THE OUTPUT FILE
# ********************************************

catalog_in_name = './input_data/catalog/archive_Gaia_DR3_G13.csv'
catalog_out_name= './input_data/catalog/Gaia_DR3_G13.csv'

# ************************
# SET THE FILE FOR BGMFAST
# ************************

#Create a set input for bgmfast class object setting as input the catalog file
catalog_file = set_input_for_bgmfast(catalog_in_name, fileformat='csv')

#Select the necessary columns
sel_columns_catalog = ['phot_g_mean_mag', 'bp_rp', 'l', 'b', 'parallax']
catalog_file.select_columns(sel_columns_catalog)
print(catalog_file.df['all'])

#Change the name of some columns to set the file for BGM FASt
old_columns_names = sel_columns_catalog[:-1]
desired_columns = ['G', 'BpRp', 'longitude', 'latitude']
catalog_file.change_column_name(old_columns_names, desired_columns)

#Apply the basic data filter
print(catalog_file.df['all'])
catalog_file.basic_filter(colnames=['parallax', 'BpRp'])

#Compute absolute magnitudes
catalog_file.compute_absolute_magnitude(colnames=['G', 'parallax'])

#Save the final catalog with absolute magnitudes and the proper columns names
final_columns = desired_columns.append('Mvarpi')
catalog_file.save(catalog_out_name, columns=final_columns, filetype='all')




