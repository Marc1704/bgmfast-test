'''
Set MS for BGM FASt

This script is an example on how to join the Mother Simulation FITS files.
'''

# ***************
# GENERAL IMPORTS
# ***************

import os, glob

# ***************
# BGMFAST IMPORTS
# ***************

from bgmfast.set_inputs_for_bgmfast import set_input_for_bgmfast

# **************************************************************
# NAME OF THE MOTHER SIMULATION FILES FOLDER AND THE OUTPUT FILE
# **************************************************************

ms_in_folder = './input_data/ms/'
ms_out_dir = './input_data/ms/ms_G13.csv'

#Make a list with the names of the folders containing the different Mother Simulation files
folders = os.listdir(ms_in_folder)
folders = [item for item in folders if os.path.isdir(os.path.join(ms_in_folder, item))]
print(folders)

#Open the file into which we will put all the Mother Simulation
all_ms = open(ms_out_dir, 'w')

#Add each one of the files into the general Mother Simulation file
for folder in folders:
    filename = glob.glob(ms_in_folder + folder + '/*.fits')[0]
    tempname = filename.split('.')[0] + '.temp'
    print(filename)

    sel_columns_ms = ['G', 'BP_RP', 'G_RP', 'Parallax', 'PopBin', 'Age', 'longitude', 'latitude', 'IniMass']
    ms_file = set_input_for_bgmfast(filename, fileformat='fits')
    ms_file.select_columns(sel_columns_ms)
    ms_file.save(tempname)

    file_from_csv = open(tempname, 'r')
    counter = 0
    for line in file_from_csv.readlines():
        counter += 1
        if counter==1:
            if folder==folders[0]:
                all_ms.write(line)
                continue
            else:
                continue
        all_ms.write(line)
    os.remove(tempname)
