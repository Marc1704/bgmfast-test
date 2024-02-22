'''
Set inputs for BGM FASt

This script is intended to put the original Mother Simulation file in the proper format for BGM FASt.
'''

from astropy.io import fits
from astropy.table import Table
import pandas as pd
import numpy as np
from bgmfast.parameters import popbin_parameters


class set_input_for_bgmfast:
    '''
    Set the Mother Simulation file ready for BGM FASt
    '''

    def __init__(self, filename, fileformat='fits', filetype='all'):
        '''
        Import the Mother Simulation file

        Input parameters
        ----------------
        filename : str --> directory of the file. Example: /home/username/bgmfast/inputs/ms_G13_errors.csv
        fileformat: str ['fits', 'csv'] --> format of the file
        filetype : str ['all', 'nwd', 'wd'] --> what kind of objects are contained in the file: all of them (all), no white dwarfs (nwd) or only white dwarfs (wd). This will also become the name of the table for future actions. See complementary description in white_dwarfs_filter function
        '''

        self.filetype = 'all'
        self.df = {'all': 'Empty', 'nwd': 'Empty', 'wd': 'Empty'}

        if fileformat=='fits':
            hdul = fits.open(filename, memmap=True)
            hdu = hdul[1]
            table = Table(hdu.data)
            self.df[self.filetype] = table.to_pandas()
        elif fileformat=='csv':
            self.df[self.filetype] = pd.read_csv(filename)
        elif fileformat=='pd':
            self.df[self.filetype] = filename
        else:
            print('Format %s not recognized' %fileformat)


    def select_columns(self, columns, filetype='all'):
        '''
        Select columns from the Mother Simulation file

        Input parameters
        ----------------
        columns : str or list --> name of the columns we want to keep from the file
        filetype : str ['all', 'nwd', 'wd'] --> name of the table into which we want to apply the selection of columns. See complementary description in __init__ function
        '''

        self.filetype = filetype

        if type(columns)==type(''):
            self.df[self.filetype] = self.df[self.filetype][[columns]]
        elif type(columns)==type([]):
            self.df[self.filetype] = self.df[self.filetype][columns]
        else:
            print('Not valid columns input')


    def change_column_name(self, old_columns, new_columns, filetype='all'):
        '''
        Change the names of the columns in the Mother Simulation file

        Input parameters
        ----------------
        old_columns : str or list --> original names of the columns to be changed
        new_columns : str or list --> new names of the columns to be changed
        filetype : str ['all', 'nwd', 'wd'] --> see description in select_columns function
        '''

        self.filetype = filetype

        if type(old_columns)==type(''):
            self.df[self.filetype][new_columns] = self.df[self.filetype][old_columns]
            self.df[self.filetype].drop(columns=[old_columns], inplace=True)

        elif type(old_columns)==type([]):
            for oldc, newc in zip(old_columns, new_columns):
                self.df[self.filetype][newc] = self.df[self.filetype][oldc]
            self.df[self.filetype].drop(columns=old_columns, inplace=True)

        else:
            print('old_columns input not recognized')


    def basic_filter(self, colnames={'Parallax': 'parallaxerr', 'Color': 'GRperr'}, filetype='all'):
        '''
        Remove stars without value of parallax or G-Rp colour or with negative values of parallax

        Input parameters
        ----------------
        colnames : dict --> name/s of the column/s containing the parallax and the Bp-Rp or G-Rp colour
        filetype : str ['all', 'nwd', 'wd'] --> see description in select_columns function
        '''

        self.filetype = filetype
        
        if 'Parallax' in colnames.keys() and 'Color' in colnames.keys():
            self.df[self.filetype] = self.df[self.filetype][(self.df[self.filetype][colnames['Parallax']]!='') & (self.df[self.filetype][colnames['Parallax']]>0) & (self.df[self.filetype][colnames['Color']]!='')]
        elif 'Parallax' in colnames.keys():
            self.df[self.filetype] = self.df[self.filetype][(self.df[self.filetype][colnames['Parallax']]!='') & (self.df[self.filetype][colnames['Parallax']]>0)]
        elif 'Color' in colnames.keys():
            self.df[self.filetype] = self.df[self.filetype][(self.df[self.filetype][colnames['Color']]!='')]


    def compute_absolute_magnitude(self, colnames={'G': 'Gerr', 'Parallax': 'parallaxerr'}, filetype='all'):
        '''
        Compute absolute magnitudes and add them in a new column

        Input parameters
        ----------------
        colnames : list --> name of the columns containing the G magnitude and the parallax
        filetype : str ['all', 'nwd', 'wd'] --> see description in select_columns function
        '''
        self.filetype = filetype

        M_G_prime = lambda G, parallax: G + 5*np.log10(parallax/1000) + 5

        self.df[self.filetype]['Mvarpi'] = M_G_prime(self.df[self.filetype][colnames['G']], self.df[self.filetype][colnames['Parallax']])


    def compute_popbin(self, 
                       thin_disc_limits=popbin_parameters['thin_disc_limits'].value,
                       bgm_popbin=popbin_parameters['bgm_popbin'].value,
                       gaia_popbin=popbin_parameters['gaia_popbin'].value,
                       colnames={'Popbin': 'PopBin', 'Age': 'Age'},
                       filetype='all'):
        '''
        Compute PopBins from age and add them in a new column

        Input parameters
        ----------------
        thin_disc_limits : list --> age limits of the thin disk subpopulations used for the generation of the Mother Simulation
        bgm_popbin : dict --> relation between the name and the PopBin of the populations used for the generation of the Mother Simulation
        gaia_popbin : dict --> relation between the name and the PopBin(s) of the populations according to Gaia
        filetype : str ['all', 'nwd', 'wd'] --> see description in select_columns function
        '''

        self.filetype = filetype

        bgm_popbin_keys = list(bgm_popbin.keys())
        bgm_popbin_values = list(bgm_popbin.values())

        new_popbin = []
        for index, row in self.df[self.filetype].iterrows():
            popbin = int(row[colnames['Popbin']])

            if popbin==bgm_popbin['thin_disk']:
                age = float(row[colnames['Age']])

                for i in range(len(gaia_popbin['thin_disk'])):
                    if age==0:
                        new_popbin.append(gaia_popbin['thin_disk'][0])
                        break
                    elif thin_disc_limits[i]<age<=thin_disc_limits[i+1]:
                        new_popbin.append(gaia_popbin['thin_disk'][i])
                        break

            else:
                popbin_name = bgm_popbin_keys[bgm_popbin_values.index(popbin)]
                new_popbin.append(gaia_popbin[popbin_name])

        self.df[self.filetype]['OldPopBin'] = self.df[self.filetype]['PopBin']
        self.df[self.filetype]['PopBin'] = new_popbin


    def filter_by_V(self, colnames=['Gerr', 'BpRperr'], filetype='all'):

        '''
        Obtain V magnitudes from G magnitudes and Bp-Rp colours (https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu5pho/cu5pho_sec_photSystem/cu5pho_ssec_photRelations.html) and add them in a new column

        Input parameters
        ----------------
        colnames : list --> name of the columns containing the G magnitude and the Bp-Rp colour
        filetype : str ['all', 'nwd', 'wd'] --> see description in select_columns function
        '''

        G_V = lambda bprp: -0.02704 + 0.01424*bprp - 0.2156*bprp**2 + 0.01426*bprp**3
        V = lambda G, bprp: G - G_V(bprp)

        self.df[self.filetype]['V'] = V(self.df[self.filetype][colnames[0]], self.df[self.filetype][colnames[1]])


    def white_dwarfs_filter(self, limits=[[-1, 6], [4, 20]], colnames=['Mvarpi', 'BpRperr'], filetype='all'):
        '''
        Substract white dwarfs from the Mother Simulation and generate two new tables just with them or without them
        WARNING: function not recommended

        Input parameters
        ----------------
        limits : list --> range of Bp-Rp colours and absolute magnitudes in the HR diagram defining the location of the white dwarfs
        colnames : list --> name of the columns containing the absolute magnitude and the Bp-Rp colour
        filetype : str ['all', 'nwd', 'wd'] --> see description in select_columns function
        '''

        self.filetype = filetype

        #M_G' = a + b*color(bp-rp)
        b = (limits[0][1] - limits[1][1])/(limits[0][0] - limits[1][0])
        a = limits[0][1] - limits[0][0]*b
        f = lambda x: a + b*x

        self.df['wd'] = self.df[self.filetype][self.df[self.filetype][colnames[0]]>f(self.df[self.filetype][colnames[1]])]
        self.df['nwd'] = self.df[self.filetype][self.df[self.filetype][colnames[0]]<f(self.df[self.filetype][colnames[1]])]


    def save(self, output_file, columns='all', filetype='all'):
        '''
        Save a pandas dataframe as a CSV file

        Input parameters
        ----------------
        output_file : str --> directory of the output file. Example: /home/username/bgmfast/inputs/ms_G13_errors_bgmfast.csv
        columns: str ['all'] or list --> whether we want to save all columns or just some of them
        filetype : str ['all', 'nwd', 'wd'] --> see description in select_columns function
        '''

        self.filetype = filetype

        if columns!='all':
            self.df[self.filetype].to_csv(output_file, columns=columns, index=False)
        else:
            self.df[self.filetype].to_csv(output_file, index=False)
