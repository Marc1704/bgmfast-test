'''
Gaia instrument model filter

This script is intended to apply the Gaia instrument error model (
) to the Mother Simulation generated with BGM to obtain an artificially affected catalog.
'''


import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.table import Table


def G_error(G):
    '''
    Affect G magnitude of errors

    Input parameters
    ----------------
    G : int or float --> G magnitude not affected of errors

    Output parameters
    -----------------
    Gerr : float --> G magnitude affected of errors

    '''

    if G<13:
        Gerr = np.random.normal(G, 0.0002)
    else:
        Gerr = np.random.normal(G, 0.0003)

    return Gerr


def parallax_sigma(Gerr, T_factor=1):
    '''
    Generate parallax standard deviation

    Input parameters
    ----------------
    Gerr : int or float --> G magnitude affected of errors
    T_factor : int or float --> temporal improvement factor in the parallax uncercainty allowed by adding more data that span a longer time interval: T_factor = sqrt(34/num_months), where the value 34 comes from the number of months corresponding to DR3 and num_months is the actual number of months (for DR3 34, for DR4 66 and for DR5 132)

    Output parameters
    -----------------
    sigma : float --> parallax standard deviation
    '''

    z = max([10**(0.4*(13 - 15)), 10**(0.4*(Gerr - 15))])
    sigma = T_factor*np.sqrt(40 + 800*z + 30*z**2)
    sigma = sigma/1000

    return sigma


def bprp_error(Gerr, bprp):
    '''
    Affect Bp-Rp colour of errors

    Input parameters
    ----------------
    Gerr : int or float --> G magnitude affected of errors
    bprp : int or float --> Bp-Rp colour not affected of errors

    Output parameters
    -----------------
    bprp_err : float --> Bp-Rp colour affected of errors
    '''

    if Gerr<13:
        bprp_err = np.random.normal(bprp, 0.002)
    else:
        bprp_err = np.random.normal(bprp, 0.006)

    return bprp_err


def grp_error(Gerr, grp):
    '''
    Affect G-Rp colour of errors

    Input parameters
    ----------------
    Gerr : int or float --> G magnitude affected of errors
    grp : int or float --> G-Rp colour not affected of errors

    Output parameters
    -----------------
    grp_err : float --> G-Rp colour affected of errors
    '''

    if Gerr<13:
        grp_err = np.random.normal(grp, 0.002)
    else:
        grp_err = np.random.normal(grp, 0.006)

    return grp_err


def parallax_error(parallax, Gerr):
    '''
    Affect parallax of errors

    Input parameters
    ----------------
    parallax : int or float --> parallax not affected of errors
    Gerr : int or float --> G magnitude affected of errors

    Output parameters
    -----------------
    parallax_err : float --> parallax affected of errors
    '''

    if Gerr<3:
        parallax_err = np.random.normal(parallax, parallax_sigma(13))
    else:
        parallax_err = np.random.normal(parallax, parallax_sigma(Gerr))

    return parallax_err


class Gaia_instrument_model:
    '''
    Affect G magnitude, G-Rp colour, and parallax of errors the magnitudes of a Mother Simulation file
    '''

    def __init__(self, filename, fileformat='fits', all_magnitudes=True, colnames=['G', 'G_RP', 'Parallax'], T_factor=1):
        '''
        Import the Mother Simulation file

        Input parameters
        ----------------
        filename : str --> directory of the file. Example: /home/username/bgmfast/inputs/ms_G13.csv
        fileformat: str ['fits', 'csv'] --> format of the file
        all_magnitudes : True, str ['G', 'GRp', 'parallax'] or list ['G', 'GRp', 'parallax'] --> whether if we want to affect errors to all magnitudes or just some of them
        colnames : list --> name of the columns corresponding to the G magnitude, G-Rp colour and parallax
        T_factor : int or float --> see description in parallax_sigma function
        '''

        self.all_magnitudes = all_magnitudes
        self.G_colname, self.grp_colname, self.parallax_colname = colnames

        if fileformat=='fits':
            hdul = fits.open(filename, memmap=True)
            hdu = hdul[1]
            table = Table(hdu.data)
            self.df = table.to_pandas()
        elif fileformat=='csv':
            self.df = pd.read_csv(filename)
        else:
            print('MS format %s not recognized' %fileformat)


    def select_columns(self, columns):
        '''
        Select columns from the Mother Simulation file

        Input parameters
        ----------------
        columns : str or list --> name of the columns we want to keep from the file
        '''

        if type(columns)==type(''):
            self.df = self.df[[columns]]
        elif type(columns)==type([]):
            self.df = self.df[columns]
        else:
            print('Not valid columns input')


    def affect_errors(self, all_magnitudes=True, output_colnames=['Gerr', 'GRperr', 'parallaxerr']):
        '''
        Affect of errors the corresponding magnitudes

        Input parameters
        ----------------
        all_magnitudes : True, str ['G', 'GRp', 'parallax'] or list ['G', 'GRp', 'parallax'] --> whether we want to affect of errors all magnitudes or just some of them
        output_colnames : str or list --> name of the columns containing the magnitudes affected of errors

        Output parameters
        -----------------
        self.df : pandas dataframe --> table containing the magnitudes affected of errors
        '''

        self.Gerr_colname, self.GRperr_colname, self.Parallaxerr_colname = output_colnames

        if all_magnitudes:
            self.df[self.Gerr_colname] = self.df[[self.G_colname]].apply(lambda x: G_error(x[self.G_colname]), axis=1)
            self.df[self.GRperr_colname] = self.df[[self.Gerr_colname, self.grp_colname]].apply(lambda x: grp_error(x[self.Gerr_colname], x[self.grp_colname]), axis=1)
            self.df[self.Parallaxerr_colname] = self.df[[self.Gerr_colname, self.parallax_colname]].apply(lambda x: parallax_error(x[self.parallax_colname], x[self.Gerr_colname]), axis=1)

        else:
            if 'G' in all_magnitudes:
                self.df[self.Gerr_colname] = self.df[[self.G_colname]].apply(lambda x: G_error(x[self.G_colname]), axis=1)
            if 'GRp' in all_magnitudes and 'G' in all_magnitudes:
                self.df[self.GRperr_colname] = self.df[[self.Gerr_colname, self.grp_colname]].apply(lambda x: grp_error(x[self.Gerr_colname], x[self.grp_colname]), axis=1)
            elif 'GRp' in all_magnitudes and 'G' not in all_magnitudes:
                self.df[self.GRperr_colname] = self.df[[self.G_colname, self.grp_colname]].apply(lambda x: grp_error(x[self.G_colname], x[self.grp_colname]), axis=1)
            if 'parallax' in all_magnitudes and 'G' in all_magnitudes:
                self.df[self.Parallaxerr_colname] = self.df[[self.Gerr_colname, self.parallax_colname]].apply(lambda x: parallax_error(x[self.parallax_colname], x[self.Gerr_colname]), axis=1)
            elif 'parallax' in all_magnitudes and 'G' not in all_magnitudes:
                self.df[self.Parallaxerr_colname] = self.df[[self.G_colname, self.parallax_colname]].apply(lambda x: parallax_error(x[self.parallax_colname], x[self.G_colname]), axis=1)

            return self.df


    def save(self, output_file, columns='all'):
        '''
        Save a pandas dataframe as a CSV file

        Input parameters
        ----------------
        output_file : str --> directory of the output file. Example: /home/username/bgmfast/inputs/ms_G13_errors.csv
        columns: str ['all'] or list --> whether we want to save all columns or just some of them
        '''

        if columns!='all':
            self.df.to_csv(output_file, columns=columns, index=False)
        else:
            self.df.to_csv(output_file, index=False)




