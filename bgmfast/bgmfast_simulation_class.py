'''
BGM FASt simulation class

This script is intented to include all the functions working within the Pyspark environment.
'''


import numpy as np
import time, sys
import pandas as pd
import ast
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.accumulators import AccumulatorParam

from bgmfast.auxiliary_functions import *
from bgmfast.parameters import acc_parameters, binning_parameters, general_parameters, ms_parameters, ps_parameters, constraints_parameters, bgmfast_parameters


class MatrixAccumulatorParam(AccumulatorParam):

    '''
    Define a matrix accumulator of 4 dimensions

    Input parameters
    ----------------
    AccumulatorParam : pyspark.accumulators.AccumulatorParam --> Pyspark parameter needed for the definition of Pyspark accumulator matrixs
    '''

    def zero(self, inimatriu):
        '''
        Define a matrix of full of zeros

        Input parameters
        ----------------
        inimatriu : numpy array --> numpy array with the shape we want for the Pyspark accumulator

        Output parameters
        -----------------
        MATRIXINI : numpy array --> numpy array with the shape we want for the Pyspark accumulator full of zeros
        '''

        MATRIXINI = np.zeros(inimatriu.shape)

        return MATRIXINI


    def addInPlace(self, mAdd, sindex):
        '''
        Add an element to the accumulator in a given place

        Input parameters
        ----------------
        mAdd : Pyspark accumulator? --> accumulator into which we want to add an element
        sindex : list --> list with one or five elements containing the coordinates in the 4-dimensional (latitude, longitude, G-Rp and M_G') space of the Pyspark accumulator into which we want to put the element. The last element defines the value of the weight we want to add in that coordinates

        Output parameters
        -----------------
        mAdd : Pyspark accumulator? --> updated mAdd
        '''

        if type(sindex)==list:
            mAdd[sindex[0], sindex[1], sindex[2], sindex[3] ] += sindex[4]
        else:
            mAdd += sindex

        return mAdd
    
    
class SmallMatrixAccumulatorParam(AccumulatorParam):

    '''
    Define a matrix accumulator of 2 dimensions

    Input parameters
    ----------------
    AccumulatorParam : pyspark.accumulators.AccumulatorParam --> Pyspark parameter needed for the definition of Pyspark accumulator matrixs
    '''

    def zero(self, inimatriu):
        '''
        Define a matrix of full of zeros

        Input parameters
        ----------------
        inimatriu : numpy array --> numpy array with the shape we want for the Pyspark accumulator

        Output parameters
        -----------------
        MATRIXINI : numpy array --> numpy array with the shape we want for the Pyspark accumulator full of zeros
        '''

        MATRIXINI = np.zeros(inimatriu.shape)

        return MATRIXINI


    def addInPlace(self, mAdd, sindex):
        '''
        Add an element to the accumulator in a given place

        Input parameters
        ----------------
        mAdd : Pyspark accumulator? --> accumulator into which we want to add an element
        sindex : list --> list with one or three elements containing the coordinates in the 2-dimensional (mass, popbin) space of the Pyspark accumulator into which we want to put the element. The last element defines the value we want to add in that coordinates

        Output parameters
        -----------------
        mAdd : Pyspark accumulator? --> updated mAdd
        '''

        if type(sindex)==list:
            mAdd[sindex[0], sindex[1]] += sindex[2]
        else:
            mAdd += sindex

        return mAdd


# ****************
# WEIGHT FUNCTIONS
# ****************

def pes_catalog(x, Xmin, Xmax, Ymin, Ymax, Bmin, Bmax, Lmin, Lmax, blims, llims, Xstep, Ystep, acc, simple):

    '''
    Build the Hess diagram of the observed catalog using a Pyspark accumulator

    Input parameters
    ----------------
    x : list --> each one of the rows in the Mother Simulation file
    Xmin : int or float --> minimum value for the binning in G-Rp range
    Xmax : int or float --> maximum value for the binning in G-Rp range
    Ymin : int or float --> minimum value for the binning in M_G' range
    Ymax : int or float --> maximum value for the binning in M_G' range
    Bmin : int or float --> minimum value for the binning in latitude
    Bmax : int or float --> maximum value for the binning in latitude
    Lmin : int or float --> minimum value for the binning in longitude
    Lmax : int or float --> maximum value for the binning in longitude
    blims : list --> limits of the different absolute latitude ranges
    llims : list --> limits of the different longitude ranges
    Xstep : list --> G-Rp steps of the different G-Rp ranges
    Ystep : list --> M_G' steps of the different G-Rp ranges
    acc : pyspark accumulator --> 4-dimensional Pyspark accumulator (Hess diagram + latitude + longitude) containing the complete Hess diagram
    simple : pyspark accumulator --> Pyspark simple accumulator that counts the stars that are not within the considered ranges or that have suffered some problem during the computations

    Output parameters
    -----------------
    cpes : int --> weight of the star. In this case, since we are dealing with the observed data, the value of the weight is always 1
    '''

    GRp = float(x[1])
    longitude = float(x[2])
    latitude = float(x[3])
    Mvarpi = float(x[4])

    xinput = [GRp, longitude, latitude, Mvarpi]
    matindex = binning_4D_Mvarpi(xinput, Xmin, Xmax, Ymin, Ymax, Bmin, Bmax, Lmin, Lmax, blims, llims, Xstep, Ystep)

    cpes=1

    if (np.isnan(matindex[0]) or np.isnan(matindex[1]) or np.isnan(matindex[2]) or np.isnan(matindex[3])):
        simple.add(1)
    else:
        acc.add([int(matindex[0]),int(matindex[1]),int(matindex[2]),int(matindex[3]),cpes])

    return cpes


def wpes_func(WP, x1, x2_ps, x3_ps, K1_ps, K2_ps, K3_ps, alpha1_ps, alpha2_ps, alpha3_ps, SigmaParam_ps, midpopbin_ps, lastpopbin_ps, bin_nor_ps, x2_ms, x3_ms, K1_ms, K2_ms, K3_ms, alpha1_ms, alpha2_ms, alpha3_ms, SigmaParam_ms, midpopbin_ms, lastpopbin_ms, bin_nor_ms, ThickParamYoung, HaloParam, BarParam, ThickParamOld, mass_min, mass_step, tau_ranges, acc, smallacc):

    '''
    Compute the weight of a given mass-age bin and assign it to all the stars within the bin. It uses Equation (37) from Mor et al. 2018 without integrating. The integral is conceptual, because it defines the integration over an increment (bin) of the N-dimensional space defined in Eq. (6). We reduce this increment until the end, when we only consider the star itself. At that point, the increment is exactly equal to the differential (both of them are the star itself) and the integral blows up.

    Input parameters
    ----------------
    WP : list -->
    x1 : int or float --> minimum mass to generate a star
    x2_ps : int or float --> first mass limit of the IMF for the BGM FASt simulation
    x3_ps : int or float --> second mass limit of the IMF for the BGM FASt simulation
    K1_ps : int or float --> first continuity coeficient of the IMF for the BGM FASt simulation
    K2_ps : int or float --> second continuity coeficient of the IMF for the BGM FASt simulation
    K3_ps : int or float --> third continuity coeficient of the IMF for the BGM FASt simulation
    alpha1_ps : int or float --> first slope (alpha) of the IMF for the BGM FASt simulation
    alpha2_ps : int or float --> second slope (alpha) of the IMF for the BGM FASt simulation
    alpha3_ps : int or float --> third slope (alpha) of the IMF for the BGM FASt simulation
    SigmaParam_ps : list --> surface density at the position of the Sun for the different age subpopulations of the thin disc for the BGM FASt simulation. In case ThickParamYoung=='fit', SigmaParam_ps is a list with SigmaParam_ps and T_SigmaParam_ps
    midpopbin_ps : list --> surface density at the position of the Sun for the four subdivisions of the 5th and 6th age subpopulations of the thin disc (3-5 Gyr and 5-7 Gyr) for the BGM FASt simulation
    lastpopbin_ps : list --> surface density at the position of the Sun for the three subdivisions of the last (7th) age subpopulation of the thin disc (7-10 Gyr) for the BGM FASt simulation. In case ThickParamYoung=='fit', lastpopbin_ps is a list with lastpopbin_ps and T_lastpopbin_ps
    bin_nor_ps : int or float --> normalization coeficient for binaries for the BGM FASt simulation. In case ThickParamYoung=='fit', bin_nor_ps is a list with bin_nor_ps and T_bin_nor_ps
    x2_ms : int or float --> first mass limit of the IMF for the Mother Simulation
    x3_ms : int or float --> second mass limit of the IMF for the Mother Simulation
    K1_ms : int or float --> first continuity coeficient of the IMF for the Mother Simulation
    K2_ms : int or float --> second continuity coeficient of the IMF for the Mother Simulation
    K3_ms : int or float --> third continuity coeficient of the IMF for the Mother Simulation
    alpha1_ms : int or float --> first slope (alpha) of the IMF for the Mother Simulation
    alpha2_ms : int or float --> second slope (alpha) of the IMF for the Mother Simulation
    alpha3_ms : int or float --> third slope (alpha) of the IMF for the Mother Simulation
    SigmaParam_ms : list --> surface density at the position of the Sun for the different age subpopulations of the thin disc for the Mother Simulation. In case ThickParamYoung=='fit', SigmaParam_ms is a list with SigmaParam_ms and T_SigmaParam_ms
    midpopbin_ms : list --> surface density at the position of the Sun for the four subdivisions of the 5th and 6th age subpopulations of the thin disc (3-5 Gyr and 5-7 Gyr) for the Mother Simulation
    lastpopbin_ms : list --> surface density at the position of the Sun for the three subdivisions of the last (7th) age subpopulation of the thin disc (7-10 Gyr) for the Mother Simulation. In case ThickParamYoung=='fit', lastpopbin_ms is a list with lastpopbin_ms and T_lastpopbin_ms
    bin_nor_ms : int or float --> normalization coeficient for binaries for the Mother Simulation. In case ThickParamYoung=='fit', bin_nor_ms is a list with bin_nor_ms and T_bin_nor_ms
    ThickParamYoung : int or float --> weight of the stars in the Young Thick disc. Set to "fit" to compute it by adding SFH9T, SFH10T, SFH11T, and SFH12T to the galactic parameters to fit
    HaloParam : int or float --> weight of the stars in the Halo
    BarParam : int or float --> weight of the stars in the Bar
    ThickParamOld : int or float --> weight of the stars in the Old Thick disc
    mass_min : 
    mass_step : 
    tau_ranges : 
    acc : pyspark accumulator --> 4-dimensional Pyspark accumulator (Hess diagram + latitude + longitude) containing the complete Hess diagram
    smallacc : pyspark accumulator --> 2-dimensional Pyspark accumulator (Mass + Popbin)

    Output parameters
    -----------------
    wpes : int --> weight of the star derived from BGM FASt
    '''
    
    
    wpes = 1
    popbin = float(WP[0]) # BGM population bin
    tau = float(WP[1]) # Age of the star
    mass = float(WP[2]) # Mass of the star
    matindex = WP[3]
    
    if ThickParamYoung=='fit':
        SigmaParam_ps, T_SigmaParam_ps = SigmaParam_ps
        bin_nor_ps, T_bin_nor_ps = bin_nor_ps
        SigmaParam_ms, T_SigmaParam_ms = SigmaParam_ms
        bin_nor_ms, T_bin_nor_ms = bin_nor_ms

    if popbin<=7:
        if 3<=tau<=4:
            imidpoptau = 0
        elif 4<tau<=5:
            imidpoptau = 1
        elif 5<tau<=6:
            imidpoptau = 2
        elif 6<tau<=7:
            imidpoptau = 3
        else:
            imidpoptau = np.nan
        if 7<tau<=8:
            ilastpoptau = 0
        elif 8<tau<=9:
            ilastpoptau = 1
        elif 9<tau<=10:
            ilastpoptau = 2
        else:
            ilastpoptau = np.nan
            
    elif popbin==8 and ThickParamYoung=='fit':
        T_imidpoptau = np.nan
        T_ilastpoptau = np.nan
        T_midpopbin_ms = np.nan
        T_midpopbin_ps = np.nan
        T_lastpopbin_ms = np.nan
        T_lastpopbin_ps = np.nan
        if 8<=tau<=9:
            itau = 0
        elif 9<tau<=10:
            itau = 1
        elif 10<tau<=11:
            itau = 2
        elif 11<tau<=12:
            itau = 3
        else:
            itau = np.nan

    if popbin<=7:
        itau = int(popbin)-1

        PS = float(Simplified_Gi_Primal_func_NONP(itau, mass, x1, x2_ps, x3_ps, K1_ps, K2_ps, K3_ps, alpha1_ps, alpha2_ps, alpha3_ps, SigmaParam_ps, bin_nor_ps, midpopbin_ps, lastpopbin_ps, imidpoptau, ilastpoptau, structure='thin'))

        MS = float(Simplified_Gi_Primal_func_NONP(itau, mass, x1, x2_ms, x3_ms, K1_ms, K2_ms, K3_ms, alpha1_ms, alpha2_ms, alpha3_ms, SigmaParam_ms, bin_nor_ms, midpopbin_ms, lastpopbin_ms, imidpoptau, ilastpoptau, structure='thin'))

        if PS==0:
            wpes = 0
        elif MS==0:
            wpes = 1
        else:
            wpes = PS/MS
        
        S = [mass, popbin, tau]
        masstaubin = binning_2D(S, mass_min, mass_step, tau_ranges, ThickParamYoung)
        smallacc.add([int(masstaubin[0]), int(masstaubin[1]), wpes*len(matindex)])
    
    elif popbin==8 and ThickParamYoung=='fit':
        
        PS = float(Simplified_Gi_Primal_func_NONP(itau, mass, x1, x2_ps, x3_ps, K1_ps, K2_ps, K3_ps, alpha1_ps, alpha2_ps, alpha3_ps, T_SigmaParam_ps, T_bin_nor_ps, T_midpopbin_ps, T_lastpopbin_ps, T_imidpoptau, T_ilastpoptau, structure='youngthick'))

        MS = float(Simplified_Gi_Primal_func_NONP(itau, mass, x1, x2_ms, x3_ms, K1_ms, K2_ms, K3_ms, alpha1_ms, alpha2_ms, alpha3_ms, T_SigmaParam_ms, T_bin_nor_ms, T_midpopbin_ms, T_lastpopbin_ms, T_imidpoptau, T_ilastpoptau, structure='youngthick'))

        if PS==0:
            wpes = 0
        elif MS==0:
            wpes = 1
        else:
            wpes = PS/MS
            
        S = [mass, popbin, tau]
        masstaubin = binning_2D(S, mass_min, mass_step, tau_ranges, ThickParamYoung)
        smallacc.add([int(masstaubin[0]), int(masstaubin[1]), wpes*len(matindex)])

    elif popbin==8:
        wpes = ThickParamYoung

    elif popbin==9:
        wpes = HaloParam

    elif popbin==10:
        wpes = BarParam

    elif popbin==11:
        wpes = ThickParamOld
        
    else:
        print('Not correct popbin %i or age %f' %(popbin, tau))
        import sys
        sys.exit()
        
    for starbin in matindex:
        acc.add([int(starbin[0]),int(starbin[1]),int(starbin[2]),int(starbin[3]),wpes])
        
    return wpes


# *********************
# SIMULATION CLASS (C1)
# *********************

class bgmfast_simulation:
    '''
    Run the BGM FASt simulations
    '''

    def __init__(self, logfile=str(datetime.now().strftime("%Y_%m_%dT%H_%M_%S"))+'_bgmfast_sim.log'):
        '''
        Initialize the bgmfast_simulation class

        Input parameters
        ----------------
        logfile : str or False --> directory of the log file
        '''

        print('\n=======================================================================')
        print('\n****************** Welcome to BGM FASt version 0.1.0 ******************\n')
        print('=======================================================================')

        self.num_sim = 0
        self.logfile = logfile

        if self.logfile!=False:
            with open(logfile, 'w') as logs:
                logs.write('simulation_number,foreach_initialization_datetime,foreach_duration\n')

        pass


    def open_spark_session(self, setLogLevel="WARN"):

        '''
        Open the Spark Session

        Output parameters
        -----------------
        sc : spark.sparkContext --> Pyspark parameter needed for internal funcionality
        spark : pyspark.sql.session.SparkSession --> Pyspark parameter needed for internal funcionality
        '''
        print('\nOpening Spark Session...\n')

        self.spark = SparkSession.builder.appName("Strangis").getOrCreate()
        self.spark.sparkContext.setLogLevel(setLogLevel)
        print(self.spark)
        self.sc = self.spark.sparkContext

        return self.sc, self.spark


    def set_acc_parameters(self,
                           nLonbins=acc_parameters['nLonbins'].value,
                           nLatbins=acc_parameters['nLatbins'].value, nColorbins=acc_parameters['nColorbins'].value, nGbins=acc_parameters['nGbins'].value):

        '''
        Set accumulators parameters

        Input parameters
        ----------------
        nLonbins : int --> number of bins in longitude of the complete sample
        nLatbins : int --> number of bins in latitude of the complete sample
        nColorbins : int --> number of bins in G-Rp color of the complete sample
        nGbins : int --> number of bins in M_G' magnitude of the complete sample
        '''

        print('\nSetting accumulators parameters...\n')

        self.nLonbins = nLonbins
        self.nLatbins = nLatbins
        self.nColorbins = nColorbins
        self.nGbins = nGbins
        self.MatrixAccumulatorParam = MatrixAccumulatorParam
        self.SmallMatrixAccumulatorParam = SmallMatrixAccumulatorParam


    def set_binning_parameters(self,
                               Xmin=binning_parameters['Xmin'].value, Xmax=binning_parameters['Xmax'].value, Ymin=binning_parameters['Ymin'].value, Ymax=binning_parameters['Ymax'].value, Bmin=binning_parameters['Bmin'].value, Bmax=binning_parameters['Bmax'].value, Lmin=binning_parameters['Lmin'].value, Lmax=binning_parameters['Lmax'].value, blims=binning_parameters['blims'].value, llims=binning_parameters['llims'].value, Xstep=binning_parameters['Xstep'].value, Ystep=binning_parameters['Ystep'].value, 
                               mass_step=binning_parameters['mass_step'].value, 
                               popbin_list=binning_parameters['popbin_list'].value):

        '''
        Set binning parameters

        Input parameters
        ----------------
        Xmin : int or float --> minimum value for the binning in G-Rp range
        Xmax : int or float --> maximum value for the binning G-Rp range
        Ymin : int or float --> minimum value for the binning M_G' range
        Ymax : int or float --> maximum value for the binning M_G' range
        Bmin : int or float --> minimum value for the binning of latitude
        Bmax : int or float --> maximum value for the binning of latitude
        Lmin : int or float --> minimum value for the binning of longitude
        Lmax : int or float --> maximum value for the binning of longitude
        blims : list --> limits of the latitude in the different M_G' ranges
        llims : list --> limits of the longitude in the different M_G' ranges
        Xstep : list --> G-Rp steps of the different G-Rp colour ranges
        Ystep : list --> M_G' steps of the different G-Rp colour ranges
        mass_step : int or float --> Mass step for the definition of the reduced Mother Simulation
        popbin_list : list --> Gaia PopBin values
        '''

        print('\nSetting binning parameters...\n')

        self.Xmin = Xmin
        self.Xmax = Xmax
        self.Ymin = Ymin
        self.Ymax = Ymax
        self.Bmin = Bmin
        self.Bmax = Bmax
        self.Lmin = Lmin
        self.Lmax = Lmax
        self.blims = blims
        self.llims = llims
        self.Xstep = Xstep
        self.Ystep = Ystep
        self.mass_step = mass_step
        self.popbin_list = popbin_list


    def set_general_parameters(self,
                               x1=general_parameters['x1'].value, x4=general_parameters['x4'].value, tau_min_edges=general_parameters['tau_min_edges'].value, tau_max_edges=general_parameters['tau_max_edges'].value,
                               T_tau_min_edges=general_parameters['T_tau_min_edges'].value, T_tau_max_edges=general_parameters['T_tau_max_edges'].value,
                               tau_ranges=general_parameters['tau_ranges'].value,
                               T_tau_ranges=general_parameters['T_tau_ranges'].value,
                               ThickParamYoung=general_parameters['ThickParamYoung'].value, HaloParam=general_parameters['HaloParam'].value, BarParam=general_parameters['BarParam'].value, ThickParamOld=general_parameters['ThickParamOld'].value):

        '''
        Set general parameters

        Input parameters
        ----------------
        x1 : int or float --> minimum mass to generate a star
        x4 : int or float --> maximum mass to generate a star
        tau_min_edges : list --> lower limits of the age subpopulations of the thin disc
        tau_max_edges : list --> upper limits of the age subpopulations of the thin disc
        T_tau_min_edges : list --> lower limits of the age intervals of the young thick disc
        T_tau_max_edges : list --> upper limits of the age intervals of the young thick disc
        tau_ranges : list --> age ranges for of the non-parametric SFH of the Thin Disc
        T_tau_ranges : list --> age ranges for the non-parametric SFH of the Young Thick disc
        ThickParamYoung : int, float or str --> weight of the young thick disc stars. Set to "fit" to compute it by adding SFH9T, SFH10T, SFH11T, and SFH12T to the Galactic parameters to fit
        HaloParam : int or float --> weight of the halo stars
        BarParam : int or float --> weight of the bar stars
        ThickParamOld : int or float --> weight of the old thick disc stars
        '''

        print('\nSetting general parameters...\n')

        self.x1 = x1
        self.x4 = x4
        self.tau_min_edges = tau_min_edges
        self.tau_max_edges = tau_max_edges
        self.tau_ranges = tau_ranges
        self.ThickParamYoung = ThickParamYoung
        self.HaloParam = HaloParam
        self.BarParam = BarParam
        self.ThickParamOld = ThickParamOld
        
        if self.ThickParamYoung=='fit':
            self.T_tau_min_edges = T_tau_min_edges
            self.T_tau_max_edges = T_tau_max_edges
            self.T_tau_ranges = T_tau_ranges


    def set_ms_parameters(self,
                          x2_ms=ms_parameters['x2_ms'].value, x3_ms=ms_parameters['x3_ms'].value, alpha1_ms=ms_parameters['alpha1_ms'].value, alpha2_ms=ms_parameters['alpha2_ms'].value, alpha3_ms=ms_parameters['alpha3_ms'].value, SigmaParam_ms=ms_parameters['SigmaParam_ms'].value, midpopbin_ms=ms_parameters['midpopbin_ms'].value, lastpopbin_ms=ms_parameters['lastpopbin_ms'].value,
                          T_SigmaParam_ms=ms_parameters['T_SigmaParam_ms'].value):

        '''
        Set Mother Simulation parameters

        Input parameters
        ----------------
        x2_ms : int or float --> first mass limit of the IMF for the Mother Simulation
        x3_ms : int or float --> second mass limit of the IMF for the Mother Simulation
        alpha1_ms : int or float --> first slope (alpha) of the IMF for the Mother Simulation
        alpha2_ms : int or float --> second slope (alpha) of the IMF for the Mother Simulation
        alpha3_ms : int or float --> third slope (alpha) of the IMF for the Mother Simulation
        SigmaParam_ms : list --> surface density at the position of the Sun for the different age subpopulations of the thin disc for the Mother Simulation
        midpopbin_ms : list --> surface density at the position of the Sun for the four subdivisions of the 5th and 6th age subpopulations of the thin disc (3-5 Gyr and 5-7 Gyr) for the Mother Simulation
        lastpopbin_ms : list --> surface density at the position of the Sun for the three subdivisions of the last (7th) age subpopulation of the thin disc (7-10 Gyr) for the Mother Simulation
        T_SigmaParam_ms : list --> surface density at the position of the Sun for the age population of the young thick disc for the Mother Simulation
        '''

        print('\nSetting Mother Simulation parameters...\n')

        self.x2_ms = x2_ms
        self.x3_ms = x3_ms
        self.alpha1_ms = alpha1_ms
        self.alpha2_ms = alpha2_ms
        self.alpha3_ms = alpha3_ms
        self.SigmaParam_ms = SigmaParam_ms
        self.midpopbin_ms = midpopbin_ms
        self.lastpopbin_ms = lastpopbin_ms

        self.K1_ms, self.K2_ms, self.K3_ms = Continuity_Coeficients_func(self.alpha1_ms, self.alpha2_ms, self.alpha3_ms, self.x1, self.x2_ms, self.x3_ms, self.x4)

        self.bin_nor_ms = bin_nor_func(self.x1, self.x2_ms, self.x3_ms, self.x4, self.K1_ms, self.K2_ms, self.K3_ms, self.alpha1_ms, self.alpha2_ms, self.alpha3_ms, self.SigmaParam_ms, self.tau_min_edges, self.tau_max_edges)
        
        if self.ThickParamYoung=='fit':
            self.T_SigmaParam_ms = T_SigmaParam_ms
            self.T_bin_nor_ms = bin_nor_func(self.x1, self.x2_ms, self.x3_ms, self.x4, self.K1_ms, self.K2_ms, self.K3_ms, self.alpha1_ms, self.alpha2_ms, self.alpha3_ms, self.T_SigmaParam_ms, self.T_tau_min_edges, self.T_tau_max_edges)


    def set_ps_parameters(self,
                          x2_ps=ps_parameters['x2_ps'].value, x3_ps=ps_parameters['x3_ps'].value):

        '''
        Set Pseudo Simulation (BGM FASt) parameters

        Input parameters
        ----------------
        x2_ps : int or float --> first mass limit of the IMF for the BGM FASt simulation
        x3_ps : int or float --> second mass limit of the IMF for the BGM FASt simulation
        '''

        print('\nSetting BGMFASt simulation (pseudo-simulation) parameters...\n')

        self.x2_ps = x2_ps
        self.x3_ps = x3_ps


    def set_constraints_parameters(self,
                                   tau_min=constraints_parameters['tau_min'].value, tau_max=constraints_parameters['tau_max'].value,
                                   T_tau_min=constraints_parameters['T_tau_min'].value, T_tau_max=constraints_parameters['T_tau_max'].value,
                                   mass_min=constraints_parameters['mass_min'].value, mass_max=constraints_parameters['mass_max'].value, l_min=constraints_parameters['l_min'].value, l_max=constraints_parameters['l_max'].value, b_min=constraints_parameters['b_min'].value, b_max=constraints_parameters['b_max'].value, r_min=constraints_parameters['r_min'].value, r_max=constraints_parameters['r_max'].value):

        '''
        Set stars constraints parameters

        Input parameters
        ----------------
        tau_min : int or float --> minimum age of a thin disc star
        tau_max : int or float --> maximum age of a thin disc star
        T_tau_min : int or float --> minimum age of a young thick disc star
        T_tau_max : int or float --> maximum age of a young thick disc star
        mass_min : int or float --> minimum mass to generate a star
        mass_max : int or float --> maximum mass to generate a star
        l_min : int or float --> minimum Galactic longitude
        l_max : int or float --> maximum Galactic longitude
        b_min : int or float --> minimum Galactic latitude
        b_max : int or float --> maximum Galactic latitude
        r_min : int or float --> minimum distance
        r_max : int or float --> maximum distance
        '''

        print('\nSetting constraints parameters...\n')

        self.tau_min = tau_min
        self.tau_max = tau_max
        self.mass_min = mass_min
        self.mass_max = mass_max
        self.l_min = l_min
        self.l_max = l_max
        self.b_min = b_min
        self.b_max = b_max
        self.r_min = r_min
        self.r_max = r_max
        
        if self.ThickParamYoung=='fit':
            self.T_tau_min = T_tau_min
            self.T_tau_max = T_tau_max
        
    
    def set_bgmfast_parameters(self,
                            free_params=bgmfast_parameters['free_params'].value,
                            fixed_params=bgmfast_parameters['fixed_params'].value):
        
        '''
        Set the values of the BGM FASt parameters that are fixed and the positions in the input list for run_simulation function of the BGM FASt parameters that are free
        
        Input parameters
        ----------------
        free_params : dict --> dictionary with the names of the free parameters as keys and the position in the list of free parameters as values
        fixed_params : dict --> dictionary with the names of the fixed parameters and their values
        '''
        
        print('\nSetting free and fixed BGM FASt parameters...\n')
        
        self.all_params = {}
        for param, value in fixed_params.items():
            self.all_params[param] = ['fixed', value] 
        for param, position in free_params.items():
            if param in self.all_params.keys():
                print('A parameter cannot be fixed and free at the same time')
                sys.exit()
            self.all_params[param] = ['free', position]


    def read_catalog(self, filename, sel_columns, Gmax):

        '''
        Read the catalog with real data using Spark. The data is filtered taking only stars with Gerr<Gmax and with positive parallax

        Input parameters
        ----------------
        filename : str --> directory of the catalog file. Example: /home/username/bgmfast/inputs/gaiaDR3_G13.csv
        sel_columns : dict --> name of the columns we want to keep from the file. The dictionary must follow this order: G, color, longitude, latitude, Mvarpi and parallax
        Gmax : int or float --> limitting magnitude

        Output parameters
        -----------------
        catalog : dataframe --> table with the data of the catalog file
        '''

        print('\nReading the catalog file...\n')

        spark = self.spark
        
        catalog = spark.read.option("header","true").csv(filename).select(list(sel_columns.values()))
        self.catalog = catalog.filter((catalog[sel_columns['G']]<float(Gmax)) & (catalog[sel_columns['parallax']]!='') & (catalog[sel_columns['parallax']]>float(0.0)) & (catalog[sel_columns['color']]!=''))
        
        return self.catalog


    def read_ms(self, filename, sel_columns, Gmax, parquet, num_partitions='default'):

        '''
        Read the Mother Simulation file using Spark. The data is filtered taking only stars with Gerr<Gmax and with positive parallax

        Input parameters
        ----------------
        filename : str --> directory of the Mother Simulation file or the parquet file with the reduced Mother Simulation (see parquet parameter). Example: /home/username/bgmfast/inputs/ms_G13_errors_bgmfast.csv
        sel_columns : dict --> name of the columns we want to keep from the file. The dictionary must follow this order: G, color, PopBin, age, mass, longitude, latitude, parallax, Mvarpi
        Gmax : int or float --> limitting magnitude
        parquet : str or bool --> wether we want to "generate" a parquet file with the reduced Mother Simulation for faster import in the future, or "open" the reduced Mother Simulation generated in another run
        num_partitions : str or int --> number of partitions into which we want to divide the Pyspark dataframe. 'default' option does not modify the original number of partitions

        Output parameters
        -----------------
        Mother_Simulation_DF : dataframe --> table with the data of the Mother Simulation file
        '''

        spark = self.spark
        
        if parquet=='generate':
            print('\nCreating reduced Mother Simulation parquet file...')
            
            Xmin = self.Xmin
            Xmax = self.Xmax
            Ymin = self.Ymin
            Ymax = self.Ymax
            Bmin = self.Bmin
            Bmax = self.Bmax
            Lmin = self.Lmin
            Lmax = self.Lmax
            blims = self.blims
            llims = self.llims
            Xstep = self.Xstep
            Ystep = self.Ystep
            mass_step = self.mass_step
            popbin_list = self.popbin_list
            
            tau_min = self.tau_min
            T_tau_min = self.T_tau_min
            tau_max = self.tau_max
            T_tau_max = self.T_tau_max
            mass_min = self.mass_min
            mass_max = self.mass_max
            l_min = self.l_min
            l_max = self.l_max
            b_min = self.b_min
            b_max = self.b_max
            r_min = self.r_min
            r_max = self.r_max
            ThickParamYoung = self.ThickParamYoung
            
            tau_ranges = self.tau_ranges
            T_tau_ranges = self.T_tau_ranges
            
            x1 = self.x1
            x4 = self.x4
            
            Mother_Simulation_DFa = spark.read.option("header","true").csv(filename).select(list(sel_columns.values()))
            
            Mother_Simulation_DF = Mother_Simulation_DFa.filter((Mother_Simulation_DFa[sel_columns['G']]<float(Gmax)) & (Mother_Simulation_DFa[sel_columns['parallax']]>float(0.0)))
            
            del Mother_Simulation_DFa

            Mother_Simulation_DF.cache() # Checking that the file is correct
            
            if ThickParamYoung=='fit':
                self.reduced_Mother_Simulation = generate_reduced_MS(Mother_Simulation_DF, x1, x4, mass_step, popbin_list, sel_columns, [tau_ranges, T_tau_ranges], Xmin, Xmax, Ymin, Ymax, Bmin, Bmax, Lmin, Lmax, blims, llims, Xstep, Ystep, [tau_min, T_tau_min], [tau_max, T_tau_max], mass_min, mass_max, l_min, l_max, b_min, b_max, r_min, r_max, ThickParamYoung)
            else:
                self.reduced_Mother_Simulation = generate_reduced_MS(Mother_Simulation_DF, x1, x4, mass_step, popbin_list, sel_columns, tau_ranges, Xmin, Xmax, Ymin, Ymax, Bmin, Bmax, Lmin, Lmax, blims, llims, Xstep, Ystep, tau_min, tau_max, mass_min, mass_max, l_min, l_max, b_min, b_max, r_min, r_max, ThickParamYoung)
                
            self.reduced_Mother_Simulation = self.spark.createDataFrame(self.reduced_Mother_Simulation)
            self.reduced_Mother_Simulation.write.parquet(filename.split('.csv')[0] + '_reduced.parquet', mode='overwrite')
        
        elif parquet=='open':
            print('\nOpening reduced Mother Simulation parquet file...')
            self.reduced_Mother_Simulation = spark.read.parquet(filename)
        
        print('\nDefault number of partitions:', self.reduced_Mother_Simulation.rdd.getNumPartitions())
        if num_partitions!='default':
            self.reduced_Mother_Simulation = self.reduced_Mother_Simulation.repartition(num_partitions)
            print('\nNew number of partitions:', self.reduced_Mother_Simulation.rdd.getNumPartitions())
        
        return self.reduced_Mother_Simulation


    def accumulators_init(self):

        '''
        Initialize one 4-dimensional and one 2-dimensional Pyspark accumulators with nLonbins times nLatbins times nColorbins times nGbins bins, and nMassbins times nTaubins bins, respectively

        Output parameters
        -----------------
        acc : pyspark accumulator --> 4-dimensional Pyspark accumulator (Hess diagram + latitude + longitude) containing the Hess diagram
        simple : pyspark accumulator --> Pyspark simple accumulator that counts the stars that are not within the considered ranges or that have suffered some problem during the computations
        smallacc : pyspark accumulator --> 2-dimensional Pyspark accumulator (Mass + Popbin)
        '''

        sc = self.sc
        MatrixAccumulatorParam = self.MatrixAccumulatorParam
        SmallMatrixAccumulatorParam = self.SmallMatrixAccumulatorParam
        
        self.nMassbins = int((self.x4 - self.x1)/self.mass_step) + 1
        if self.ThickParamYoung=='fit':
            self.nTaubins = len([tau_range for popbin_range in self.tau_ranges for tau_range in popbin_range]) + len(self.T_tau_ranges)
        else:
            self.nTaubins = len([tau_range for popbin_range in self.tau_ranges for tau_range in popbin_range])
        
        MATRIXCMD = np.zeros((self.nLonbins, self.nLatbins, self.nColorbins, self.nGbins))
        MATRIXsmall = np.zeros((self.nTaubins, self.nMassbins))

        self.acc = sc.accumulator(MATRIXCMD, MatrixAccumulatorParam())
        self.simple = sc.accumulator(0)
        self.smallacc = sc.accumulator(MATRIXsmall, SmallMatrixAccumulatorParam())

        return self.acc, self.simple, self.smallacc


    def generate_catalog_cmd(self):

        '''
        Generate catalog Hess diagram

        Output parameters
        -----------------
        catalog_data : numpy array --> 4-dimensional numpy array with the Hess diagrams corresponding to each one of the longitude and latitude ranges
        '''

        Xmin = self.Xmin
        Xmax = self.Xmax
        Ymin = self.Ymin
        Ymax = self.Ymax
        Bmin = self.Bmin
        Bmax = self.Bmax
        Lmin = self.Lmin
        Lmax = self.Lmax
        blims = self.blims
        llims = self.llims
        Xstep = self.Xstep
        Ystep = self.Ystep

        acc, simple = self.accumulators_init()[:2]

        self.catalog.foreach(lambda x: pes_catalog(x, Xmin, Xmax, Ymin, Ymax, Bmin, Bmax, Lmin, Lmax, blims, llims, Xstep, Ystep, acc, simple))

        self.acc = acc
        self.simple = simple

        self.catalog_data = self.return_cmd()[1]

        return self.catalog_data


    def return_cmd(self):

        '''
        Return the accumulators and obtain the variable with the two accumulators appended

        Output parameters
        -----------------
        MATRIXCMD : numpy array --> 4-dimensional numpy array (Hess diagram + latitude + longitude) containing the complete Hess diagram
        data : numpy array --> 4-dimensional numpy array with the Hess diagrams corresponding to each one of the longitude and latitude ranges
        '''

        MATRIXCMD = self.acc.value
        data = np.reshape(self.acc.value, np.size(self.acc.value))

        return MATRIXCMD, data


    def run_simulation(self, param):

        '''
        Run one BGM FASt simulation

        Input parameters
        ----------------
        param : list --> list with the free BGM FASt parameters (currently SFH + IMF)

        Output parameters
        -----------------
        simulation_data : numpy array --> 4-dimensional numpy array with the Hess diagrams corresponding to each one of the longitude and latitude ranges
        '''

        sc = self.sc
        MatrixAccumulatorParam = self.MatrixAccumulatorParam

        x2_ps = self.x2_ps
        x3_ps = self.x3_ps

        x1 = self.x1
        x4 = self.x4
        tau_min_edges = self.tau_min_edges
        tau_max_edges = self.tau_max_edges
        ThickParamYoung = self.ThickParamYoung
        HaloParam = self.HaloParam
        BarParam = self.BarParam
        ThickParamOld = self.ThickParamOld
        mass_min = self.mass_min
        mass_step = self.mass_step
        tau_ranges = self.tau_ranges
        
        if ThickParamYoung=='fit':
            T_tau_min_edges = self.T_tau_min_edges
            T_tau_max_edges = self.T_tau_max_edges
            T_tau_ranges = self.T_tau_ranges

        x2_ms = self.x2_ms
        x3_ms = self.x3_ms
        alpha1_ms = self.alpha1_ms
        alpha2_ms = self.alpha2_ms
        alpha3_ms = self.alpha3_ms
        SigmaParam_ms = self.SigmaParam_ms
        midpopbin_ms = self.midpopbin_ms
        lastpopbin_ms = self.lastpopbin_ms
        K1_ms = self.K1_ms
        K2_ms = self.K2_ms
        K3_ms = self.K3_ms
        bin_nor_ms = self.bin_nor_ms
        
        if ThickParamYoung=='fit':
            T_SigmaParam_ms = self.T_SigmaParam_ms
            T_bin_nor_ms = self.T_bin_nor_ms

        acc, simple, smallacc = self.accumulators_init()

        # Explored parameters
        for key, value in self.all_params.items():
            if key=='alpha1':
                if value[0]=='free':
                    alpha1_ps = param[value[1]]
                elif value[0]=='fixed':
                    alpha1_ps = value[1]
            elif key=='alpha2':
                if value[0]=='free':
                    alpha2_ps = param[value[1]]
                elif value[0]=='fixed':
                    alpha2_ps = value[1]
            elif key=='alpha3':
                if value[0]=='free':
                    alpha3_ps = param[value[1]]
                elif value[0]=='fixed':
                    alpha3_ps = value[1]
            elif key=='SFH0':
                if value[0]=='free':
                    SFH0_ps = param[value[1]]
                elif value[0]=='fixed':
                    SFH0_ps = value[1]
            elif key=='SFH1':
                if value[0]=='free':
                    SFH1_ps = param[value[1]]
                elif value[0]=='fixed':
                    SFH1_ps = value[1]
            elif key=='SFH2':
                if value[0]=='free':
                    SFH2_ps = param[value[1]]
                elif value[0]=='fixed':
                    SFH2_ps = value[1]
            elif key=='SFH3':
                if value[0]=='free':
                    SFH3_ps = param[value[1]]
                elif value[0]=='fixed':
                    SFH3_ps = value[1]
            elif key=='SFH4':
                if value[0]=='free':
                    SFH4_ps = param[value[1]]
                elif value[0]=='fixed':
                    SFH4_ps = value[1]
            elif key=='SFH5':
                if value[0]=='free':
                    SFH5_ps = param[value[1]]
                elif value[0]=='fixed':
                    SFH5_ps = value[1]
            elif key=='SFH6':
                if value[0]=='free':
                    SFH6_ps = param[value[1]]
                elif value[0]=='fixed':
                    SFH6_ps = value[1]
            elif key=='SFH7':
                if value[0]=='free':
                    SFH7_ps = param[value[1]]
                elif value[0]=='fixed':
                    SFH7_ps = value[1]
            elif key=='SFH8':
                if value[0]=='free':
                    SFH8_ps = param[value[1]]
                elif value[0]=='fixed':
                    SFH8_ps = value[1]
            elif key=='SFH9':
                if value[0]=='free':
                    SFH9_ps = param[value[1]]
                elif value[0]=='fixed':
                    SFH9_ps = value[1]
            elif key=='SFH10':
                if value[0]=='free':
                    SFH10_ps = param[value[1]]
                elif value[0]=='fixed':
                    SFH10_ps = value[1]
            elif key=='SFH9T':
                if value[0]=='free':
                    SFH9T_ps = param[value[1]]
                elif value[0]=='fixed':
                    SFH9T_ps = value[1]
            elif key=='SFH10T':
                if value[0]=='free':
                    SFH10T_ps = param[value[1]]
                elif value[0]=='fixed':
                    SFH10T_ps = value[1]
            elif key=='SFH11T':
                if value[0]=='free':
                    SFH11T_ps = param[value[1]]
                elif value[0]=='fixed':
                    SFH11T_ps = value[1]
            elif key=='SFH12T':
                if value[0]=='free':
                    SFH12T_ps = param[value[1]]
                elif value[0]=='fixed':
                    SFH12T_ps = value[1]
        
        SigmaParam_ps = np.array([SFH0_ps, SFH1_ps, SFH2_ps, SFH3_ps, SFH4_ps + SFH5_ps, SFH6_ps + SFH7_ps, SFH8_ps + SFH9_ps + SFH10_ps])
        midpopbin_ps = np.array([SFH4_ps, SFH5_ps, SFH6_ps, SFH7_ps])
        lastpopbin_ps = np.array([SFH8_ps, SFH9_ps, SFH10_ps])
        
        if ThickParamYoung=='fit':
            T_SigmaParam_ps = np.array([SFH9T_ps, SFH10T_ps, SFH11T_ps, SFH12T_ps])

        # If some surface mass density is negative, then Lr='inf' and we redraw again
        if ThickParamYoung=='fit':
            if ThickParamOld>0 and SigmaParam_ps[SigmaParam_ps<0].size==0 and lastpopbin_ps[lastpopbin_ps<0].size==0 and midpopbin_ps[midpopbin_ps<0].size==0 and T_SigmaParam_ps[T_SigmaParam_ps<0].size==0:
                
                K1_ps,K2_ps,K3_ps = Continuity_Coeficients_func(alpha1_ps, alpha2_ps, alpha3_ps, x1, x2_ps, x3_ps, x4)
                bin_nor_ps = bin_nor_func(x1, x2_ps, x3_ps, x4, K1_ps, K2_ps, K3_ps, alpha1_ps, alpha2_ps, alpha3_ps, SigmaParam_ps, tau_min_edges, tau_max_edges)
                T_bin_nor_ps = bin_nor_func(x1, x2_ps, x3_ps, x4, K1_ps, K2_ps, K3_ps, alpha1_ps, alpha2_ps, alpha3_ps, T_SigmaParam_ps, T_tau_min_edges, T_tau_max_edges)
                
                start = time.time()
                current_datetime = datetime.now()
                formatted_datetime = str(current_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-4])
                
                self.reduced_Mother_Simulation.foreach(lambda x: wpes_func(x, x1, x2_ps, x3_ps, K1_ps, K2_ps, K3_ps, alpha1_ps, alpha2_ps, alpha3_ps, [SigmaParam_ps, T_SigmaParam_ps], midpopbin_ps, lastpopbin_ps, [bin_nor_ps, T_bin_nor_ps], x2_ms, x3_ms, K1_ms, K2_ms, K3_ms, alpha1_ms, alpha2_ms, alpha3_ms, [SigmaParam_ms, T_SigmaParam_ms], midpopbin_ms, lastpopbin_ms, [bin_nor_ms, T_bin_nor_ms], ThickParamYoung, HaloParam, BarParam, ThickParamOld, mass_min, mass_step, [tau_ranges, T_tau_ranges], acc, smallacc))

                end = time.time()
                self.num_sim += 1

                if self.logfile!=False:
                    with open(self.logfile, 'a') as logs:
                        logs.write(str(self.num_sim) + ',' + formatted_datetime + ',' + str(round(end - start, 2)) + '\n')

                self.acc = acc
                self.simple = simple
                self.smallacc = smallacc

                self.simulation_data = self.return_cmd()[1]
                
            else:
                self.simulation_data = np.array([0])
            
        elif ThickParamYoung>0 and ThickParamOld>0 and SigmaParam_ps[SigmaParam_ps<0].size==0 and lastpopbin_ps[lastpopbin_ps<0].size==0 and midpopbin_ps[midpopbin_ps<0].size==0:

            K1_ps,K2_ps,K3_ps = Continuity_Coeficients_func(alpha1_ps, alpha2_ps, alpha3_ps, x1, x2_ps, x3_ps, x4)
            bin_nor_ps = bin_nor_func(x1, x2_ps, x3_ps, x4, K1_ps, K2_ps, K3_ps, alpha1_ps, alpha2_ps, alpha3_ps, SigmaParam_ps, tau_min_edges, tau_max_edges)

            start = time.time()
            current_datetime = datetime.now()
            formatted_datetime = str(current_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-4])

            self.reduced_Mother_Simulation.foreach(lambda x: wpes_func(x, x1, x2_ps, x3_ps, K1_ps, K2_ps, K3_ps, alpha1_ps, alpha2_ps, alpha3_ps, SigmaParam_ps, midpopbin_ps, lastpopbin_ps, bin_nor_ps, x2_ms, x3_ms, K1_ms, K2_ms, K3_ms, alpha1_ms, alpha2_ms, alpha3_ms, SigmaParam_ms, midpopbin_ms, lastpopbin_ms, bin_nor_ms, ThickParamYoung, HaloParam, BarParam, ThickParamOld, mass_min, mass_step, tau_ranges, acc, smallacc))

            end = time.time()
            self.num_sim += 1

            if self.logfile!=False:
                with open(self.logfile, 'a') as logs:
                    logs.write(str(self.num_sim) + ',' + formatted_datetime + ',' + str(round(end - start, 2)) + '\n')

            self.acc = acc
            self.simple = simple
            self.smallacc = smallacc

            self.simulation_data = self.return_cmd()[1]

        else:
            self.simulation_data = np.array([0])

        return self.simulation_data





