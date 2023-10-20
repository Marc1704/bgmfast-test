'''
BGM FASt simulation class

This script is intented to include all the functions working within the Pyspark environment.
'''


import numpy as np
from pyspark.sql import SparkSession
from pyspark.accumulators import AccumulatorParam

from bgmfast.auxiliary_functions import *
from bgmfast import parameters


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
        sindex : list --> list with one or five elements containing the coordinates in the 4-dimensional (latitude, longitude, Bp-Rp and M_G') space of the Pyspark accumulator into which we want to put the element. The last element defines the value of the weight we want to add in that coordinates

        Output parameters
        -----------------
        mAdd : Pyspark accumulator? --> updated mAdd
        '''

        if type(sindex)==list:
            mAdd[sindex[0], sindex[1], sindex[2], sindex[3] ] += sindex[4]
        else:
            mAdd += sindex

        return mAdd


# ****************
# WEIGHT FUNCTIONS
# ****************

def pes_catalog(x, Xmin, Xmax, Ymin, Ymax, Bmin, Bmax, Lmin, Lmax, blims, llims, Ylims, Ylims_Xsteps, Ylims_Ysteps, acc_complete, acc, acc2, simple):

    '''
    Build the Hess diagram of the observed catalog using a Pyspark accumulator

    Input parameters
    ----------------
    x : list --> each one of the rows in the Mother Simulation file
    Xmin : int or float --> minimum value for the binning in Bp-Rp range
    Xmax : int or float --> maximum value for the binning in Bp-Rp range
    Ymin : int or float --> minimum value for the binning in M_G' range
    Ymax : int or float --> maximum value for the binning in M_G' range
    Bmin : int or float --> minimum value for the binning in latitude
    Bmax : int or float --> maximum value for the binning in latitude
    Lmin : int or float --> minimum value for the binning in longitude
    Lmax : int or float --> maximum value for the binning in longitude
    blims : list --> limits of the different absolute latitude ranges
    llims : list --> limits of the different longitude ranges
    Ylims : list --> limits of the different Bp-Rp ranges
    Ylims_Xsteps : list --> Bp-Rp steps of the different Bp-Rp ranges
    Ylims_Ysteps : list --> M_G' steps of the different Bp-Rp ranges
    acc_complete : pyspark accumulator --> 4-dimensional Pyspark accumulator (Hess diagram + latitude + longitude) containing the complete Hess diagram
    acc : pyspark accumulator --> 4-dimensional Pyspark accumulator (Hess diagram + latitude + longitude) for stars in the first considered range of M_G'
    acc2 : pyspark accumulator --> 4-dimensional Pyspark accumulator (Hess diagram + latitude + longitude) for stars in the second considered range of M_G'
    simple : pyspark accumulator --> Pyspark simple accumulator that counts the stars that are not within the considered ranges or that have suffered some problem during the computations

    Output parameters
    -----------------
    cpes : int --> weight of the star. In this case, since we are dealing with the observed data, the value of the weight is always 1
    '''

    BpRp = float(x[1])
    longitude = float(x[2])
    latitude = float(x[3])
    Mvarpi = float(x[4])

    xinput = [BpRp, longitude, latitude, Mvarpi]
    matindex = binning_4D_Mvarpi(xinput, Xmin, Xmax, Ymin, Ymax, Bmin, Bmax, Lmin, Lmax, blims, llims, Ylims, Ylims_Xsteps, Ylims_Ysteps)

    cpes=1

    if (np.isnan(matindex[0]) or np.isnan(matindex[1]) or np.isnan(matindex[2]) or np.isnan(matindex[3])):
        simple.add(1)
    elif(Ylims[0][0]<=Mvarpi<Ylims[0][1]):
        acc.add([int(matindex[0]),int(matindex[1]),int(matindex[2]),int(matindex[3]),cpes])
    elif(Ylims[1][0]<=Mvarpi<Ylims[1][1]):
        acc2.add([int(matindex[0]),int(matindex[1]),int(matindex[2]),int(matindex[3]),cpes])
    else:
        simple.add(1)

    if (np.isnan(matindex[0]) or np.isnan(matindex[1]) or np.isnan(matindex[4]) or np.isnan(matindex[5])):
        pass
    else:
        acc_complete.add([int(matindex[0]),int(matindex[1]),int(matindex[4]),int(matindex[5]),cpes])

    return cpes


def wpes_func(WP, Xmin, Xmax, Ymin, Ymax, Bmin, Bmax, Lmin, Lmax, blims, llims, Ylims, Ylims_Xsteps, Ylims_Ysteps, tau_min, tau_max, mass_min, mass_max, l_min, l_max, b_min, b_max, r_min, r_max, x1, x2_ps, x3_ps, K1_ps, K2_ps, K3_ps, alpha1_ps, alpha2_ps, alpha3_ps, SigmaParam_ps, midpopbin_ps, lastpopbin_ps, bin_nor_ps, x2_ms, x3_ms, K1_ms, K2_ms, K3_ms, alpha1_ms, alpha2_ms, alpha3_ms, SigmaParam_ms, midpopbin_ms, lastpopbin_ms, bin_nor_ms, ThickParamYoung, ThickParamOld, HaloParam, acc_complete, acc, acc2, simple):

    '''
    Compute the weight of a given star. It uses Equation (37) from Mor et al. 2018 without integrating. The integral is conceptual, because it defines the integration over an increment (bin) of the N-dimensional space defined in Eq. (6). We reduce this increment until the end, when we only consider the star itself. At that point, the increment is exactly equal to the differential (both of them are the star itself) and the integral blows up.

    Input parameters
    ----------------
    WP : list -->
    Xmin : int or float --> minimum value for the binning in Bp-Rp range
    Xmax : int or float --> maximum value for the binning in Bp-Rp range
    Ymin : int or float --> minimum value for the binning in M_G' range
    Ymax : int or float --> maximum value for the binning in M_G' range
    Bmin : int or float --> minimum value for the binning in latitude
    Bmax : int or float --> maximum value for the binning in latitude
    Lmin : int or float --> minimum value for the binning in longitude
    Lmax : int or float --> maximum value for the binning in longitude
    blims : list --> limits of the different absolute latitude ranges
    llims : list --> limits of the different longitude ranges
    Ylims : list --> limits of the different Bp-Rp ranges
    Ylims_Xsteps : list --> Bp-Rp steps of the different Bp-Rp ranges
    Ylims_Ysteps : list --> M_G' steps of the different Bp-Rp ranges
    tau_min : int or float --> minimum age of a thin disc star
    tau_max : int or float --> maximum age of a thin disc star
    mass_min : int or float --> minimum mass to generate a star
    mass_max : int or float --> maximum mass to generate a star
    l_min : int or float --> minimum Galactic longitude
    l_max : int or float --> maximum Galactic longitude
    b_min : int or float --> minimum Galactic latitude
    b_max : int or float --> maximum Galactic latitude
    r_min : int or float --> minimum distance
    r_max : int or float --> maximum distance
    x1 : int or float --> minimum mass to generate a star
    x2_ps : int or float --> first mass limit of the IMF for the BGM FASt simulation
    x3_ps : int or float --> second mass limit of the IMF for the BGM FASt simulation
    K1_ps : int or float --> first continuity coeficient of the IMF for the BGM FASt simulation
    K2_ps : int or float --> second continuity coeficient of the IMF for the BGM FASt simulation
    K3_ps : int or float --> third continuity coeficient of the IMF for the BGM FASt simulation
    alpha1_ps : int or float --> first slope (alpha) of the IMF for the BGM FASt simulation
    alpha2_ps : int or float --> second slope (alpha) of the IMF for the BGM FASt simulation
    alpha3_ps : int or float --> third slope (alpha) of the IMF for the BGM FASt simulation
    SigmaParam_ps : list --> surface density at the position of the Sun for the different age subpopulations of the thin disc for the BGM FASt simulation
    midpopbin_ps : list --> surface density at the position of the Sun for the four subdivisions of the 5th and 6th age subpopulations of the thin disc (3-5 Gyr and 5-7 Gyr) for the BGM FASt simulation
    lastpopbin_ps : list --> surface density at the position of the Sun for the three subdivisions of the last (7th) age subpopulation of the thin disc (7-10 Gyr) for the BGM FASt simulation
    bin_nor_ps : int or float --> normalization coeficient for binaries for the BGM FASt simulation
    x2_ms : int or float --> first mass limit of the IMF for the Mother Simulation
    x3_ms : int or float --> second mass limit of the IMF for the Mother Simulation
    K1_ms : int or float --> first continuity coeficient of the IMF for the Mother Simulation
    K2_ms : int or float --> second continuity coeficient of the IMF for the Mother Simulation
    K3_ms : int or float --> third continuity coeficient of the IMF for the Mother Simulation
    alpha1_ms : int or float --> first slope (alpha) of the IMF for the Mother Simulation
    alpha2_ms : int or float --> second slope (alpha) of the IMF for the Mother Simulation
    alpha3_ms : int or float --> third slope (alpha) of the IMF for the Mother Simulation
    SigmaParam_ms : list --> surface density at the position of the Sun for the different age subpopulations of the thin disc for the Mother Simulation
    midpopbin_ms : list --> surface density at the position of the Sun for the four subdivisions of the 5th and 6th age subpopulations of the thin disc (3-5 Gyr and 5-7 Gyr) for the Mother Simulation
    lastpopbin_ms : list --> surface density at the position of the Sun for the three subdivisions of the last (7th) age subpopulation of the thin disc (7-10 Gyr) for the Mother Simulation
    bin_nor_ms : int or float --> normalization coeficient for binaries for the Mother Simulation
    ThickParamYoung : int or float --> weight of the stars in the Young Thick disc
    ThickParamOld : int or float --> weight of the stars in the Old Thick disc
    HaloParam : int or float --> weight of the stars in the Halo
    acc_complete : pyspark accumulator --> 4-dimensional Pyspark accumulator (Hess diagram + latitude + longitude) containing the complete Hess diagram
    acc : pyspark accumulator --> 4-dimensional Pyspark accumulator (Hess diagram + latitude + longitude) for stars in the first considered range of M_G'
    acc2 : pyspark accumulator --> 4-dimensional Pyspark accumulator (Hess diagram + latitude + longitude) for stars in the second considered range of M_G'
    simple : pyspark accumulator --> Pyspark simple accumulator that counts the stars that are not within the considered ranges or that have suffered some problem during the computations

    Output parameters
    -----------------
    wpes : int --> weight of the star derived from BGM FASt
    '''

    wpes = 1
    BpRperr = float(WP[1])
    popbin = float(WP[2]) # BGM population bin
    tau = float(WP[3]) # Age of the star
    mass = float(WP[4]) # Mass of the star
    lstar = float(WP[5]) # Longitude of the star
    bstar = float(WP[6]) # Latitude of the star
    parallax = float(WP[7])
    rstar = 1/parallax*1000. # Distance of the star (pc)
    Mvarpi = float(WP[8])

    if 3<tau<=4:
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

    Sinput = [BpRperr, lstar, bstar, Mvarpi]
    matindex = binning_4D_Mvarpi(Sinput, Xmin, Xmax, Ymin, Ymax, Bmin, Bmax, Lmin, Lmax, blims, llims, Ylims, Ylims_Xsteps, Ylims_Ysteps)

    if popbin==8:
        wpes = ThickParamYoung

    elif popbin==9:
        wpes = HaloParam

    elif popbin==11:
        wpes = ThickParamOld

    elif (tau_min<=tau<=tau_max and mass_min<=mass<=mass_max and l_min<=lstar<=l_max and b_min<=bstar<=b_max and r_min<=rstar<=r_max):
        itau = int(popbin)-1

        PS = float(Simplified_Gi_Primal_func_NONP(itau, mass, x1, x2_ps, x3_ps, K1_ps, K2_ps, K3_ps, alpha1_ps, alpha2_ps, alpha3_ps, SigmaParam_ps, bin_nor_ps, midpopbin_ps, lastpopbin_ps, imidpoptau, ilastpoptau))

        MS = float(Simplified_Gi_Primal_func_NONP(itau, mass, x1, x2_ms, x3_ms, K1_ms, K2_ms, K3_ms, alpha1_ms, alpha2_ms, alpha3_ms, SigmaParam_ms, bin_nor_ms, midpopbin_ms, lastpopbin_ms, imidpoptau, ilastpoptau))

        if PS==0:
            wpes = 0
        elif MS==0:
            wpes = 1
        else:
            wpes = PS/MS

    else:
        print(popbin, tau)
        wpes = 1

    if (np.isnan(matindex[0]) or np.isnan(matindex[1]) or np.isnan(matindex[2]) or np.isnan(matindex[3])):
        simple.add(1)
    else:
        if Ylims[0][0]<Mvarpi<Ylims[0][1]:
            acc.add([int(matindex[0]),int(matindex[1]),int(matindex[2]),int(matindex[3]),wpes])
        elif Ylims[1][0]<Mvarpi<Ylims[1][1]:
            acc2.add([int(matindex[0]),int(matindex[1]),int(matindex[2]),int(matindex[3]),wpes])
        else:
            simple.add(1)

    if (np.isnan(matindex[0]) or np.isnan(matindex[1]) or np.isnan(matindex[4]) or np.isnan(matindex[5])):
        pass
    else:
        acc_complete.add([int(matindex[0]),int(matindex[1]),int(matindex[4]),int(matindex[5]),wpes])

    return wpes


# *********************
# SIMULATION CLASS (C1)
# *********************

class bgmfast_simulation:
    '''
    Run the BGM FASt simulations
    '''

    def __init__(self):
        '''
        Initialize the bgmfast_simulation class
        '''

        print('=======================================================================')
        print('\n******************* Welcome to BGM FASt version 4.3 *******************\n')
        print('=======================================================================')

        pass


    def open_spark_session(self):

        '''
        Open the Spark Session

        Output parameters
        -----------------
        sc : spark.sparkContext --> Pyspark parameter needed for internal funcionality
        spark : pyspark.sql.session.SparkSession --> Pyspark parameter needed for internal funcionality
        '''
        print('\nOpening Spark Session...\n')

        self.spark = SparkSession.builder.appName("Strangis").getOrCreate()
        print(self.spark)
        self.sc = self.spark.sparkContext

        return self.sc, self.spark


    def set_acc_parameters(self):

        '''
        Set accumulators parameters
        '''

        print('\nSetting accumulators parameters...\n')

        acc_parameters = parameters.acc_parameters

        self.nLonbins = acc_parameters['nLonbins'].value
        self.nLatbins = acc_parameters['nLatbins'].value
        self.nColorbins = acc_parameters['nColorbins'].value
        self.nGbins = acc_parameters['nGbins'].value
        self.nLonbins1 = acc_parameters['nLonbins1'].value
        self.nLatbins1 = acc_parameters['nLatbins1'].value
        self.nColorbins1 = acc_parameters['nColorbins1'].value
        self.nGbins1 = acc_parameters['nGbins1'].value
        self.nLonbins2 = acc_parameters['nLonbins2'].value
        self.nLatbins2 = acc_parameters['nLatbins2'].value
        self.nColorbins2 = acc_parameters['nColorbins2'].value
        self.nGbins2 = acc_parameters['nGbins2'].value
        self.MatrixAccumulatorParam = MatrixAccumulatorParam


    def set_binning_parameters(self):

        '''
        Set binning parameters
        '''

        print('\nSetting binning parameters...\n')

        binning_parameters = parameters.binning_parameters

        self.Xmin = binning_parameters['Xmin'].value
        self.Xmax = binning_parameters['Xmax'].value
        self.Ymin = binning_parameters['Ymin'].value
        self.Ymax = binning_parameters['Ymax'].value
        self.Bmin = binning_parameters['Bmin'].value
        self.Bmax = binning_parameters['Bmax'].value
        self.Lmin = binning_parameters['Lmin'].value
        self.Lmax = binning_parameters['Lmax'].value
        self.blims = binning_parameters['blims'].value
        self.llims = binning_parameters['llims'].value
        self.Ylims = binning_parameters['Ylims'].value
        self.Ylims_Xsteps = binning_parameters['Ylims_Xsteps'].value
        self.Ylims_Ysteps = binning_parameters['Ylims_Ysteps'].value


    def set_general_parameters(self):

        '''
        Set general parameters
        '''

        print('\nSetting general parameters...\n')

        general_parameters = parameters.general_parameters

        self.x1 = general_parameters['x1'].value
        self.x4 = general_parameters['x4'].value
        self.tau_min_edges = general_parameters['tau_min_edges'].value
        self.tau_max_edges = general_parameters['tau_max_edges'].value
        self.ThickParamYoung = general_parameters['ThickParamYoung'].value
        self.ThickParamOld = general_parameters['ThickParamOld'].value
        self.HaloParam = general_parameters['HaloParam'].value


    def set_ms_parameters(self):

        '''
        Set Mother Simulation parameters
        '''

        print('\nSetting Mother Simulation parameters...\n')

        ms_parameters = parameters.ms_parameters

        self.x2_ms = ms_parameters['x2_ms'].value
        self.x3_ms = ms_parameters['x3_ms'].value
        self.alpha1_ms = ms_parameters['alpha1_ms'].value
        self.alpha2_ms = ms_parameters['alpha2_ms'].value
        self.alpha3_ms = ms_parameters['alpha3_ms'].value
        self.SigmaParam_ms = ms_parameters['SigmaParam_ms'].value
        self.midpopbin_ms = ms_parameters['midpopbin_ms'].value
        self.lastpopbin_ms = ms_parameters['lastpopbin_ms'].value

        self.K1_ms, self.K2_ms, self.K3_ms = Continuity_Coeficients_func(self.alpha1_ms, self.alpha2_ms, self.alpha3_ms, self.x1, self.x2_ms, self.x3_ms, self.x4)

        self.bin_nor_ms = bin_nor_func(self.x1, self.x2_ms, self.x3_ms, self.x4, self.K1_ms, self.K2_ms, self.K3_ms, self.alpha1_ms, self.alpha2_ms, self.alpha3_ms, self.SigmaParam_ms, self.tau_min_edges, self.tau_max_edges)


    def set_ps_parameters(self):

        '''
        Set Pseudo Simulation (BGM FASt) parameters
        '''

        print('\nSetting BGMFASt simulation (pseudo-simulation) parameters...\n')

        ps_parameters = parameters.ps_parameters

        self.x2_ps = ps_parameters['x2_ps'].value
        self.x3_ps = ps_parameters['x3_ps'].value


    def set_constraints_parameters(self):

        '''
        Set stars constraints parameters
        '''

        print('\nSetting constraints parameters...\n')

        constraints_parameters = parameters.constraints_parameters

        self.tau_min = constraints_parameters['tau_min'].value
        self.tau_max = constraints_parameters['tau_max'].value
        self.mass_min = constraints_parameters['mass_min'].value
        self.mass_max = constraints_parameters['mass_max'].value
        self.l_min = constraints_parameters['l_min'].value
        self.l_max = constraints_parameters['l_max'].value
        self.b_min = constraints_parameters['b_min'].value
        self.b_max = constraints_parameters['b_max'].value
        self.r_min = constraints_parameters['r_min'].value
        self.r_max = constraints_parameters['r_max'].value


    def read_catalog(self, filename, sel_columns, Gmax):

        '''
        Read the catalog with real data using Spark. The data is filtered taking only stars with Gerr<Gmax and with positive parallax

        Input parameters
        ----------------
        filename : str --> directory of the catalog file. Example: /home/username/bgmfast/inputs/gaiaDR3_G13.csv
        sel_columns : str or list --> name of the columns we want to keep from the file
        Gmax : int or float --> limitting magnitude

        Output parameters
        -----------------
        catalog : dataframe --> table with the data of the catalog file
        '''

        print('\nReading the catalog file...\n')

        spark = self.spark

        catalog = spark.read.option("header","true").csv(filename).select(sel_columns)
        self.catalog = catalog.filter((catalog.G<Gmax) & (catalog.parallax!='') & (catalog.parallax>0) & (catalog.BpRp!=''))

        return self.catalog


    def read_ms(self, filename, sel_columns, Gmax):

        '''
        Read the Mother Simulation file using Spark. The data is filtered taking only stars with Gerr<Gmax and with positive parallax

        Input parameters
        ----------------
        filename : str --> directory of the Mother Simulation file. Example: /home/username/bgmfast/inputs/ms_G13_errors_bgmfast.csv
        sel_columns : str or list --> name of the columns we want to keep from the file
        Gmax : int or float --> limitting magnitude

        Output parameters
        -----------------
        Mother_Simulation_DF : dataframe --> table with the data of the Mother Simulation file
        '''

        print('\nReading the Mother Simulation file...\n')

        spark = self.spark

        Mother_Simulation_DFa = spark.read.option("header","true").csv(filename).select(sel_columns)
        self.Mother_Simulation_DF = Mother_Simulation_DFa.filter((Mother_Simulation_DFa.Gerr<Gmax) & (Mother_Simulation_DFa.parallaxerr>0))

        self.Mother_Simulation_DF.cache() # Checking that the file is correct

        return self.Mother_Simulation_DF


    def accumulators_init(self):

        '''
        Initialize three 4-dimensional Pyspark accumulators with nLonbins times nLatbins times nColorbins times nGbins bins

        Output parameters
        -----------------
        acc_complete : pyspark accumulator --> 4-dimensional Pyspark accumulator (Hess diagram + latitude + longitude) containing the complete Hess diagram
        acc : pyspark accumulator --> 4-dimensional Pyspark accumulator (Hess diagram + latitude + longitude) for stars in the first considered range of M_G'
        acc2 : pyspark accumulator --> 4-dimensional Pyspark accumulator (Hess diagram + latitude + longitude) for stars in the second considered range of M_G'
        simple : pyspark accumulator --> Pyspark simple accumulator that counts the stars that are not within the considered ranges or that have suffered some problem during the computations
        '''

        sc = self.sc
        MatrixAccumulatorParam = self.MatrixAccumulatorParam

        MATRIXCMD_complete = np.zeros((self.nLonbins, self.nLatbins, self.nColorbins, self.nGbins))
        MATRIXCMD = np.zeros((self.nLonbins1, self.nLatbins1, self.nColorbins1, self.nGbins1))
        MATRIXCMD2 = np.zeros((self.nLonbins2, self.nLatbins2, self.nColorbins2, self.nGbins2))

        self.acc_complete = sc.accumulator(MATRIXCMD_complete, MatrixAccumulatorParam())
        self.acc = sc.accumulator(MATRIXCMD, MatrixAccumulatorParam())
        self.acc2 = sc.accumulator(MATRIXCMD2, MatrixAccumulatorParam())
        self.simple = sc.accumulator(0)

        return self.acc_complete, self.acc, self.acc2, self.simple


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
        Ylims = self.Ylims
        Ylims_Xsteps = self.Ylims_Xsteps
        Ylims_Ysteps = self.Ylims_Ysteps

        acc_complete, acc, acc2, simple = self.accumulators_init()

        self.catalog.foreach(lambda x: pes_catalog(x, Xmin, Xmax, Ymin, Ymax, Bmin, Bmax, Lmin, Lmax, blims, llims, Ylims, Ylims_Xsteps, Ylims_Ysteps, acc_complete, acc, acc2, simple))

        self.acc_complete = acc_complete
        self.acc = acc
        self.acc2 = acc2
        self.simple = simple

        self.catalog_data = self.return_cmd()[3]

        return self.catalog_data


    def return_cmd(self):

        '''
        Return the accumulators and obtain the variable with the two accumulators appended

        Output parameters
        -----------------
        MATRIXCMD_complete : numpy array --> 4-dimensional numpy array (Hess diagram + latitude + longitude) containing the complete Hess diagram
        MATRIXCMD : numpy array --> 4-dimensional Pyspark accumulator (Hess diagram + latitude + longitude) containing the Hess diagrams of the first M_G' range
        MATRIXCMD2 : numpy array --> 4-dimensional Pyspark accumulator (Hess diagram + latitude + longitude) containing the Hess diagrams of the second M_G' range
        data : numpy array --> 4-dimensional numpy array with the Hess diagrams corresponding to each one of the longitude and latitude ranges
        '''

        MATRIXCMD_complete = self.acc_complete.value
        MATRIXCMD = self.acc.value
        MATRIXCMD2 = self.acc2.value

        acumulador_complete = np.reshape(self.acc_complete.value, np.size(self.acc_complete.value))
        acumulador1 = np.reshape(self.acc.value, np.size(self.acc.value))
        acumulador2 = np.reshape(self.acc2.value, np.size(self.acc2.value))
        data = np.concatenate((acumulador1, acumulador2))

        return MATRIXCMD_complete, MATRIXCMD, MATRIXCMD2, data


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

        global acc_complete, acc, acc2, simple # this line should not be necessary

        sc = self.sc
        MatrixAccumulatorParam = self.MatrixAccumulatorParam

        x2_ps = self.x2_ps
        x3_ps = self.x3_ps

        x1 = self.x1
        x4 = self.x4
        tau_min_edges = self.tau_min_edges
        tau_max_edges = self.tau_max_edges
        ThickParamYoung = self.ThickParamYoung
        ThickParamOld = self.ThickParamOld
        HaloParam = self.HaloParam

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
        Ylims = self.Ylims
        Ylims_Xsteps = self.Ylims_Xsteps
        Ylims_Ysteps = self.Ylims_Ysteps

        tau_min = self.tau_min
        tau_max = self.tau_max
        mass_min = self.mass_min
        mass_max = self.mass_max
        l_min = self.l_min
        l_max = self.l_max
        b_min = self.b_min
        b_max = self.b_max
        r_min = self.r_min
        r_max = self.r_max

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

        acc_complete, acc, acc2, simple = self.accumulators_init()

        # Explored parameters
        alpha1_ps, alpha2_ps, alpha3_ps = param[0:3]
        SigmaParam_ps = np.array([param[3], param[4], param[5], param[6], param[7]+param[8], param[9]+param[10], param[11]+param[12]+param[13]])
        midpopbin_ps = np.array(param[7:11])
        lastpopbin_ps = np.array(param[11:14])

        # If some surface mass density is negative, then Lr='inf' and we redraw again
        if SigmaParam_ps[SigmaParam_ps<0].size==0 and ThickParamYoung>0 and ThickParamOld>0 and lastpopbin_ps[lastpopbin_ps<0].size==0 and midpopbin_ps[midpopbin_ps<0].size==0:

            K1_ps,K2_ps,K3_ps = Continuity_Coeficients_func(alpha1_ps, alpha2_ps, alpha3_ps, x1, x2_ps, x3_ps, x4)
            bin_nor_ps = bin_nor_func(x1, x2_ps, x3_ps, x4, K1_ps, K2_ps, K3_ps, alpha1_ps, alpha2_ps, alpha3_ps, SigmaParam_ps, tau_min_edges, tau_max_edges)

            self.Mother_Simulation_DF.foreach(lambda x: wpes_func(x, Xmin, Xmax, Ymin, Ymax, Bmin, Bmax, Lmin, Lmax, blims, llims, Ylims, Ylims_Xsteps, Ylims_Ysteps, tau_min, tau_max, mass_min, mass_max, l_min, l_max, b_min, b_max, r_min, r_max, x1, x2_ps, x3_ps, K1_ps, K2_ps, K3_ps, alpha1_ps, alpha2_ps, alpha3_ps, SigmaParam_ps, midpopbin_ps, lastpopbin_ps, bin_nor_ps, x2_ms, x3_ms, K1_ms, K2_ms, K3_ms, alpha1_ms, alpha2_ms, alpha3_ms, SigmaParam_ms, midpopbin_ms, lastpopbin_ms, bin_nor_ms, ThickParamYoung, ThickParamOld, HaloParam, acc_complete, acc, acc2, simple))

            self.acc_complete = acc_complete
            self.acc = acc
            self.acc2 = acc2
            self.simple = simple

            self.simulation_data = self.return_cmd()[3]

        else:
            self.simulation_data = np.array([0])

        return self.simulation_data





