'''
Auxiliary functions for BGM FASt

This script includes a set of functions that are needed to make BGM FASt work.
'''


import numpy as np
import pandas as pd
import scipy.integrate as integrate
from bgmfast import parameters


def generate_reduced_MS(Mother_Simulation_DF, x1, x4, mass_step, popbin_list, sel_columns, tau_ranges, Xmin, Xmax, Ymin, Ymax, Bmin, Bmax, Lmin, Lmax, blims, llims, Xstep, Ystep, tau_min, tau_max, mass_min, mass_max, l_min, l_max, b_min, b_max, r_min, r_max, ThickParamYoung):
    
    '''
    Generate a reduced Mother Simulation with fake stars given a discretized range of masses and the different age bins in the SFH (tau ranges).
    
    Input parameters
    ----------------
    Mother_Simulation_DF : pyspark df --> original Mother Simulation filtered by limitting apparent magnitude and parallax>0.0
    x1 : int or float --> minimum mass to generate a star
    x4 : int or float --> maximum mass to generate a star
    mass_step : int or float --> mass step for the definition of the reduced Mother Simulation
    popbin_list : list --> Gaia PopBin values
    sel_columns : dict --> name of the columns we want to keep from the file. The dictionary must follow this order: G, color, PopBin, age, mass, longitude, latitude, parallax, Mvarpi
    tau_ranges : list --> ranges of the different bins of the SFH, both for thin and thick disc in case ThickParamYoung='fit'
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
    ThickParamYoung : int, float or str --> weight of the young thick disc stars. Set to "fit" to compute it by adding SFH9T, SFH10T, SFH11T, and SFH12T to the Galactic parameters to fit
    
    Output parameters
    -----------------
    reduced_Mother_Simulation : pandas df --> table containing the reduced Mother Simulation
    '''
    
    if ThickParamYoung=='fit':
        tau_ranges, T_tau_ranges = tau_ranges
        tau_min, T_tau_min = tau_min
        tau_max, T_tau_max = tau_max 
    
    Mother_Simulation_DF = Mother_Simulation_DF.toPandas().astype(float)
    
    mass_iterations = int((x4 - x1)/mass_step) + 1

    red_PopBin = []
    red_Age = []
    red_MassOut = []
    red_matindex = []
    count_valid_stars = 0
    count_discarded_stars = 0
    count_bins = 0
    for popbin in popbin_list:
        popbin_MS_DF = Mother_Simulation_DF[Mother_Simulation_DF[sel_columns['popbin']]==popbin]
        if popbin<=7:
            for age_ranges in tau_ranges[popbin - 1]:
                low_age, up_age = [float(age) for age in age_ranges]
                mean_age = low_age + (up_age - low_age)/2
                age_popbin_MS_DF = popbin_MS_DF[(popbin_MS_DF[sel_columns['age']]>low_age) & (popbin_MS_DF[sel_columns['age']]<=up_age)]
                for mass_iter in range(mass_iterations):
                    count_bins += 1
                    low_mass = float(x1 + mass_iter*mass_step)
                    up_mass = float(x1 + (mass_iter + 1)*mass_step)
                    mean_mass = x1 + (mass_iter + 0.5)*mass_step
                    red_PopBin.append(popbin)
                    red_Age.append(mean_age)
                    red_MassOut.append(mean_mass)
                    mass_age_popbin_MS_DF = age_popbin_MS_DF[(age_popbin_MS_DF[sel_columns['mass']]>=low_mass) & (age_popbin_MS_DF[sel_columns['mass']]<up_mass)]
                    group_matindex = [compute_bin_func(x, Xmin, Xmax, Ymin, Ymax, Bmin, Bmax, Lmin, Lmax, blims, llims, Xstep, Ystep, [tau_min, T_tau_min], [tau_max, T_tau_max], mass_min, mass_max, l_min, l_max, b_min, b_max, r_min, r_max, ThickParamYoung) for index, x in mass_age_popbin_MS_DF.iterrows()]
                    valid_stars = [matindex for matindex in group_matindex if matindex!=False]
                    red_matindex.append(valid_stars)
                    count_valid_stars += len(valid_stars)
                    count_discarded_stars += len(group_matindex) - len(valid_stars)

        elif popbin==8:
            if ThickParamYoung=='fit':
                for age_ranges in T_tau_ranges:
                    low_age, up_age = [float(age) for age in age_ranges]
                    mean_age = low_age + (up_age - low_age)/2
                    age_popbin_MS_DF = popbin_MS_DF[(popbin_MS_DF[sel_columns['age']]>low_age) & (popbin_MS_DF[sel_columns['age']]<=up_age)]
                    for mass_iter in range(mass_iterations):
                        count_bins += 1
                        low_mass = float(x1 + mass_iter*mass_step)
                        up_mass = float(x1 + (mass_iter + 1)*mass_step)
                        mean_mass = x1 + (mass_iter + 0.5)*mass_step
                        red_PopBin.append(popbin)
                        red_Age.append(mean_age)
                        red_MassOut.append(mean_mass)
                        mass_age_popbin_MS_DF = age_popbin_MS_DF[(age_popbin_MS_DF[sel_columns['mass']]>=low_mass) & (age_popbin_MS_DF[sel_columns['mass']]<up_mass)]
                        group_matindex = [compute_bin_func(x, Xmin, Xmax, Ymin, Ymax, Bmin, Bmax, Lmin, Lmax, blims, llims, Xstep, Ystep, [tau_min, T_tau_min], [tau_max, T_tau_max], mass_min, mass_max, l_min, l_max, b_min, b_max, r_min, r_max, ThickParamYoung) for index, x in mass_age_popbin_MS_DF.iterrows()]
                        valid_stars = [matindex for matindex in group_matindex if matindex!=False]
                        red_matindex.append(valid_stars)
                        count_valid_stars += len(valid_stars)
                        count_discarded_stars += len(group_matindex) - len(valid_stars)
            else:
                count_bins += 1
                red_PopBin.append(popbin)
                red_Age.append(0)
                red_MassOut.append(0)
                mass_age_popbin_MS_DF = popbin_MS_DF
                group_matindex = [compute_bin_func(x, Xmin, Xmax, Ymin, Ymax, Bmin, Bmax, Lmin, Lmax, blims, llims, Xstep, Ystep, tau_min, tau_max, mass_min, mass_max, l_min, l_max, b_min, b_max, r_min, r_max, ThickParamYoung) for index, x in mass_age_popbin_MS_DF.iterrows()]
                valid_stars = [matindex for matindex in group_matindex if matindex!=False]
                red_matindex.append(valid_stars)
                count_valid_stars += len(valid_stars)
                count_discarded_stars += len(group_matindex) - len(valid_stars)
                                    
        elif popbin>8:
            count_bins += 1
            red_PopBin.append(popbin)
            red_Age.append(0)
            red_MassOut.append(0)
            mass_age_popbin_MS_DF = popbin_MS_DF
            if ThickParamYoung=='fit':
                group_matindex = [compute_bin_func(x, Xmin, Xmax, Ymin, Ymax, Bmin, Bmax, Lmin, Lmax, blims, llims, Xstep, Ystep, [tau_min, T_tau_min], [tau_max, T_tau_max], mass_min, mass_max, l_min, l_max, b_min, b_max, r_min, r_max, ThickParamYoung) for index, x in mass_age_popbin_MS_DF.iterrows()]
            else:
                group_matindex = [compute_bin_func(x, Xmin, Xmax, Ymin, Ymax, Bmin, Bmax, Lmin, Lmax, blims, llims, Xstep, Ystep, tau_min, tau_max, mass_min, mass_max, l_min, l_max, b_min, b_max, r_min, r_max, ThickParamYoung) for index, x in mass_age_popbin_MS_DF.iterrows()]
            valid_stars = [matindex for matindex in group_matindex if matindex!=False]
            red_matindex.append(valid_stars)
            count_valid_stars += len(valid_stars)
            count_discarded_stars += len(group_matindex) - len(valid_stars)
            
    print('Number of valid stars: %i' %count_valid_stars)
    print('Number of stars discarded: %i' %count_discarded_stars)
    print('Number of bins in the reduced Mother Simulation: %i' %count_bins)
    
    reduced_Mother_Simulation = {'PopBin': red_PopBin, 'Age': red_Age, 'MassOut': red_MassOut, 'Matindex': red_matindex}
    reduced_Mother_Simulation = pd.DataFrame(reduced_Mother_Simulation)
    
    return reduced_Mother_Simulation


def Continuity_Coeficients_func(alpha1, alpha2, alpha3, x1, x2, x3, x4):

    '''
    Compute the continuity coeficients of the Initial Mass Function (IMF), which we describe as a three-trunkated power law. We are looking for the coeficients that will let us have continuity in the IMF by applying the normalization stated next to Eq. (30) in Mor et al. 2018. Therefore, the three conditions we apply to find them are: normalization, continuity in x2 and continuity in x3. Applying these conditions we can find the functions that are used to compute the continuity coeficients. 

    Input parameters
    ----------------
    alpha1 : int or float --> first slope (alpha) of the IMF
    alpha2 : int or float --> second slope (alpha) of the IMF
    alpha3 : int or float --> third slope (alpha) of the IMF
    x1 : int or float --> minimum mass to generate a star
    x2 : int or float --> first mass limit of the IMF
    x3 : int or float --> second mass limit of the IMF
    x4 : int or float --> maximum mass of a star

    Output parameters
    -----------------
    K1 : float --> first continuity coeficient
    K2 : float --> second continuity coeficient
    K3 : float --> third continuity coeficient
    '''

    # We use one equation for each IMF slope, f1,f2,f3.
    f1 = lambda m: m**(-alpha1)*m
    f2 = lambda m: m**(-alpha2)*m
    f3 = lambda m: m**(-alpha3)*m

    # Integration and computation of the Continuity Coeficients
    I1 = integrate.quad(f1, x1, x2)
    I2 = integrate.quad(f2, x2, x3)
    I3 = integrate.quad(f3, x3, x4)

    K1K2=x2**(-alpha2 + alpha1)
    K2K3=x3**(-alpha3 + alpha2)

    K1 = 1/(I1[0] + 1/K1K2*I2[0] + 1/(K1K2*K2K3)*I3[0])
    K2 = K1/K1K2
    K3 = K1/(K1K2*K2K3)

    return K1, K2, K3


def binning_4D_Mvarpi(S, Xmin, Xmax, Ymin, Ymax, Bmin, Bmax, Lmin, Lmax, blims, llims, Xstep, Ystep):

    '''
    Compute the position of a give star in bins in the discretized space of the 4-dimensional (latitude, longitude, G-Rp colour and absolute magnitude) accumulators

    Input parameters
    ----------------
    S : list [G-Rp, longitude, latitude, M_G'] --> colour, longitude, latitude and absolute magnitude of the star
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

    Output parameters
    -----------------
    ILS : int --> bin number in the longitude discretized space
    IBS : int --> bin number in the latitude discretized space
    IXS : int --> bin number in the G-Rp discretized space
    IYS : int --> bin number in the M_G' discretized space
    '''

    XS = float(S[0]) # Color G-Rp
    lS = float(S[1]) # Longitude
    bS = float(S[2]) # Latitude
    YS = float(S[3]) # M_G' magnitude

    # Checking that the values are inside the desired limits
    if (Ymin<=YS<Ymax and Bmin<=bS<=Bmax and Lmin<=lS<=Lmax and Xmin<=XS<Xmax):

        for blim, i in zip(blims, range(len(blims))):
            if blim[0]<=abs(bS)<=blim[1]:
                IBS = i

        for llim, i in zip(llims, range(len(llims))):
            if llim[0]<=abs(lS)<=llim[1]:
                ILS = i

        IYS = int((YS - Ymin)/Ystep)
        IXS = int((XS - Xmin)/Xstep)

    # Giving a value for this stars which is not in the desired ranges
    else:
        IYS = np.nan
        IXS = np.nan
        IBS = np.nan
        ILS = np.nan

    return ILS, IBS, IXS, IYS


def compute_bin_func(WP, Xmin, Xmax, Ymin, Ymax, Bmin, Bmax, Lmin, Lmax, blims, llims, Xstep, Ystep, tau_min, tau_max, mass_min, mass_max, l_min, l_max, b_min, b_max, r_min, r_max, ThickParamYoung):

    '''
    Same as in binning_4D_Mvarpi function but checking also the constraints in age, mass, longitude, latitude, and distance.

    Input parameters
    ----------------
    WP : list -->
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
    tau_min : int or float or list --> minimum age of a thin disc star. In case ThickParamYoung=='fit', tau_min is a list with tau_min and T_tau_min
    tau_max : int or float or list --> maximum age of a thin disc star. In case ThickParamYoung=='fit', tau_max is a list with tau_max and T_tau_max
    mass_min : int or float --> minimum mass to generate a star
    mass_max : int or float --> maximum mass to generate a star
    l_min : int or float --> minimum Galactic longitude
    l_max : int or float --> maximum Galactic longitude
    b_min : int or float --> minimum Galactic latitude
    b_max : int or float --> maximum Galactic latitude
    r_min : int or float --> minimum distance
    r_max : int or float --> maximum distance
    ThickParamYoung : int or float --> weight of the stars in the Young Thick disc. Set to "fit" to compute it by adding SFH9T, SFH10T, SFH11T, and SFH12T to the galactic parameters to fit

    Output parameters
    -----------------
    wpes : int --> weight of the star derived from BGM FASt
    '''
    
    GRperr = float(WP[1])
    popbin = float(WP[2])
    tau = float(WP[3]) # Age of the star
    mass = float(WP[4]) # Mass of the star
    lstar = float(WP[5]) # Longitude of the star
    bstar = float(WP[6]) # Latitude of the star
    parallax = float(WP[7])
    rstar = 1/parallax*1000. # Distance of the star (pc)
    Mvarpi = float(WP[8])
    
    Sinput = [GRperr, lstar, bstar, Mvarpi]
    matindex = binning_4D_Mvarpi(Sinput, Xmin, Xmax, Ymin, Ymax, Bmin, Bmax, Lmin, Lmax, blims, llims, Xstep, Ystep)
    
    if ThickParamYoung=='fit':
        tau_min, T_tau_min = tau_min
        tau_max, T_tau_max = tau_max

    if (popbin<=7 and tau_min<=tau<=tau_max and mass_min<=mass<=mass_max and l_min<=lstar<=l_max and b_min<=bstar<=b_max and r_min<=rstar<=r_max) or (popbin==8 and ThickParamYoung=='fit' and T_tau_min<=tau<=T_tau_max and mass_min<=mass<=mass_max and l_min<=lstar<=l_max and b_min<=bstar<=b_max and r_min<=rstar<=r_max) or popbin==8 or popbin==9 or popbin==10 or popbin==11:
        pass
        
    else:
        print('Not correct popbin %i or age %f' %(popbin, tau))
        import sys
        sys.exit()

    if (np.isnan(matindex[0]) or np.isnan(matindex[1]) or np.isnan(matindex[2]) or np.isnan(matindex[3])):
        return False
    else:
        return matindex


def Simplified_Gi_Primal_func_NONP(itau, smass, x1, x2, x3, K1, K2, K3, alpha1, alpha2, alpha3, SigmaParam, bin_nor, midpopbin, lastpopbin, imidpoptau, ilastpoptau, structure):

    '''
    Computation of the numerator or the denominator of Equation (37) from Mor et al. 2018. Sigma_all/bin_nor is exactly Sigma_primal. This computation corresponds to a single star and it is used to obtain the weight of that star.

    Input parameters
    ----------------
    itau : int --> index corresponding to the subpopulation (itau = popbin - 1)
    smass : int or float --> mass of the star
    x1 : int or float --> minimum mass to generate a star
    x2 : int or float --> first mass limit of the IMF
    x3 : int or float --> second mass limit of the IMF
    K1 : int or float --> first continuity coeficient of the IMF
    K2 : int or float --> second continuity coeficient of the IMF
    K3 : int or float --> third continuity coeficient of the IMF
    alpha1 : int or float --> first slope (alpha) of the IMF
    alpha2 : int or float --> second slope (alpha) of the IMF
    alpha3 : int or float --> third slope (alpha) of the IMF
    SigmaParam : list --> surface density at the position of the Sun for the different age subpopulations
    bin_nor : int or float --> normalization coeficient for binaries
    midpopbin : list --> surface density at the position of the Sun for the four subdivisions of the 5th and 6th age subpopulations of the thin disc (3-5 Gyr and 5-7 Gyr)
    lastpopbin : list --> surface density at the position of the Sun for the three subdivisions of the last (7th) age subpopulation of the thin disc (7-10 Gyr) or surface density at the position of the Sun for the two subdivions of the age population of the young thick disc(8-10 Gyr)
    imidpoptau : int --> corresponding index of the midpopbin list
    ilastpoptau : int --> corresponding index of the lastpopbin list
    structure : str --> whether we are working with the "thin" disc or the "youngthick" disc

    Output parameters
    -----------------
    integralout : float --> numerator or denominator of Eq. (37) from Mor et al. 2018 for a given star
    '''

    imassmin = smass
    imassmax = smass + 0.025

    if x1<=smass<x2:
        imf = (K1*(imassmax**(-alpha1 + 2) - ((imassmin)**(-alpha1 + 2)))/(-alpha1 + 2))
    elif x2<=smass<=x3:
        imf = (K2*(imassmax**(-alpha2 + 2) - ((imassmin)**(-alpha2 + 2)))/(-alpha2 + 2))
    elif smass>x3:
        imf = (K3*(((imassmax)**(-alpha3 + 2)) - ((imassmin)**(-alpha3 + 2)))/(-alpha3 + 2))

    # Notice the result is divided by bin_nor as Sigma_all/bin_nor is Sigma_primal
    if structure=='thin':
        if itau==4 or itau==5:
            integralout = (midpopbin[imidpoptau])*imf/bin_nor
        elif itau==6:
            integralout = (lastpopbin[ilastpoptau])*imf/bin_nor
        else:
            integralout = (SigmaParam[itau])*imf/bin_nor
    elif structure=='youngthick':
        integralout = (SigmaParam[itau])*imf/bin_nor

    return integralout


# **********************************************
# BINARIES COMPUTATION (PROBABILITIES) FUNCTIONS
# **********************************************

def Omega_func_aux2(taumax, taumin, m):

    '''
    Computation of the auxiliary function needed to compute the Omega function. The expressions for the age limit given the mass are fitted to the stellar evolutionary tracks inculded in Besançon Galaxy Model (Czekaj, et al. 2014) and Bertelli et al. 2008.

    Input parameters
    ----------------
    taumax : int or float --> upper limit of the age subpopulation interval in years
    taumin : int or float --> lower limit of the age subpopulation interval in years
    m : int or float --> mass of the star

    Output parameters
    -----------------
    omega : float --> auxiliari value needed for later computations
    '''

    if (m>20):
        taulim = np.exp(-0.61533562*np.log(m)+17.85867643) # New after sumbitting the paper
    elif (7<=m<=20): # Equation (53) from Mor et al. 2018
        taulim = np.exp(-1.57703799*np.log(m)+20.77243055) # tau limit is the maximum age of a star given its mass
    elif (2.2<m<7):
        taulim = np.exp(-2.72386552*np.log(m)+23.0181053) # Equation (54) from Mor et al. 2018
    elif (2<m<2.2):
        taulim = np.exp(-2.72386552*np.log(2.2)+23.0181053)   # Equation (55) from Mor et al. 2018
    elif (m<=2):
        taulim = np.exp(-3.44517144*np.log(m)+23.26518166) # Equation (56) from Mor et al. 2018

    # The next piece of code is the Equation (57) from Mor et al. 2018
    if (taulim>=taumax) :
        omega = 1
    elif (taulim<=taumin) :
        omega = 0
    else:
        omega = (taulim - taumin)/(taumax - taumin)

    return omega


def Omega_func(m, itau, tau_min_edges, tau_max_edges):

    '''
    Computation of Equation (57) from Mor et al. 2018. We define one Omega function for each thin disc age sub-population of the BGM. See Robin et al. 2003, Mor et al. 2017 to get the used age limits for each interval.

    Input parameters
    ----------------
    m : int or float --> mass of the star
    itau : int --> index corresponding to the subpopulation (itau = popbin - 1). In case ThickParamYoung=='fit' and itau==7, itau is set to 0
    tau_min_edges : list --> lower limits of the age subpopulations intervals
    tau_max_edges : list --> upper limits of the age subpopulations intervals

    Output parameters
    -----------------
    Omega_interv : float --> exactly omega (see description in Omega_func_aux2 function)
    '''

    taumin = tau_min_edges[itau]*10**9
    taumax = tau_max_edges[itau]*10**9
    Omega_interv = Omega_func_aux2(taumax, taumin, m)

    return Omega_interv


def prob_M_m_func(m, x1):

    '''
    Computation of the probability to obtain a secondary star of mass m given a primary of mass M. We approximate this probability as a uniform distribution between x1 (the adopted minimum mass for the BGM thin disc from Czekaj et al. 2014) and M (the mass of the primary component). This is part of Equation (48) from Mor et al. 2018

    Input parameters
    ----------------
    m : int or float --> mass of the star
    x1 : int or float --> minimum mass to generate a star

    Output parameters
    -----------------
    prob_M_m : float --> value of the probability mentioned in the description
    '''

    prob_M_m = 1/(m - x1)

    return prob_M_m


def int_prob_M_m_func(m, x1):

    '''
    Computation of the integral corresponding to the probabilities defined in the description of the prob_M_m_func function

    Input parameters
    ----------------
    m : int or float --> mass of the star
    x1 : int or float --> minimum mass to generate a star

    Output parameters
    -----------------
    I1_M_m : float --> value of the integral mentioned in the description
    '''

    f_M_m = lambda M: prob_M_m_func(m, x1)*M
    I1_M_m = integrate.quad(f_M_m, x1, m)[0]

    return I1_M_m


def bin_prob_func(M1):

    '''
    Computation of the probability that a star of mass M1 belongs to a multiple system. This is part of the implementaiton of Equation (49) from Mor et al. 2018. This expression of the probability is taken directly from the BGM Standard code and defers a bit from the expression in the Arenou's paper. This is the expression of Frederic Arenou for the main sequence stars. For the moment we assume that all stars have this pdf. Maybe later we can think how we can introduce the 60% forced for Giants.

    Input parameters
    ----------------
    M1 : int or float --> mass of the star

    Output parameters
    -----------------
    bin_Prob : float --> value of the probability mentioned in the description
    '''

    bin_Prob = 0.85*np.tanh(0.55*M1 + 0.095)

    return bin_Prob


def f_toint1_func3_NONP(itau, x1, x2, x3, x4, K1, K2, K3, alpha1, alpha2, alpha3, SigmaParam, tau_min_edges, tau_max_edges):

    '''
    Computation of the binarity normalization term. From Equation (31) from  Mor et al. 2018 one can write Sigma_all=Sigma_primals+Sigma_Secondaries; Sigma_Secondaries=Sigma_primal*Term, then Sigma_all=Sigma_primals*(1+Term); This function gives us Term for a given subpopulation. Its formulation comes from Equation (52) from Mor et al. 2018 dividing both sides by Sigma_primal. In this case, we compute the binarity normalization term for a non-parametric SFH.

    Input parameters
    ----------------
    itau : int --> index corresponding to the subpopulation (itau = popbin - 1)
    x1 : int or float --> minimum mass to generate a star
    x2 : int or float --> first mass limit of the IMF
    x3 : int or float --> second mass limit of the IMF
    x4 : int or float --> maximum mass of a star
    K1 : int or float --> first continuity coeficient of the IMF
    K2 : int or float --> second continuity coeficient of the IMF
    K3 : int or float --> third continuity coeficient of the IMF
    alpha1 : int or float --> first slope (alpha) of the IMF
    alpha2 : int or float --> second slope (alpha) of the IMF
    alpha3 : int or float --> third slope (alpha) of the IMF
    SigmaParam : list --> surface density at the position of the Sun for the different age subpopulations
    tau_min_edges : list --> lower limits of the age subpopulations intervals
    tau_max_edges : list --> upper limits of the age subpopulations intervals

    Output parameters
    -----------------
    integralout : float --> value of the normalization term of the given subpopulation
    '''
    
    f1 = lambda m:m**(-alpha1)*Omega_func(m, itau, tau_min_edges, tau_max_edges)*bin_prob_func(m)*int_prob_M_m_func(m, x1)
    f2 = lambda m:m**(-alpha2)*Omega_func(m, itau, tau_min_edges, tau_max_edges)*bin_prob_func(m)*int_prob_M_m_func(m, x1)
    f3 = lambda m:m**(-alpha3)*Omega_func(m, itau, tau_min_edges, tau_max_edges)*bin_prob_func(m)*int_prob_M_m_func(m, x1)

    I1_s = integrate.quad(f1,x1,x2)
    I2_s = integrate.quad(f2,x2,x3)
    I3_s = integrate.quad(f3,x3,x4)

    integral = K1*I1_s[0] + K2*I2_s[0] + K3*I3_s[0]
    integralout = (SigmaParam[itau]/sum(SigmaParam))*integral

    return integralout


def bin_nor_func(x1, x2, x3, x4, K1, K2, K3, alpha1, alpha2, alpha3, SigmaParam, tau_min_edges, tau_max_edges):

    '''
    Computation of the normalization due to secondary (binary) stars. See complemetary information in the description of f_toint1_func3_NONP function

    Input parameters
    ----------------
    x1 : int or float --> minimum mass to generate a star
    x2 : int or float --> first mass limit of the IMF
    x3 : int or float --> second mass limit of the IMF
    x4 : int or float --> maximum mass of a star
    K1 : int or float --> first continuity coeficient of the IMF
    K2 : int or float --> second continuity coeficient of the IMF
    K3 : int or float --> third continuity coeficient of the IMF
    alpha1 : int or float --> first slope (alpha) of the IMF
    alpha2 : int or float --> second slope (alpha) of the IMF
    alpha3 : int or float --> third slope (alpha) of the IMF
    SigmaParam : list --> surface density at the position of the Sun for the different age subpopulations
    tau_min_edges : list --> lower limits of the age subpopulations intervals
    tau_max_edges : list --> upper limits of the age subpopulations intervals

    Output parameters
    -----------------
    bin_nor : float --> value of the normalization term
    '''

    binarity_norm = []
    for i in range(len(SigmaParam)):
            binarity_norm.append(f_toint1_func3_NONP(i, x1, x2, x3, x4, K1, K2, K3, alpha1, alpha2, alpha3, SigmaParam, tau_min_edges, tau_max_edges))

    bin_nor = sum(binarity_norm) + 1.

    return bin_nor


# **********************************************

def compute_bin_quotient(H_t_bin, H0_bin, thresh=parameters.distance_parameters['dist_thresh'].value):
    '''
    Compute the quotient in a given bin of two Hess diagrams
    
    Input parameters
    ----------------
    H_t_bin : int or float --> number of counts in the given catalog bin
    H0_bin : int or float --> number of counts in the given simulation bin
    thresh : int or float --> minimum threshold for the number of stars per bin in the catalog to consider that bin for the computation of the distance. Set the threshold to -1 to deactivate the threshold
    
    Output parameters
    -----------------
    quot_bin : float --> quotient in the bin between the catalog and the simulation
    '''
    
    if H_t_bin<thresh:
        quot_bin = 1
    elif H_t_bin==0 or H0_bin==0:
        quot_bin = (H0_bin + 1)/(H_t_bin + 1)
    else:
        quot_bin = H0_bin/H_t_bin
    
    return quot_bin


def compute_bin_distance(H_t_bin, H0_bin, thresh=parameters.distance_parameters['dist_thresh'].value):
    
    '''
    Compute the distance metric in a given bin of two Hess diagrams
    
    Input parameters
    ----------------
    H_t_bin : int or float --> number of counts in the given catalog bin
    H0_bin : int or float --> number of counts in the given simulation bin
    thresh : int or float --> minimum threshold for the number of stars per bin in the catalog to consider that bin for the computation of the distance. Set the threshold to -1 to deactivate the threshold
    
    Output parameters
    -----------------
    dist_bin : float --> distance in the bin between the catalog and the simulation
    '''
    
    if H_t_bin<thresh:
        dist_bin = 0
    elif H_t_bin==0 or H0_bin==0:
        R_bin = compute_bin_quotient(H_t_bin, H0_bin, thresh)
        dist_bin = (H_t_bin + 1)*(1 - R_bin + np.log(R_bin)) 
    else:
        R_bin = compute_bin_quotient(H_t_bin, H0_bin, thresh)
        dist_bin = H_t_bin*(1 - R_bin + np.log(R_bin))
    
    return dist_bin


def dist_metric_gdaf2(H_t, HO, thresh=parameters.distance_parameters['dist_thresh'].value):

    '''
    Computation of the distance metric to be used in the ABC code. We use the so-called Poissonian distance. The expression can be found in the Equation (58) from Mor et al. 2018.

    Input parameters
    ----------------
    H_t : 4-dimensional accumulator -->  catalog data in the 4-dimensional space (Hess diagram + latitude + longitude) used as a summary statistics
    H0 : 4-dimensional accumulator --> simulation data in the 4-dimensional space (Hess diagram + latitude + longitude) used as a summary statistics
    thresh : int or float --> minimum threshold for the number of stars per bin in the catalog to consider that bin for the computation of the distance. Set the threshold to -1 to deactivate the threshold

    Output parameters
    -----------------
    lrout : float --> value of the distance
    '''

    if HO.size==1:
        lrout = np.inf

    else:
        Lrb = [compute_bin_distance(i, j, thresh) for i,j in zip(H_t, HO)]

        lrout = np.abs(sum(Lrb))
        if np.isnan(lrout):
            lrout = np.inf

    return lrout
