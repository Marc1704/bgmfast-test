'''
Auxiliary functions for BGM FASt

This script includes a set of functions that are needed to make BGM FASt work.
'''


import numpy as np
import scipy.integrate as integrate
from bgmfast import parameters


def Continuity_Coeficients_func(alpha1, alpha2, alpha3, x1, x2, x3, x4):

    '''
    Compute the continuity coeficients of the Initial Mass Function (IMF), which we describe as a three-trunkated power law

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
    f1=lambda m: m**(-alpha1+1)
    f2=lambda m: m**(-alpha2+1)
    f3=lambda m: m**(-alpha3+1)

    # Integration and computation of the Continuity Coeficients
    I1=integrate.quad(f1, x1, x2)
    I2=integrate.quad(f2, x2, x3)
    I3=integrate.quad(f3, x3, x4)

    K1K2=x2**(-alpha2+1)/x2**(-alpha1+1)
    K2K3=x3**(-alpha3+1)/x3**(-alpha2+1)

    K1 = 1/(I1[0] + 1/K1K2*I2[0] + 1/(K1K2*K2K3)*I3[0])
    K2 = K1/K1K2
    K3 = K1/(K1K2*K2K3)

    return K1, K2, K3


def binning_4D_Mvarpi(S, Xmin, Xmax, Ymin, Ymax, Bmin, Bmax, Lmin, Lmax, blims, llims, Ylims, Ylims_Xsteps, Ylims_Ysteps):

    '''
    Compute the position of a give star in bins in the discretized space of the 4-dimensional (latitude, longitude, Bp-Rp colour and absolute magnitude) accumulators

    Input parameters
    ----------------
    S : list [Bp-Rp, longitude, latitude, M_G'] --> colour, longitude, latitude and absolute magnitude of the star
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

    Output parameters
    -----------------
    ILS : int --> bin number in the longitude discretized space
    IBS : int --> bin number in the latitude discretized space
    IXS : int --> bin number in the Bp-Rp discretized space
    IYS : int --> bin number in the M_G' discretized space
    IXS_complete : int --> bin number in the Bp-Rp discretized space given even in the case of being outside the first range of M_G'
    IYS_complete : int --> bin number in the M_G' discretized space given even in the case of being outside the first range of M_G'
    '''

    XS = float(S[0]) # Color Bp-Rp
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

        #for Ylim, i in zip(Ylims, range(len(Ylims))):
            #if not Xmin<=XS<Xmax:
                #break
            #if Ylim[0]<YS<=Ylim[1]:
                #IYS = int((YS - Ylim[1])/Ylims_Ysteps[i])
                #IXS = int((XS - Xmin)/Ylims_Xsteps[i])
        
        if Ylims[0][0]<=YS<Ylims[0][1]:
            IYS = int((YS - Ylims[0][0])/Ylims_Ysteps[0])
            IXS = int((XS - Xmin)/Ylims_Xsteps[0])
        elif Ylims[1][0]<=YS<Ylims[1][1]:
            IYS = int((YS - Ylims[1][0])/Ylims_Ysteps[1])
            IXS = int((XS - Xmin)/Ylims_Xsteps[1])

        # Giving a value for this stars which is not in the desired ranges
        else:
            IYS = np.nan
            IXS = np.nan
            IBS = np.nan
            ILS = np.nan

        if np.isnan(IBS) or np.isnan(ILS) or np.isnan(IXS) or np.isnan(IYS):
            IYS_complete = np.nan
            IXS_complete = np.nan
        else:
            IYS_complete = int((YS - Ymin)/Ylims_Ysteps[0])
            IXS_complete = int((XS - Xmin)/Ylims_Xsteps[0])

    # Giving a value for this stars which is not in the desired ranges
    else:
        IYS = np.nan
        IXS = np.nan
        IBS = np.nan
        ILS = np.nan
        IYS_complete = np.nan
        IXS_complete = np.nan

    return ILS, IBS, IXS, IYS, IXS_complete, IYS_complete


def Simplified_Gi_Primal_func_NONP(itau, smass, x1, x2, x3, K1, K2, K3, alpha1, alpha2, alpha3, SigmaParam, bin_nor, midpopbin, lastpopbin, imidpoptau, ilastpoptau):

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
    SigmaParam : list --> surface density at the position of the Sun for the different age subpopulations of the thin disc
    bin_nor : int or float --> normalization coeficient for binaries
    midpopbin : list --> surface density at the position of the Sun for the four subdivisions of the 5th and 6th age subpopulations of the thin disc (3-5 Gyr and 5-7 Gyr)
    lastpopbin : list --> surface density at the position of the Sun for the three subdivisions of the last (7th) age subpopulation of the thin disc (7-10 Gyr)
    imidpoptau : int --> corresponding index of the midpopbin list
    ilastpoptau : int --> corresponding index of the lastpopbin list

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
    if itau==4 or itau==5:
        integralout = (midpopbin[imidpoptau])*imf/bin_nor
    elif itau==6:
        integralout = (lastpopbin[ilastpoptau])*imf/bin_nor
    else:
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
    itau : int --> index corresponding to the subpopulation (itau = popbin - 1)
    tau_min_edges : list --> lower limits of the age subpopulations intervals of the thin disc
    tau_max_edges : list --> upper limits of the age subpopulations intervals of the thin disc

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
    SigmaParam : list --> surface density at the position of the Sun for the different age subpopulations of the thin disc
    tau_min_edges : list --> lower limits of the age subpopulations intervals of the thin disc
    tau_max_edges : list --> upper limits of the age subpopulations intervals of the thin disc

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
    SigmaParam : list --> surface density at the position of the Sun for the different age subpopulations of the thin disc
    tau_min_edges : list --> lower limits of the age subpopulations intervals of the thin disc
    tau_max_edges : list --> upper limits of the age subpopulations intervals of the thin disc

    Output parameters
    -----------------
    bin_nor : float --> value of the normalization term
    '''

    binarity_norm = []
    for i in range(0,7):
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
