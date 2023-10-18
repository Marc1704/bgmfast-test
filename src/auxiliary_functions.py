'''
Auxiliary functions for BGMFASt simuation.
'''



# *******
# IMPORTS
# *******

import numpy as np
import scipy.integrate as integrate
import math


# ************************************
# CONTINUITY COEFICIENTS FUNCTION (F1)
# ************************************

def Continuity_Coeficients_func(alpha1, alpha2, alpha3, x1, x2, x3, x4):

    '''
    Compute the continuity coeficients of the IMF. We describe the IMF as three trunkated power laws.
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


# ***********************************
# FROM DATA TO CMD BINS FUNCTION (F2)
# ***********************************

def binning_4D_Mvarpi(S, Xmin, Xmax, Ymin, Ymax, Bmin, Bmax, Lmin, Lmax, blims, llims, Ylims, Ylims_Xsteps, Ylims_Ysteps):

    '''
    Compute the position of each one of the stars in bins in the discretized space of the 4-dimensional accumulators.
    '''

    XS = float(S[1]) # Color Bp-Rp
    lS = float(S[2]) # Longitude
    bS = float(S[3]) # Latitude
    YS = float(S[4]) # M_G' magnitude

    # Checking that the values are inside the desired limits
    if (Ymin<=YS<Ymax and Bmin<=bS<=Bmax and Lmin<=lS<=Lmax):

        for blim, i in zip(blims, range(len(blims))):
            if blim[0]<abs(bS)<=blim[1]:
                IBS = i

        for llim, i in zip(llims, range(len(llims))):
            if llim[0]<abs(lS)<=llim[1]:
                ILS = i

        #for Ylim, i in zip(Ylims, range(len(Ylims))):
            #if not Xmin<=XS<Xmax:
                #break
            #if Ylim[0]<YS<=Ylim[1]:
                #IYS = int((YS - Ylim[1])/Ylims_Ysteps[i])
                #IXS = int((XS - Xmin)/Ylims_Xsteps[i])

        if Ylims[1][0]<YS<Ylims[1][1]:
            IYS = int((YS - Ylims[1][0])/Ylims_Ysteps[1])
            IXS = int((XS - Xmin)/Ylims_Xsteps[1])

        elif Xmin<=XS<Xmax:
            if Ylims[0][0]<YS<Ylims[0][1]:
                IYS = int((YS - Ylims[0][0])/Ylims_Ysteps[0])
                IXS = int((XS - Xmin)/Ylims_Xsteps[0])

        # Giving a value for this stars which is not in the desired ranges
        else:
            IYS = np.nan
            IXS = np.nan
            IBS = np.nan
            ILS = np.nan

        if np.isnan(IBS) or np.isnan(ILS) or np.isnan(IXS) or np.isnan(IYS) or XS<Xmin or XS>=Xmax:
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



# ************************
# GI PRIMAL FUNCTIONS (F4)
# ************************

def Simplified_Gi_Primal_func_NONP(itau, smass, x1, x2, x3, NK1, NK2, NK3, alpha1, alpha2, alpha3, SigmaParam, bin_nor, midpopbin, lastpopbin, imidpoptau, ilastpoptau):

    '''
    Numerator of Equation (37) from Mor et al. 2018. Sigma_all/bin_nor is exactly Sigma_primal. This function is exactly the same as Simplified_Gi_Primal_func but using a non-parametric SFH.
    '''

    imassmin = smass
    imassmax = smass + 0.025

    if x1<=smass<x2:
        imf = (NK1*(imassmax**(-alpha1 + 2) - ((imassmin)**(-alpha1 + 2)))/(-alpha1 + 2))
    elif x2<=smass<=x3:
        imf = (NK2*(imassmax**(-alpha2 + 2) - ((imassmin)**(-alpha2 + 2)))/(-alpha2 + 2))
    elif smass>x3:
        imf = (NK3*(((imassmax)**(-alpha3 + 2)) - ((imassmin)**(-alpha3 + 2)))/(-alpha3 + 2))

    # Notice the result is divided by bin_nor as Sigma_all/bin_nor is Sigma_primal
    if itau==4 or itau==5:
        integralout = (midpopbin[imidpoptau])*imf/bin_nor
    elif itau==6:
        integralout = (lastpopbin[ilastpoptau])*imf/bin_nor
    else:
        integralout = (SigmaParam[itau])*imf/bin_nor

    return integralout


# **********************************************************
# BINARIES COMPUTATION (PROBABILITIES) FUNCTIONS (F3.1-F3.3)
# **********************************************************

def Omega_func_aux2(taumax9, taumin9, m):

    '''
    F3.1.1) Auxiliary function to compute the Omega function. The expressions for the age limit given the mass are fitted to the stellar evolutionary tracks inculded in Besançon Galaxy Model (Czekaj, et al. 2014) and Bertelli et al. 2008.
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
    if (taulim>=taumax9) :
        omega = 1
    elif (taulim<=taumin9) :
        omega = 0
    else:
        omega = (taulim - taumin9)/(taumax9 - taumin9)

    return omega


def Omega_func(m, itau, tau_min_edges, tau_max_edges):

    '''
    F3.1) We define one Omega function for each thin disc age sub-population of the BGM. See Robin et al. 2003, Mor et al. 2017 to get the used age limits for each interval. Here we compute taumin and taumax in yrs and we call the Omega_func_aux2 function. We do that to apply Equation (57) from Mor et al. 2018.
    '''

    taumin = tau_min_edges[itau]*10**9
    taumax = tau_max_edges[itau]*10**9
    Omega_interv = Omega_func_aux2(taumax,taumin,m)

    return Omega_interv


def prob_M_m_func(m, x1):

    '''
    F3.3.1) Approximation of the probability to obtain a secondary star of mass m given a primary of mass M. We approximate this probability as a uniform distribution between x1 (the adopted minimum mass for the BGM thin disc from Czekaj et al. 2014) and M (the mass of the primary component). This is part of Equation (48) from Mor et al. 2018.
    '''

    prob_M_m = 1/(m - x1)

    return prob_M_m


def int_prob_M_m_func(m, x1):

    '''
    F3.3) Approximation of the probability to obtain a secondary of mass m given a primary of Mass M. We approximate this probability as a uniform distribution between x1 (the adopted minimum mass for the BGM thin disc) and M (the mass of the primary component). This is part of Equation (48) Mor et al. 2018.
    '''

    f_M_m = lambda M: prob_M_m_func(m, x1)*M
    I1_M_m = integrate.quad(f_M_m, x1, m)

    return I1_M_m[0]


def bin_prob_func(M1):

    '''
    F3.2) Function of the probability that a star of mass M1 belongs to a multiple system. This is part of the implementaiton of Equation (49) from Mor et al. 2018. This expression of the probability is taken directly from the BGM Standard code and defers a bit from the expression in the Arenou's paper. This is the expression of Frederic Arenou for the main sequence stars for the moment we assume that all stars have this pdf. Maybe later we can think how we can introduce the 60% forced for Giants.
    '''

    bin_Prob = 0.85*np.tanh(0.55*M1 + 0.095)

    return bin_Prob


def f_toint1_func3_NONP(itau, x1, x2, x3, x4, NK1, NK2, NK3, alpha1, alpha2, alpha3, SigmaParam, tau_min_edges, tau_max_edges):

    '''
    F3) Function used to compute the binarity normalization term. From Equation (31) from  Mor et al. 2018 one can write Sigma_all=Sigma_primals+Sigma_Secondaries; Sigma_Secondaries=Sigma_primal*Term, then Sigma_all=Sigma_primals*(1+Term); This function gives us Term. Its formulation comes from Equation (52) from Mor et al. 2018 dividing both sides by Sigma_primal. In this case, we compute the binarity normalization term for a non-parametric SFH.
    '''

    f1 = lambda m:m**(-alpha1)*Omega_func(m, itau, tau_min_edges, tau_max_edges)*bin_prob_func(m)*int_prob_M_m_func(m, x1)
    f2 = lambda m:m**(-alpha2)*Omega_func(m, itau, tau_min_edges, tau_max_edges)*bin_prob_func(m)*int_prob_M_m_func(m, x1)
    f3 = lambda m:m**(-alpha3)*Omega_func(m, itau, tau_min_edges, tau_max_edges)*bin_prob_func(m)*int_prob_M_m_func(m, x1)

    I1_s = integrate.quad(f1,x1,x2)
    I2_s = integrate.quad(f2,x2,x3)
    I3_s = integrate.quad(f3,x3,x4)

    integral = NK1*I1_s[0] + NK2*I2_s[0] + NK3*I3_s[0]
    integralout = (SigmaParam[itau]/sum(SigmaParam))*integral

    return integralout


def bin_nor_func(x1, x2, x3, x4, K1, K2, K3, alpha1, alpha2, alpha3, SigmaParam, tau_min_edges, tau_max_edges):

    '''
    Computation of the normalization due to secondary (binary) stars.
    '''

    binarity_norm = []
    for i in range(0,7):
        binarity_norm.append(f_toint1_func3_NONP(i, x1, x2, x3, x4, K1, K2, K3, alpha1, alpha2, alpha3, SigmaParam, tau_min_edges, tau_max_edges))

    bin_nor = sum(binarity_norm) + 1.

    return bin_nor


# **********************
# DISTANCE FUNCTION (F5)
# **********************

def dist_metric_gdaf2(H_t, HO):

    '''
    Distance metric to be used in the ABC code. We use the so-called Poissonian distance. The expression can be found in the Equation (58) from Mor et al. 2018.
    '''

    if HO.size==1:
        lrout = np.inf

    else:
        Lrb = [(i*(1.-(j/i)+ np.log(j/i))) if j!=0 and i!=0 else ((i+1)*(1.-((j+1)/(i+1))+ np.log((j+1)/(i+1))))  for i,j in zip(H_t, HO)]

        lrout = np.abs(sum(Lrb))
        if math.isnan(lrout):
            lrout = np.inf

    return lrout
