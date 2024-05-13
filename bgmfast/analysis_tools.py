from bgmfast.bgmfast_simulation_class import bgmfast_simulation
from bgmfast.auxiliary_functions import *
from bgmfast import parameters
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde, pearsonr
from datetime import datetime
import corner


def kde_fit(data, quantiles=[0.16, 0.84], num_points=1000, return_val='mpv_quant'):
    '''
    Estimate the probability density function of a random variable in a non-parametric way by means of a Gaussian kernel
    
    Input parameters
    ----------------
    data : 
    quantiles : list --> the lower and the upper quantiles chosen to represent the uncertainty of the most probable value
    num_points : float or int --> number of points we want to have for the derived probability destribution function 
    return_val : string --> whether we want the most probable value and the quantiles ('mpv_quant'); the most probable value, the quantiles, the independent variable and its estimated probability distribution function ('mpv_quant_dist'); or the median and the quantiles ('median_quant')
    
    Output parameters
    -----------------
    [lower quantile, most probable value, upper quantile] : list --> if return_val=='mpv_quant'
    [lower quantile, most probable value, upper quantile], x, density : list --> if return_val=='mpv_quant_dist'
    [lower quantile, median, upper quantile] : list --> if return_val=='median_quant'
    '''

    if return_val=='mpv_quant':
        kde = gaussian_kde(data)
        x = np.linspace(min(data), max(data), num=num_points)
        density = kde(x)
        most_probable_value = x[np.argmax(density)]
        quant = np.quantile(data, quantiles)
        return [quant[0], most_probable_value, quant[1]]

    elif return_val=='mpv_quant_dist':
        kde = gaussian_kde(data)
        x = np.linspace(min(data), max(data), num=num_points)
        density = kde(x)
        most_probable_value = x[np.argmax(density)]
        quant = np.quantile(data, quantiles)
        return [quant[0], most_probable_value, quant[1]], x, density

    elif return_val=='median_quant':
        quant = np.quantile(data, quantiles)
        return [quant[0], np.median(data), quant[1]]

    else:
        print('No valid value of return_val [mpv_quant/median_quant]')
        sys.exit


class output_file_analysis:
    
    '''
    Analyze the results arising from BGM FASt + ABC
    '''
    
    def __init__(self, bgmfast_output_file, params_keypos, num_acc_sim, quantiles=[0.16, 0.84]):
        
        '''
        Import the output file arising from BGM FASt + ABC
        
        Input parameters
        ----------------
        bgmfast_output_file : str --> BGM FASt + ABC output file name
        num_acc_sim : int --> number of accepted simulations per step 
        quantiles : list --> the lower and the upper quantiles chosen to represent the uncertainty of the most probable value
        '''
        
        self.datetimes = []
        self.distances = []
        self.data = {key: [] for key, value in params_keypos.items() if value[1]!='datetime' and value[1]!='distance'}
        self.data_no_corr = {key: [] for key, value in params_keypos.items() if value[1]!='datetime' and value[1]!='distance'}
        
        datafile = open(bgmfast_output_file, 'r')
        
        counter = 0
        for line in datafile.readlines():
            counter += 1
            if counter==1:
                self.header = line.split('\t')
                self.num_columns = len(self.header)
                continue
            
            if line[0]=='':
                break
            
            dataline = line.split('\t')[:-1]
            for key, value in params_keypos.items():
                pos, corr_factor = value
                if corr_factor=='datetime':
                    self.datetimes.append(datetime.fromisoformat(dataline[pos][:-1]))
                elif corr_factor=='distance':
                    self.distances.append(float(dataline[pos]))
                else:
                    self.data[key].append(float(dataline[pos])*corr_factor)
                    self.data_no_corr[key].append(float(dataline[pos]))
                
        self.params_keypos = params_keypos
        self.num_datapoints = len(self.data[list(self.data.keys())[0]])
        self.num_acc_sim = num_acc_sim
        self.num_steps = int(self.num_datapoints/self.num_acc_sim)
        self.quantiles = quantiles
        self.quantiles_median = [quantiles[0], 0.5, quantiles[1]]
        
        print('\nNumber of data points:', self.num_datapoints)
        print('Number of accepted simulations per step:', self.num_acc_sim)
        print('Number of steps:', self.num_steps)
    
    
def final_params(data, num_acc_sim, quantiles=[0.16, 0.84], return_val='mpv_quant', step_range=False, show=False):
    
    '''
    Obtain the most probable value of the parameters derived with BGM FASt + ABC
    
    Input parameters
    ----------------
    return_val : string --> whether we want the most probable value and the quantiles ('mpv_quant'); the most probable value, the quantiles, the independent variable and its estimated probability distribution function ('mpv_quant_dist'); or the median and the quantiles ('median_quant')
    show : bool --> whether we want to show the final parameters or not 
    step_range : list --> range of steps you want to take into account to compute the final parameters. Example: [60, 100]
    
    Output parameters
    -----------------
    self.final_parameters : dict --> dictionary with the values of the parameters derived with BGM FASt + ABC
    '''
    
    final_parameters = {}
    for param in data.keys():
        if step_range!=False:
            final_parameters[param] = kde_fit(data[param][step_range[0]*num_acc_sim:step_range[1]*num_acc_sim], quantiles=quantiles, return_val=return_val)
        else:
            final_parameters[param] = kde_fit(data[param], quantiles=quantiles, return_val=return_val)
    
    if show:
        print('\nFinal parameters [{:.2f} quantile, most probable value, {:.2f} quantile]:'.format(quantiles[0], quantiles[1]))
        for param in data.keys():
            q1, mpv, q2 = final_parameters[param]
            print('\t%s:' %param, '[{:.3f}, {:.3f}, {:.3f}]'.format(q1, mpv, q2))
            
    return final_parameters
    
    
def up_to_step(data, step, num_acc_sim, quantiles=[0.16, 0.84], return_val='mpv_quant'):
    
    '''
    Return the most probable values and the quantiles of the BGM FASt + ABC parameters computed taking into account all steps up to the given one
    
    Input parameters
    ----------------
    step : int --> step up to which you want to compute the results
    return_val : string --> whether we want the most probable value and the quantiles ('mpv_quant'); the most probable value, the quantiles, the independent variable and its estimated probability distribution function ('mpv_quant_dist'); or the median and the quantiles ('median_quant')
    
    Output parameters
    -----------------
    self.step_data : dict --> dictionary with all the values of the parameters in the different steps
    self.step_parameters : dict --> dictionary with the values of the parameters derived with BGM FASt + ABC
    '''
    
    step_data = {}
    for param in data.keys():
        step_data[param] = data[param][0:(step + 1)*num_acc_sim]
        
    step_parameters = {}
    for param in data.keys():
        step_parameters[param] = kde_fit(step_data[param], quantiles=quantiles, return_val=return_val)
        
    return step_data, step_parameters
    
    
def single_step(data, step, num_acc_sim, quantiles=[0.16, 0.84], return_val='median_quant'):
    
    '''
    Return the value of the BGM FASt + ABC parameters taking into account only the given step
    
    Input parameters
    ----------------
    step : int --> step for which we want to compute the results
    return_val : string --> whether we want the most probable value and the quantiles ('mpv_quant'); the most probable value, the quantiles, the independent variable and its estimated probability distribution function ('mpv_quant_dist'); or the median and the quantiles ('median_quant')
    
    Output parameters
    -----------------
    self.single_step_data : dict --> dictionary with all the values of the parameters at the given step
    self.single_step_parameters : dict --> dictionary with the values of the parameters derived with BGM FASt + ABC
    '''
    
    single_step_data = {}
    for param in data.keys():
        single_step_data[param] = data[param][step*num_acc_sim:(step + 1)*num_acc_sim]
    
    single_step_parameters = {}
    for param in data.keys():
        single_step_parameters[param] = kde_fit(single_step_data[param], quantiles=quantiles, return_val=return_val)
        
    return single_step_data, single_step_parameters
    
    
def build_sfh(tau_values, sfh_params, tau_ranges=parameters.general_parameters['tau_ranges'].value, output=False, show=False, ms_sfh=True, prior_sfh_std=2, step=False, axis=False, return_val='mpv_quant', limits=[0, 13]):
    
    '''
    Build the star formation history plot 
    
    Input parameters
    ----------------
    output : str or bool --> directory of the output plot (set to False to avoid saving the plot)
    show : bool --> whether we want to show the plot or not
    ms_sfh : bool --> whether we want to plot the Mother Simulation SFH or not
    prior_sfh_means : str or list or bool --> values of the priors of the SFH (set to False to avoid showing the priors and to 'ms_values' to use the MS values as priors)
    prior_sfh_std : float or int --> standard deviation of the priors of the SFH
    step : int --> step for which we want to compute the SFH
    axis : matplotlib.axes._axes.Axes --> axis that we want to use to build GIF plots showing the evolution of the SFH
    return_val : str --> Whether we want the most probable value and the quantiles ('mpv_quant'); the most probable value, the quantiles, the independent variable and its estimated probability distribution function ('mpv_quant_dist'); or the median and the quantiles ('median_quant')
    limits : list --> lower and upper limits of the surface density in the SFH plot (Y axis)
    '''
    
    if not step:
        print('\nBuilding SFH...\n')
    
    thick = False
    if isinstance(tau_values, list):
        tau_values, T_tau_values = tau_values
        sfh_params, T_sfh_params = sfh_params
        thick = True

    sfh = [i[1]for i in sfh_params]
    sfh_quant1 = [i[1] - i[0] for i in sfh_params]
    sfh_quant2 = [i[2] - i[1] for i in sfh_params]
    sfh_quantiles = np.array([sfh_quant1, sfh_quant2])
    if thick:
        T_sfh = [i[1]for i in T_sfh_params]
        T_sfh_quant1 = [i[1] - i[0] for i in T_sfh_params]
        T_sfh_quant2 = [i[2] - i[1] for i in T_sfh_params]
        T_sfh_quantiles = np.array([T_sfh_quant1, T_sfh_quant2])
    
    ms_parameters = parameters.ms_parameters
    SigmaParam_ms = ms_parameters['SigmaParam_ms'].value
    SigmaParam_ms[0] = SigmaParam_ms[0]
    SigmaParam_ms[1] = SigmaParam_ms[1]
    midpopbin_ms = ms_parameters['midpopbin_ms'].value
    lastpopbin_ms = ms_parameters['lastpopbin_ms'].value
    ms_params_sfh = np.append(SigmaParam_ms[:4], midpopbin_ms)
    ms_params_sfh = np.append(ms_params_sfh, lastpopbin_ms)
    tau_ranges_mod = [[sub[0], sub[1]] for sublist in tau_ranges for sub in sublist]
    ms_params_sfh = [i/(tau_range[1] - tau_range[0]) for i, tau_range in zip(ms_params_sfh, tau_ranges_mod)]
    if thick:
        T_ms_params_sfh = ms_parameters['T_SigmaParam_ms'].value
            
    if not step:
        fig, ax = plt.subplots()
    else:
        ax = axis
        ax.clear()
        ax.set_title('Step ' + str(step), fontsize=15)
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 12)
    
    ax.errorbar(tau_values, sfh, yerr=sfh_quantiles, fmt='o--', capsize=5, label='BGM FASt thin', color='blue', zorder=3)
    if thick:
        ax.errorbar(T_tau_values, T_sfh, yerr=T_sfh_quantiles, fmt='x--', capsize=5, label='BGM FASt thick', color='blue', zorder=3)
    
    if ms_sfh==True:
        ax.errorbar(tau_values, ms_params_sfh, fmt='o--', capsize=5, label='MS thin', color='green', zorder=2)
        if thick:
            ax.errorbar(T_tau_values, T_ms_params_sfh, fmt='x--', capsize=5, label='MS thick', color='green', zorder=2)
        
    ax.invert_xaxis()
    ax.set_ylim(limits[0], limits[1])
    ax.set_xlabel('Age (Gyr)')
    ax.set_ylabel(r'Mean SFH per age bin ($M_\odot/$pc$^2/$Gyr)')
    ax.legend()
    
    if output!=False:
        fig.savefig(output, dpi=300)
    if show:
        plt.show()
    plt.close()

    
def build_imf(imf_ranges, imf_params, output=False, show=False, ms_imf=True, prior_imf_std=2, step=False, axis=False, return_val='mpv_quant', limits=[-1, 3]):
    
    '''
    Build the initial mass function slopes plot 
    
    Input parameters
    ----------------
    output : str or bool --> directory of the output plot (set to False to avoid saving the plot)
    show : bool --> whether we want to show the plot or not
    ms_imf : bool --> whether we want to plot the Mother Simulation IMF or not
    prior_imf_means : str or list or bool --> values of the priors of the IMF (set to False to avoid showing the priors and to 'ms_values' to use the MS values as priors)
    prior_imf_std : float or int --> standard deviation of the priors of the IMF
    step : int --> step for which we want to compute the IMF
    axis : matplotlib.axes._axes.Axes --> axis that we want to use to build GIF plots showing the evolution of the IMF
    return_val : str --> whether we want the most probable value and the quantiles ('mpv_quant'); the most probable value, the quantiles, the independent variable and its estimated probability distribution function ('mpv_quant_dist'); or the median and the quantiles ('median_quant') 
    limits : list --> lower and upper limits of the surface density in the IMF plot (Y axis)
    
    Note: this function is only considered for plotting the three slopes of the IMF or two of them (alpha2 and alpha3)
    '''
    
    if not step:
        print('\nBuilding IMF...\n')
    
    imf = [i[1] for i in imf_params]
    imf_quant1 = [i[0] for i in imf_params]
    imf_quant2 = [i[2] for i in imf_params]
    
    ms_parameters = parameters.ms_parameters
    alpha1_ms = ms_parameters['alpha1_ms'].value
    alpha2_ms = ms_parameters['alpha2_ms'].value
    alpha3_ms = ms_parameters['alpha3_ms'].value
    ms_params_imf = [alpha1_ms, alpha2_ms, alpha3_ms]
    
    if not step:
        fig, ax = plt.subplots()
    else:
        ax = axis
        ax.clear()
        ax.set_title('Step ' + str(step), fontsize=15)
        
    for i in range(len(imf)):
        if i==0:
            x = np.linspace(imf_ranges[i], imf_ranges[i+1])
            ax.plot(x, np.full_like(x, imf[i]), label='BGM FASt', linestyle='-', color='blue', zorder=3)
            ax.fill_between(x, imf_quant1[i], imf_quant2[i], color='yellow', alpha=0.3)
            if ms_imf==True:
                ax.plot(x, np.full_like(x, ms_params_imf[i]), label='MS', linestyle='--', color='green', zorder=2)
            ax.axvline(imf_ranges[i+1], alpha=0.5, linestyle='--', color='black', linewidth=1)
        else:
            x = np.linspace(imf_ranges[i], imf_ranges[i+1])
            ax.plot(x, np.full_like(x, imf[i]), linestyle='-', color='blue')
            ax.fill_between(x, imf_quant1[i], imf_quant2[i], color='yellow', alpha=0.3)
            if ms_imf==True:
                ax.plot(x, np.full_like(x, ms_params_imf[i]), linestyle='--', color='green', zorder=2)
            ax.axvline(imf_ranges[i+1], alpha=0.5, linestyle='--', color='black', linewidth=1)
    ax.set_xscale('log')
    ax.set_xlim(imf_ranges[0], imf_ranges[-1])
    ax.set_ylim(limits[0], limits[1])
    ax.set_xlabel(r'Stellar mass ($M/M_\odot$)')
    ax.set_ylabel(r'IMF slope $\alpha$')
    ax.legend(loc=4)
    
    if output!=False:
        fig.savefig(output, dpi=300)
    if show:
        plt.show()
    plt.close()
            
            
def build_real_imf(imf_ranges, imf_params, output=False, show=False, ms_imf=True, prior_imf_std=2, step=False, axis=False, return_val='mpv_quant'):
    
    '''
    Build the real initial mass function plot
    
    Input parameters
    ----------------
    output : str or bool --> directory of the output plot (set to False to avoid saving the plot)
    show : bool --> whether we want to show the plot or not
    ms_imf : bool --> whether we want to plot the Mother Simulation IMF or not
    prior_imf_means : str or list or bool --> values of the priors of the IMF (set to False to avoid showing the priors and to 'ms_values' to use the MS values as priors)
    prior_imf_std : float or int --> standard deviation of the priors of the IMF
    step : int --> step for which we want to compute the IMF
    axis : matplotlib.axes._axes.Axes --> axis that we want to use to build GIF plots showing the evolution of the IMF
    return_val : str --> whether we want the most probable value and the quantiles ('mpv_quant'); the most probable value, the quantiles, the independent variable and its estimated probability distribution function ('mpv_quant_dist'); or the median and the quantiles ('median_quant') 
    limits : list --> lower and upper limits of the surface density in the IMF plot (Y axis)
    
    Note: this function is only considered for plotting the IMF with the three slopes
    '''
    
    if not step:
        print('\nBuilding real IMF...\n')
        
    imf = [i[1] for i in imf_params]
    
    ms_parameters = parameters.ms_parameters
    alpha1_ms = ms_parameters['alpha1_ms'].value
    alpha2_ms = ms_parameters['alpha2_ms'].value
    alpha3_ms = ms_parameters['alpha3_ms'].value
    ms_params_imf = [alpha1_ms, alpha2_ms, alpha3_ms]
    
    K = Continuity_Coeficients_func(imf[0], imf[1], imf[2], imf_ranges[0], imf_ranges[1], imf_ranges[2], imf_ranges[3])
    K_ms = Continuity_Coeficients_func(ms_params_imf[0], ms_params_imf[1], ms_params_imf[2], imf_ranges[0], imf_ranges[1], imf_ranges[2], imf_ranges[3])
    
    if not step:
        fig, ax = plt.subplots()
    else:
        ax = axis
        ax.clear()
        ax.title('Step ' + str(step), fontsize=15)
        
    for i in range(len(imf)):
        if i==0:
            x = np.linspace(imf_ranges[i], imf_ranges[i+1])
            fi = lambda m: K[i]*m**(-imf[i])
            real_imf = [fi(m) for m in x]
            ax.plot(x, real_imf, label='BGM FASt', linestyle='-', color='blue')
            if ms_imf==True:
                fi = lambda m: K_ms[i]*m**(-ms_params_imf[i])
                real_imf_ms = [fi(m) for m in x]
                ax.plot(x, real_imf_ms, label='MS', linestyle='--', color='green', zorder=2)
            ax.axvline(imf_ranges[i+1], alpha=0.5, linestyle='--', color='black', linewidth=1)
        else:
            x = np.linspace(imf_ranges[i], imf_ranges[i+1])
            fi = lambda m: K[i]*m**(-imf[i])
            real_imf = [fi(m) for m in x]
            ax.plot(x, real_imf, linestyle='-', color='blue')
            if ms_imf==True:
                fi = lambda m: K_ms[i]*m**(-ms_params_imf[i])
                real_imf_ms = [fi(m) for m in x]
                ax.plot(x, real_imf_ms, linestyle='--', color='green', zorder=2)
            ax.axvline(imf_ranges[i+1], alpha=0.5, linestyle='--', color='black', linewidth=1)
            
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(imf_ranges[0], imf_ranges[-1])
    ax.set_xlabel(r'Stellar mass ($M/M_\odot$)')
    ax.set_ylabel(r'IMF')
    ax.legend(loc=4)

    if output!=False:
        fig.savefig(output, dpi=300)
    if show:
        plt.show()
    plt.close()
            
    
def distance_evolution(distances, num_acc_sim, output=False, show=False):
    
    '''
    Plot the evolution of the minimum, mean and maximum distance at each step
    
    Input parameters
    ----------------
    output : str or bool --> directory of the output plot (set to False to avoid saving the plot)
    show : bool --> whether we want to show the plot or not
    '''

    distances_max = []
    distances_mean = []
    distances_min = []
    steps = 0
    distance_step = []
    counter = 1
    for distance in distances:
        if counter%num_acc_sim==0:
            distance_step.append(distance)
            distances_max.append(max(distance_step))
            distances_mean.append(np.mean(distance_step))
            distances_min.append(min(distance_step))
            steps += 1
            distance_step = []
            counter = 1
        else:
            distance_step.append(distance)
            counter += 1

    fig, ax = plt.subplots()
    ax.scatter(range(steps), distances_max, marker='o', s=5, label='Maximum distance')
    ax.scatter(range(steps), distances_mean, marker='o', s=5, label='Mean distance')
    ax.scatter(range(steps), distances_min, marker='o', s=5, label='Minimum distance')
    ax.legend()
    ax.set_xlabel('Steps')
    ax.set_ylabel('Distance')

    if output!=False:
        fig.savefig(output, dpi=300)
    if show:
        plt.show()
    plt.close()
        
    
def parameter_evolution(param_name, param_data, num_acc_sim, output=False, show=False, plot_single_step={'median': True, 'median_quant': True, 'mpv': False, 'mpv_quant': False, 'min': False, 'max': False}, plot_acc={'median': False, 'median_quant': False, 'mpv': True, 'mpv_quant': True, 'min': False, 'max': False}):
    
    '''
    Plot the evolution of a BGM FASt parameter
    
    Input parameters
    ----------------
    index : int --> index of the parameter in the header (only taking into account the BGM FASt parameters)
    output : str or bool --> directory of the output plot (set to False to avoid saving the plot)
    show : bool --> whether we want to show the plot or not 
    param_name : str --> name of the parameter to show in the plot
    plot_single_step : dict --> whether we want to show the median, the quantiles with the median, the most probable value (mpv), the quantiles with the mpv, the minimum, and/or the maximum of each step.
    plot_acc : whether we want to show the median, the quantiles with the median, the most probable value (mpv), the quantiles with the mpv, the minimum, and/or the maximum accumulated up to each step.
    '''

    print('\nBuilding %s evolution...\n' %param_name)
    
    num_steps = int(len(param_data)/num_acc_sim)
    param_data = {param_name: param_data}

    fig, ax = plt.subplots()

    if plot_single_step['median']==True or plot_single_step['median_quant']==True:
        single_parameters = []
        single_mins = []
        single_maxs = []
        for step in range(num_steps):
            single_step_data, single_step_params = single_step(param_data, step, num_acc_sim, return_val='median_quant')
            single_parameters.append(single_step_params[param_name])
            single_mins.append(min(single_step_data[param_name]))
            single_maxs.append(max(single_step_data[param_name]))

        single_medians = [i[1] for i in single_parameters]

        if plot_single_step['median_quant']==True:
            single_quant1 = [i[1] - i[0] for i in single_parameters]
            single_quant2 = [i[2] - i[1] for i in single_parameters]
            single_quantiles = np.array([single_quant1, single_quant2])
            ax.errorbar(range(num_steps), single_medians, yerr=single_quantiles, fmt='o--', capsize=5, label='Single step (median)')

        elif plot_single_step['median_quant']==False:
            ax.scatter(range(num_steps), single_medians, marker='o', s=5, label='Single step (median)')

        else:
            print('plot_single_step has to be a boolean [True/False]')
            sys.exit()

    if plot_single_step['mpv']==True or plot_single_step['mpv_quant']==True:
        single_parameters = []
        single_mins = []
        single_maxs = []
        for step in range(num_steps):
            single_step_data, single_step_params = single_step(param_data, step, num_acc_sim, return_val='mpv_quant')
            single_parameters.append(single_step_params[param_name])
            single_mins.append(min(single_step_data[param_name]))
            single_maxs.append(max(single_step_data[param_name]))

        single_mpv = [i[1] for i in single_parameters]

        if plot_single_step['mpv_quant']==True:
            single_quant1 = [i[1] - i[0] for i in single_parameters]
            single_quant2 = [i[2] - i[1] for i in single_parameters]
            single_quantiles = np.array([single_quant1, single_quant2])
            ax.errorbar(range(num_steps), single_mpv, yerr=single_quantiles, fmt='o--', capsize=5, label='Single step (mpv)')

        elif plot_single_step['mpv_quant']==False:
            ax.scatter(range(num_steps), single_mpv, marker='o', s=5, label='Single step (mpv)')

        else:
            print('plot_single_step has to be a boolean [True/False]')
            sys.exit()

    if plot_single_step['min']==True:
        ax.scatter(range(num_steps), single_mins, marker='o', s=5, label='Single step (min)')
    if plot_single_step['max']==True:
        ax.scatter(range(num_steps), single_maxs, marker='o', s=5, label='Single step (max)')

    if plot_acc['median']==True or plot_acc['median_quant']==True:
        aparameters = []
        acc_mins = []
        acc_maxs = []
        for step in range(num_steps):
            step_data, step_params = up_to_step(param_data, step, num_acc_sim, return_val='median_quant')
            aparameters.append(step_params[param_name])
            acc_mins.append(min(step_data[param_name]))
            acc_maxs.append(max(step_data[param_name]))

        acc_medians = [i[1] for i in aparameters]

        if plot_acc['median_quant']==True:
            acc_quant1 = [i[1] - i[0] for i in aparameters]
            acc_quant2 = [i[2] - i[1] for i in aparameters]
            acc_quantiles = np.array([single_quant1, single_quant2])
            ax.errorbar(range(num_steps), acc_medians, yerr=acc_quantiles, fmt='o--', capsize=5, label='Accumulated (median)')

        elif plot_acc['median_quant']==False:
            ax.scatter(range(num_steps), acc_medians, marker='o', s=5, label='Accumulated (median)')

        else:
            print('plot_single_step has to be a boolean [True/False]')
            sys.exit()

    if plot_acc['mpv']==True or plot_acc['mpv_quant']==True:
        aparameters = []
        acc_mins = []
        acc_maxs = []
        for step in range(num_steps):

            step_data, step_params = up_to_step(param_data, step, num_acc_sim, return_val='mpv_quant')
            aparameters.append(step_params[param_name])
            acc_mins.append(min(step_data[param_name]))
            acc_maxs.append(max(step_data[param_name]))

        acc_mpv = [i[1] for i in aparameters]

        if plot_acc['mpv_quant']==True:
            acc_quant1 = [i[1] - i[0] for i in aparameters]
            acc_quant2 = [i[2] - i[1] for i in aparameters]
            acc_quantiles = np.array([single_quant1, single_quant2])
            ax.errorbar(range(num_steps), acc_mpv, yerr=acc_quantiles, fmt='o--', capsize=5, label='Accumulated (mpv)')

        elif plot_acc['mpv_quant']==False:
            ax.scatter(range(num_steps), acc_mpv, marker='o', s=5, label='Accumulated (mpv)')

        else:
            print('plot_single_step has to be a boolean [True/False]')
            sys.exit()

    if plot_acc['min']==True:
        ax.scatter(range(num_steps), acc_mins, marker='o', s=5, label='Accumulated (min)')
    if plot_acc['max']==True:
        ax.scatter(range(num_steps), acc_maxs, marker='o', s=5, label='Accumulated (max)')

    ax.set_xlabel('Steps')
    ax.legend()
    ax.set_ylabel(param_name)

    if output!=False:
        fig.savefig(output, dpi=300)
    if show:
        plt.show()
    plt.close()
            
        
def cornerplot(data, ranges, num_acc_sim, tau_ranges=parameters.general_parameters['tau_ranges'].value, num_decimals=2, dpi=300, output=False, show=False):
    
    '''
    Build the cornerplot of the BGM FASt parameters
    
    Input parameters
    ----------------
    output
    show
    ranges : list --> list with tuples with the ranges for the different parameters
    corner_num_params : int --> number of parameters you want to show in the cornerplot
    label : list --> name of the parameters for the plot
    num_decimals : int --> number of decimals you want to show in the plot
    dpi : int or float --> resolution of the plot
    '''
    
    print('\nBuilding cornerplot...')
    
    num_params = len(data.keys())
    num_datapoints = len(data[list(data.keys())[0]])
    samples = []
    for datapoint in range(num_datapoints):
        for param in data.keys(): 
            samples.append(data[param][datapoint])
    xy_sample = np.array(samples).reshape(-1, num_params).T.tolist()
    samples = np.array(samples).reshape([num_datapoints, num_params])
    
    if ranges=='auto':
        figure = corner.corner(samples, labels=list(data.keys()), label_kwargs={"fontsize": 8}, show_titles=True, color='darkgreen', verbose=True, title_kwargs={"fontsize": 5}, title_fmt="0.3f", hist_kwargs={"density": True, "linewidth":1})
    else:
        figure = corner.corner(samples, range=ranges, labels=list(data.keys()), label_kwargs={"fontsize": 8}, show_titles=True, color='darkgreen', verbose=True, title_kwargs={"fontsize": 5}, title_fmt="0.3f", hist_kwargs={"density": True, "linewidth":1})
        
    for ax in figure.get_axes():
        ax.tick_params(axis='both', labelsize=5)
        
    axes = np.array(figure.axes).reshape((num_params, num_params))
    params_mpv = final_params(data, num_acc_sim, return_val='mpv_quant_dist')
    params_quant = final_params(data, num_acc_sim, return_val='median_quant')
    
    ms_parameters = parameters.ms_parameters
    tau_ranges_mod = [[sub[0], sub[1]] for sublist in tau_ranges for sub in sublist]
    ms_params_sfh = [ms_parameters['SigmaParam_ms'].value[0],
                     ms_parameters['SigmaParam_ms'].value[1],
                     ms_parameters['SigmaParam_ms'].value[2], 
                     ms_parameters['SigmaParam_ms'].value[3], 
                     ms_parameters['midpopbin_ms'].value[0], 
                     ms_parameters['midpopbin_ms'].value[1], 
                     ms_parameters['midpopbin_ms'].value[2], 
                     ms_parameters['midpopbin_ms'].value[3], 
                     ms_parameters['lastpopbin_ms'].value[0], 
                     ms_parameters['lastpopbin_ms'].value[1], 
                     ms_parameters['lastpopbin_ms'].value[2]]                     
    ms_params_sfh = [i/(tau_range[1] - tau_range[0]) for i, tau_range in zip(ms_params_sfh, tau_ranges_mod)]
    
    all_ms_values = {'alpha1': ms_parameters['alpha1_ms'].value, 
                'alpha2': ms_parameters['alpha2_ms'].value, 
                'alpha3': ms_parameters['alpha3_ms'].value, 
                'sfh0': ms_params_sfh[0], 
                'sfh1': ms_params_sfh[1], 
                'sfh2': ms_params_sfh[2],
                'sfh3': ms_params_sfh[3], 
                'sfh4': ms_params_sfh[4], 
                'sfh5': ms_params_sfh[5], 
                'sfh6': ms_params_sfh[6], 
                'sfh7': ms_params_sfh[7],
                'sfh8': ms_params_sfh[8],
                'sfh9': ms_params_sfh[9],
                'sfh10': ms_params_sfh[10], 
                'sfh9T': ms_parameters['T_SigmaParam_ms'].value[0], 
                'sfh10T': ms_parameters['T_SigmaParam_ms'].value[1], 
                'sfh11T': ms_parameters['T_SigmaParam_ms'].value[2], 
                'sfh12T': ms_parameters['T_SigmaParam_ms'].value[3]}
    ms_values = [all_ms_values[key] for key in data.keys()]
    
    mpv_values = []
    for i, param_name in zip(range(num_params), data.keys()):
        ax = axes[i, i]
        #ax.hist(self.samples[:,i], color="orange", histtype='step', range=ranges[i], bins=20, lw=2, normed=True)
        for quant in params_quant[param_name]:
            ax.axvline(quant, color="black", linestyle="dashed", alpha=0.7, lw=1.0)
        ax.axvline(params_mpv[param_name][0][1], color="r", lw=1.0)
        ax.plot(params_mpv[param_name][1], params_mpv[param_name][2], color='blue')

        mpv = params_mpv[param_name][0][1]
        mpv_values.append(mpv)
        up_error = params_mpv[param_name][0][2] - mpv
        down_error = mpv - params_mpv[param_name][0][0]

        #OJO! A millorar el rounded
        mpv_rounded = np.around(mpv, num_decimals)
        up_error_rounded = np.around(up_error, num_decimals)
        down_error_rounded = np.around(down_error, num_decimals)

        title = list(data.keys())[i] + "$ = {}^{{+{}}}_{{-{}}}$".format(mpv_rounded, up_error_rounded, down_error_rounded)

        ax.set_title(title, fontsize=8)

        for j in range(i):
            ax2 = axes[i, j]
            ax2.scatter(ms_values[j], ms_values[i], marker='x', s=30, color='magenta', lw=3)
            ax2.scatter(mpv_values[j], mpv_values[i], marker='*', s=30, color='orange', lw=3)
            coef, pvalue = pearsonr(xy_sample[i], xy_sample[j])
            ax2.text(0.82, 0.88, str(round(float(coef), 2)), ha='center', va='center', transform=ax2.transAxes, color="black", fontweight='bold', fontsize=6)
            
    if output!=False:
        figure.savefig(output, dpi=dpi)
    if show:
        plt.show()
    plt.close()


class compare_hess_diagrams:
    
    '''
    Generate and compare Hess diagrams from different catalogs
    '''

    def __init__(self):
        '''
        Initialize the compare_hess_diagrams class
        '''

        self.bgmfast_sim = bgmfast_simulation()
        self.bgmfast_sim.open_spark_session()

        self.bgmfast_sim.set_acc_parameters()
        self.bgmfast_sim.set_binning_parameters()
        self.bgmfast_sim.set_general_parameters()
        self.bgmfast_sim.set_ms_parameters()
        self.bgmfast_sim.set_ps_parameters()
        self.bgmfast_sim.set_constraints_parameters()
        self.bgmfast_sim.set_bgmfast_parameters()

        pass


    def generate_catalog_hess_diagram(self, filename_catalog, colnames=parameters.catalog_file_parameters['sel_columns_catalog'].value, Gmax=parameters.catalog_file_parameters['Gmax_catalog'].value):
        '''
        Generate catalog Hess diagram
        
        Input parameters 
        ----------------
        filename_catalog : str --> directory of the catalog file
        colnames : list --> list with the name of the columns in the catalog with the following order: G, G-Rp, longitude, latitude, M_G' and parallax 
        Gmax : int or float --> limitting magnitude
        
        Output parameters
        -----------------
        catalog_cmd : numpy array --> 4-dimensional numpy array (Hess diagram + latitude + longitude) containing the complete Hess diagram only showing those points with a number of stars larger than the minimum
        catalog_data : numpy array --> 4-dimensional numpy array with the Hess diagrams corresponding to each one of the longitude and latitude ranges
        '''
        
        self.bgmfast_sim.read_catalog(filename_catalog, colnames, Gmax)
        self.bgmfast_sim.generate_catalog_cmd()
        catalog_cmd_orig = self.bgmfast_sim.return_cmd()[0]
        catalog_data = self.bgmfast_sim.return_cmd()[1]
        
        dist_threshold = parameters.distance_parameters['dist_thresh'].value
        
        catalog_cmd = np.array([[[[0 if catalog_cmd_orig[lon][lat][i][j]<dist_threshold else catalog_cmd_orig[lon][lat][i][j] for j in range(len(catalog_cmd_orig[lon][lat][i]))] for i in range(len(catalog_cmd_orig[lon][lat]))] for lat in range(len(catalog_cmd_orig[lon]))] for lon in range(len(catalog_cmd_orig))])
        
        return catalog_cmd, catalog_data


    def generate_difference_hess_diagram(self, catalog1_cmd, catalog2_cmd):
        '''
        Crossmatch two catalogs and analyze differences
        
        Input parameters
        ----------------
        catalog1_cmd : numpy array --> 4-dimensional numpy array (Hess diagram + latitude + longitude) containing the complete Hess diagram of the first catalog
        catalog2_cmd : numpy array --> 4-dimensional numpy array (Hess diagram + latitude + longitude) containing the complete Hess diagram of the second catalog
        Note: when comparing simulations and observed data, catalog 1 refers to the simulation and catalog 2 to the data according to Eq. (58) from Mor et al. 2018 for the computation of the distance. 
        
        Output parameters
        -----------------
        distance_cmd : numpy array --> 4-dimensional numpy array (Hess diagram + latitude + longitude) containing the distance per bin of the complete Hess diagrams between catalogs
        difference_cmd : numpy array --> 4-dimensional numpy array (Hess diagram + latitude + longitude) containing the absolute difference in number of stars per bin of the complete Hess diagrams between catalogs
        quocient_cmd : numpy array --> 4-dimensional numpy array (Hess diagram + latitude + longitude) containing the quocient of the number of stars per bin of the complete Hess diagrams between catalogs
        
        '''
        
        quocient_cmd = np.array([[[[compute_bin_quotient(catalog2_cmd[lon][lat][i][j], catalog1_cmd[lon][lat][i][j]) for j in range(len(catalog1_cmd[lon][lat][i]))] for i in range(len(catalog1_cmd[lon][lat]))] for lat in range(len(catalog1_cmd[lon]))] for lon in range(len(catalog1_cmd))])
        
        distance_cmd = np.array([[[[compute_bin_distance(catalog2_cmd[lon][lat][i][j], catalog1_cmd[lon][lat][i][j]) for j in range(len(quocient_cmd[lon][lat][i]))] for i in range(len(quocient_cmd[lon][lat]))] for lat in range(len(quocient_cmd[lon]))] for lon in range(len(quocient_cmd))])
        
        difference_cmd = catalog1_cmd - catalog2_cmd
        
        return distance_cmd, difference_cmd, quocient_cmd
    
    
    def build_hess_diagrams_plots(self, catalog1_cmd, catalog2_cmd, distance_cmd, difference_cmd, output=False, show=True, titles=['Catalog 1', 'Catalog 2', r'$\delta_P$(Catalog 1, Catalog 2)', 'Catalog 1 - Catalog 2'], limits='auto'):
        '''
        Build the Hess diagrams of two catalogs and their differences
        
        Input parameters
        ----------------
        catalog1_cmd : numpy array --> 4-dimensional numpy array (Hess diagram + latitude + longitude) containing the complete Hess diagram of the first catalog
        catalog2_cmd : numpy array --> 4-dimensional numpy array (Hess diagram + latitude + longitude) containing the complete Hess diagram of the second catalog
        Note: when comparing simulations and observed data, catalog 1 refers to the simulation and catalog 2 to the data according to Eq. (58) from Mor et al. 2018 for the computation of the distance
        distance_cmd : numpy array --> 4-dimensional numpy array (Hess diagram + latitude + longitude) containing the distance per bin of the complete Hess diagrams between catalogs
        difference_cmd : numpy array --> 4-dimensional numpy array (Hess diagram + latitude + longitude) containing the absolute difference in number of stars per bin of the complete Hess diagrams between catalogs
        output : str or False --> directory of the output file of the plot
        show : boolean --> wether you want the plot to be displayed or not
        titles : list --> titles of the four different columns of the plot
        limits : list or 'auto' --> upper limits of the plots in each one of the four colour bars or set to 'auto'
        '''

        acc_parameters = parameters.acc_parameters
        nLonbins = acc_parameters['nLonbins'].value
        nLatbins = acc_parameters['nLatbins'].value
        
        dist_threshold = parameters.distance_parameters['dist_thresh'].value

        binning_parameters = parameters.binning_parameters
        Xmin = binning_parameters['Xmin'].value
        Xmax = binning_parameters['Xmax'].value
        Ymin = binning_parameters['Ymin'].value
        Ymax = binning_parameters['Ymax'].value

        extent = [Xmin, Xmax, Ymax, Ymin]

        fig = plt.figure(figsize=(22, 9))
        fig.tight_layout()
        axs = fig.subplots(nLatbins, len(titles), gridspec_kw={'width_ratios': [1, 1, 1.25, 1.25]})

        if limits=='auto':
            limits_hess = max(np.array(catalog1_cmd).max(), np.array(catalog2_cmd).max())
            limits_dist = max([abs(np.quantile(distance_cmd, 0.01)), abs(np.quantile(distance_cmd, 0.99))])
            limits_diff = max([abs(np.quantile(difference_cmd, 0.01)), abs(np.quantile(difference_cmd, 0.99))])
        else:
            if limits[0]=='auto':
                limits_hess = max(np.array(catalog1_cmd).max(), np.array(catalog2_cmd).max())
            else:
                limits_hess = limits[0]
            if limits[1]=='auto':
                limits_dist = max([abs(np.quantile(distance_cmd, 0.01)), abs(np.quantile(distance_cmd, 0.99))])
            else:
                limits_dist = limits[1]
            if limits[2]=='auto':
                limits_diff = max([abs(np.quantile(difference_cmd, 0.01)), abs(np.quantile(difference_cmd, 0.99))])
            else:
                limits_diff = limits[2] 

        for lon in range(nLonbins):
            for lat in range(nLatbins):
                for col in range(len(titles)):

                    if lat==0:
                        axs[lat, col].set_title(titles[col])

                    if col==0:
                        axs[lat, col].set_ylabel(r"$M_G'$")

                    if (lat + 1)==nLatbins:
                        axs[lat, col].set_xlabel("$G-Rp$")

                    cmap = plt.cm.jet
                    cmap.set_bad(color="white")

                    cmap2 = plt.cm.get_cmap('gist_stern').reversed()
                    #cmap2 = plt.cm.get_cmap('terrain').reversed()
                    #cmap2 = plt.cm.get_cmap('gist_earth').reversed()
                    cmap2.set_bad(color="white")

                    cmap3 = plt.cm.get_cmap('RdYlGn')
                    cmap3.set_bad(color='white')

                    axs[lat, col].set_xlim(Xmin, Xmax)
                    axs[lat, col].set_ylim(Ymax, Ymin)
                    if col==0:
                        CMD = np.log10(catalog1_cmd[lon][lat]).T
                        norm_hess = colors.Normalize(vmin=np.log10(dist_threshold), vmax=np.log10(limits_hess))
                        hess_catalog = axs[lat, col].imshow(CMD, extent=extent, interpolation="nearest", cmap=cmap, aspect="auto", norm=norm_hess)
                        hess_catalog.set_clim(np.log10(dist_threshold), np.log10(limits_hess))

                    elif col==1:
                        CMD = np.log10(catalog2_cmd[lon][lat]).T
                        norm_hess = colors.Normalize(vmin=np.log10(dist_threshold), vmax=np.log10(limits_hess))
                        hess_bgmfast = axs[lat, col].imshow(CMD, extent=extent, interpolation="nearest", cmap=cmap, aspect="auto", norm=norm_hess)
                        hess_bgmfast.set_clim(np.log10(dist_threshold), np.log10(limits_hess))

                    elif col==2:
                        CMD = abs(distance_cmd[lon][lat]).T
                        norm_sum = colors.Normalize(vmin=0, vmax=limits_dist)
                        hess_sum = axs[lat, col].imshow(CMD, extent=extent, interpolation="nearest", cmap=cmap2, aspect="auto", norm=norm_sum)
                        hess_sum.set_clim(0, limits_dist)

                    elif col==3:
                        CMD = difference_cmd[lon][lat].T
                        norm_diff = colors.Normalize(vmin=-limits_diff, vmax=limits_diff)
                        hess_diff = axs[lat, col].imshow(CMD, extent=extent, interpolation="nearest", cmap=cmap3, aspect="auto", norm=norm_diff)
                        hess_diff.set_clim(-limits_diff, limits_diff)

            cax = fig.add_axes([0.065, 0.130, 0.015, 0.75])
            cb = fig.colorbar(hess_bgmfast, cax=cax, norm=norm_hess)
            cb.set_label(r"$\log(N_\star)$")
            cax.yaxis.set_label_position("left")
            cax.yaxis.set_ticks_position("left")

            cb2 = fig.colorbar(hess_sum, ax=axs[:, 2], norm=norm_sum, aspect=30)
            #cb2.set_label(r"$q|1 - R + \ln(R)|$")

            cb3 = fig.colorbar(hess_diff, ax=axs[:, 3], norm=norm_diff, aspect=30, ticklocation='left')
            #cb3.set_label(r"$N_\star$")

        if output!=False:
            fig.savefig(output, dpi=300)
        if show:
            plt.show()
        plt.close()


    def compute_distance(self, catalog2_data, catalog1_data):
        '''
        Compute the metric distance between two catalogs
        
        Input parameters
        ----------------
        catalog1_data : numpy array --> catalog data in the 4-dimensional space (Hess diagram + latitude + longitude) used as a summary statistics
        catalog2_data : numpy array --> catalog data in the 4-dimensional space (Hess diagram + latitude + longitude) used as a summary statistics
        Note: when comparing simulations and observed data, catalog 1 refers to the simulation and catalog 2 to the data according to Eq. (58) from Mor et al. 2018 for the computation of the distance
        
        Output parameters
        -----------------
        distance : float --> value of the metric distance between the catalogs
        '''

        distance = dist_metric_gdaf2(catalog2_data, catalog1_data)
        print('\nDistance between catalogs: %f\n' %distance)

        return distance
    
    
    def build_mass_age_space(self, data, mass_range, mass_bins, vrange='auto', y_tick_labels = ['SFH0', 'SFH1', 'SFH2',  'SFH3', 'SFH4', 'SFH5', 'SFH6', 'SFH7', 'SFH8', 'SFH9', 'SFH10', 'SFH9T', 'SFH10T', 'SFH11T', 'SFH12T'], show=True, output=False):
        
        from_mass_to_bins = lambda mass: int((mass - self.bgmfast_sim.x1)/self.bgmfast_sim.mass_step) + 1
        
        y_hist = [sum(data[i]) for i in range(len(data))]
        x_hist = [sum([data[j][i] for j in range(len(data))]) for i in range(len(data[0]))] 
        
        fig = plt.figure(figsize=(18, 6))
        gs = fig.add_gridspec(2, 2,  width_ratios=(12, 1), height_ratios=(1, 4), left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.02, hspace=0.05)
        
        ax = fig.add_subplot(gs[1, 0])
        
        if vrange=='auto':
            norm = colors.LogNorm(vmin=1e-1, vmax=data.max()*4/5)
            mass_age = ax.imshow(data, cmap='gist_stern_r', aspect='auto', norm=norm)
        elif vrange[0]=='auto':
            norm = colors.LogNorm(vmin=1e-1, vmax=vrange[1])
            mass_age = ax.imshow(data, cmap='gist_stern_r', aspect='auto', norm=norm)
        elif vrange[1]=='auto':
            norm = colors.LogNorm(vmin=vrange[0], vmax=data.max()*4/5)
            mass_age = ax.imshow(data, cmap='gist_stern_r', aspect='auto', norm=norm)
        else:
            norm = colors.LogNorm(vmin=vrange[0], vmax=vrange[1])
            mass_age = ax.imshow(data, cmap='gist_stern_r', aspect='auto', norm=norm)
                       
        cax = fig.add_axes([0.03, 0.11, 0.01, 0.6])
        fig.colorbar(mass_age, cax=cax, norm=norm)
        cax.yaxis.set_label_position("left")
        cax.yaxis.set_ticks_position("left")
        
        ax.set_xlabel('Mass')
        ax.set_ylabel('SFH')
        
        x_tick_labels = [round(i, 1) for i in np.linspace(mass_range[0], mass_range[1], mass_bins)]
        x_tick_labels_bin = [from_mass_to_bins(i) for i in x_tick_labels]
        ax.set_xticks(ticks=x_tick_labels_bin, labels=x_tick_labels)
        ax.set_yticks(ticks=np.arange(len(y_tick_labels)), labels=y_tick_labels)
        
        ax.set_xlim([from_mass_to_bins(mass_range[0]), from_mass_to_bins(mass_range[1])])
        ax.set_ylim([len(y_tick_labels), -1])
        
        ax.vlines(from_mass_to_bins(self.bgmfast_sim.x2_ms), list(range(len(data)))[0], list(range(len(data)))[-1], colors='grey', linestyle='dashed', linewidths=0.5)
        ax.vlines(from_mass_to_bins(self.bgmfast_sim.x3_ms), list(range(len(data)))[0], list(range(len(data)))[-1], colors='grey', linestyle='dashed', linewidths=0.5)
        ax.grid(axis='x', linewidth=0.3)
        
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)
        
        ax_histx.stairs(x_hist)
        ax_histy.stairs(y_hist, orientation='horizontal')
        
        ax_histx.set_title('Present-day mass function', fontsize=10)
        ax_histy.text(1.15, 0.5, 'Present-day age distribution', fontsize=10, rotation=-90, ha='center', va='center', transform=ax_histy.transAxes)
        
        if output!=False:
            fig.savefig(output, dpi=300)
        if show:
            plt.show()
        plt.close()
        
        total_num_stars = data.sum()
        print('Total number of stars:', data.sum())
        
        return total_num_stars

    
def cmd_to_bins_table(bgmfast_cmd, output_file):
    '''
    Convert a Hess diagram into a table with the values of the bins 
    
    Input parameters
    ----------------
    bgmfast_cmd : numpy array or list --> 4-dimensional numpy array with the Hess diagrams corresponding to each one of the longitude and latitude ranges. It is also possible to put different numpy arrays in a list 
    output_file : str --> directory of the output file of the plot
    
    Output parameters
    -----------------
    df : Pandas DataFrame --> table with the values of the bins
    '''
    
    binning_parameters = parameters.binning_parameters
    grp_step = binning_parameters['Xstep'].value[0]
    grp_min = binning_parameters['Xmin'].value
    mvarpi_step = binning_parameters['Ystep'].value[0]
    mvarpi_min = binning_parameters['Ymin'].value

    if not isinstance(bgmfast_cmd, list):
        bgmfast_cmd = [bgmfast_cmd]
    
    counts = []
    for i in range(len(bgmfast_cmd)):
        counts.append([])
    
    longitudes = []
    latitudes = []
    grps = []
    mvarpis = []
    for lon in range(len(bgmfast_cmd[0])):
        for lat in range(len(bgmfast_cmd[0][lon])):
            for grp in range(len(bgmfast_cmd[0][lon][lat])):
                for mvarpi in range(len(bgmfast_cmd[0][lon][lat][grp])):
                    longitudes.append(lon)
                    latitudes.append(lat)
                    grps.append(grp_min + grp*grp_step + grp_step/2)
                    mvarpis.append(mvarpi_min + mvarpi*mvarpi_step + mvarpi_step/2)
                    for i in range(len(bgmfast_cmd)):
                        counts[i].append(bgmfast_cmd[i][lon][lat][grp][mvarpi])
                    
    data = {'longitude_bin': longitudes, 'latitude_bin': latitudes, 'grp_bin': grps, 'mvarpi_bin': mvarpis}
    if len(bgmfast_cmd)==1:
        data['counts'] = counts[0]
    else:
        for i in range(len(bgmfast_cmd)):
            data['counts' + str(i)] = counts[i]
    df = pd.DataFrame(data)
    
    if output_file!=False:
        df.to_csv(output_file, index=False)
    
    return df


