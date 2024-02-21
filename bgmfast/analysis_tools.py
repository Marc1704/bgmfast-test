from bgmfast.bgmfast_simulation_class import bgmfast_simulation
from bgmfast.auxiliary_functions import *
from bgmfast import parameters
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
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
    
    def __init__(self, bgmfast_output_file, num_acc_sim=100, quantiles=[0.16, 0.84],
                 free_params=parameters.bgmfast_parameters['free_params'].value,
                 fixed_params=parameters.bgmfast_parameters['fixed_params'].value, dist_index=14, sfh_ranges=[0, 0.150, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], imf_ranges=[0.015, 0.5, 1.53, 120], show=True):
        
        '''
        Initialize the output_file_analysis class
        
        Input parameters
        ----------------
        bgmfast_output_file : str --> BGM FASt + ABC output file name
        num_acc_sim : int --> number of accepted simulations per step 
        quantiles : list --> the lower and the upper quantiles chosen to represent the uncertainty of the most probable value
        free_params : dict --> dictionary with the names of the free parameters as keys and the position in the list of free parameters as values
        fixed_params : dict --> dictionary with the names of the fixed parameters and their values
        dist_index : int --> column position of the distance in the output file
        sfh_ranges : list --> limits of the different SFH intervals
        show : bool --> whether we want to show the information on the output file or not
        '''
        
        self.free_params = free_params
        self.fixed_params = fixed_params
        self.sfh_indices = sorted([free_params[i] for i in free_params.keys() if 'sfh' in i])
        self.imf_indices = sorted([free_params[i] for i in free_params.keys() if 'alpha' in i])
        self.dist_index = dist_index
        self.sfh_ranges = sfh_ranges
        self.imf_ranges = imf_ranges
        
        print(self.sfh_indices)
        print(self.imf_indices)
        
        datafile = open(bgmfast_output_file, 'r')
        
        counter = 0
        for line in datafile.readlines():
            if counter==0:
                self.header = line.split('\t')
                self.red_header = self.header[1:] #not including datetime
                self.num_columns = len(self.header)
                self.num_params = self.num_columns - 3 #3 is refered to datetime, dist and wgt columns
                self.datetime = []
                self.data = {item: [] for item in self.red_header}
                counter += 1
                continue
            
            if line[0]=='':
                break
            
            dataline = line.split('\t')[:-1]
            self.datetime.append(datetime.fromisoformat(dataline[0]))
            del dataline[0]
            item_counter = 0
            for item in self.red_header:
                if item_counter in self.sfh_indices:
                    param_value = float(dataline[item_counter])
                    sfh_param = param_value/(self.sfh_ranges[item_counter - self.sfh_indices[0] + 1] - self.sfh_ranges[item_counter - self.sfh_indices[0]])
                    self.data[item].append(sfh_param)
                else:
                    param_value = float(dataline[item_counter])
                    self.data[item].append(param_value)
                item_counter += 1
            counter += 1
        
        self.num_datapoints = len(self.data[self.red_header[0]])
        self.num_rows = self.num_datapoints + 1 #1 is refered to the header
        self.ndim = self.num_params
        self.num_acc_sim = num_acc_sim
        self.num_steps = int(self.num_datapoints/self.num_acc_sim)
        self.quantiles = quantiles
        self.quantiles_median = [quantiles[0], 0.5, quantiles[1]]
        
        if show:
            print('Header:', self.header)
            print('Reduced header:', self.red_header)
            print('Number of parameters:', self.num_params)
            print('Number of columns:', self.num_columns)
            print('Number of rows:', self.num_rows)
            print('Number of data points:', self.num_datapoints)
            print('Number of steps:', self.num_steps)
            print('Number of accepted simulations per step:', self.num_acc_sim)
    
    
    def final_params(self, return_val='mpv_quant', show=False):
        
        '''
        Obtain the most probable value of the parameters derived with BGM FASt + ABC
        
        Input parameters
        ----------------
        return_val : string --> whether we want the most probable value and the quantiles ('mpv_quant'); the most probable value, the quantiles, the independent variable and its estimated probability distribution function ('mpv_quant_dist'); or the median and the quantiles ('median_quant')
        show : bool --> whether we want to show the final parameters or not 
        
        Output parameters
        -----------------
        self.final_parameters : dict --> dictionary with the values of the parameters derived with BGM FASt + ABC
        '''
        
        self.final_parameters = {}
        for param in self.red_header:
            self.final_parameters[param] = kde_fit(self.data[param], quantiles=self.quantiles, return_val=return_val)
        
        if show:
            print('Final parameters:')
            for param in self.red_header:
                print('\t %s:' %param, self.final_parameters[param])
                
        return self.final_parameters
    
    
    def up_to_step(self, step, return_val='mpv_quant'):
        
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
        
        self.step_data = {}
        for param in self.red_header:
            self.step_data[param] = self.data[param][0:(step + 1)*self.num_acc_sim]
            
        self.step_parameters = {}
        for param in self.red_header:
            self.step_parameters[param] = kde_fit(self.step_data[param], quantiles=self.quantiles, return_val=return_val)
            
        return self.step_data, self.step_parameters
    
    
    def single_step(self, step, return_val='median_quant'):
        
        '''
        Return the median of the BGM FASt + ABC parameters taking into account only the given step
        
        Input parameters
        ----------------
        step : int --> step for which we want to compute the results
        return_val : string --> whether we want the most probable value and the quantiles ('mpv_quant'); the most probable value, the quantiles, the independent variable and its estimated probability distribution function ('mpv_quant_dist'); or the median and the quantiles ('median_quant')
        
        Output parameters
        -----------------
        self.single_step_data : dict --> dictionary with all the values of the parameters at the given step
        self.single_step_parameters : dict --> dictionary with the values of the parameters derived with BGM FASt + ABC
        '''
        
        self.single_step_data = {}
        for param in self.red_header:
            self.single_step_data[param] = self.data[param][step*self.num_acc_sim:(step + 1)*self.num_acc_sim]
        
        self.single_step_parameters = {}
        for param in self.red_header:
            self.single_step_parameters[param] = kde_fit(self.single_step_data[param], quantiles=self.quantiles, return_val=return_val)
            
        return self.single_step_data, self.single_step_parameters
    
    
    def build_sfh(self, output=False, show=False, ms_sfh=True, prior_sfh_means='ms_values', prior_sfh_std=2, step=False, axis=False, return_val='mpv_quant', limits=[0, 11]):
        
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

        central_sfh_ranges = [self.sfh_ranges[i] + (self.sfh_ranges[i+1] - self.sfh_ranges[i])/2 for i in range(len(self.sfh_ranges) - 1)]
        
        for i in range(len(self.red_header)):
            if i in self.sfh_indices:
                if not step:
                    final_params = self.final_params(return_val=return_val)
                    fparameters = [final_params[list(final_params.keys())[j]] for j in range(len(final_params)) if j in self.sfh_indices]
                else:
                    step_params = self.select_step(step, return_val=return_val)[1]
                    fparameters = [step_params[list(step_params.keys())[j]] for j in range(len(step_params)) if j in self.sfh_indices]
                    
        sfh = [i[1]for i in fparameters]
        sfh_quant1 = [i[1] - i[0] for i in fparameters]
        sfh_quant2 = [i[2] - i[1] for i in fparameters]
        sfh_quantiles = np.array([sfh_quant1, sfh_quant2])
        
        ms_parameters = parameters.ms_parameters
        SigmaParam_ms = ms_parameters['SigmaParam_ms'].value
        midpopbin_ms = ms_parameters['midpopbin_ms'].value
        lastpopbin_ms = ms_parameters['lastpopbin_ms'].value
        ms_params_sfh = np.append(SigmaParam_ms[:4], midpopbin_ms)
        ms_params_sfh = np.append(ms_params_sfh, lastpopbin_ms)
        
        if prior_sfh_means!=False:
            if prior_sfh_means=='ms_values':
                prior_sfh_means = ms_params_sfh
            else:
                prior_sfh_means = [i/(self.sfh_ranges[j+1] - self.sfh_ranges[j]) for i, j in zip(prior_sfh_means, range(len(self.sfh_ranges) - 1))]
                
        if not step:
            fig, ax = plt.subplots()
        else:
            ax = axis
            ax.clear()
            ax.set_title('Step ' + str(step), fontsize=15)
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 12)
        
        ax.errorbar(central_sfh_ranges, sfh, yerr=sfh_quantiles, fmt='o--', capsize=5, label='BGM FASt', color='blue', zorder=3)
        
        if prior_sfh_means.all()!=False:
            if prior_sfh_std==False:
                ax.errorbar(central_sfh_ranges, prior_sfh_means, fmt='o--', capsize=5, label='Priors', color='red', zorder=1)
            else:
                ax.errorbar(central_sfh_ranges, prior_sfh_means, yerr=prior_sfh_std, fmt='o--', capsize=5, label='Priors', color='red', zorder=1)
        if ms_sfh==True:
            ax.errorbar(central_sfh_ranges, ms_params_sfh, fmt='o--', capsize=5, label='MS', color='green', zorder=2)
            
        ax.invert_xaxis()
        ax.set_ylim(limits[0], limits[1])
        ax.set_xlabel('Age (Gyr)')
        ax.set_ylabel(r'Mean SFH per age bin ($M_\odot/$pc$^2/$Gyr)')
        ax.legend()
        
        if output!=False:
            fig.savefig(output, dpi=300)
        if show:
            plt.show()

    
    def build_imf(self, output=False, show=False, ms_imf=True, prior_imf_means='ms_values', prior_imf_std=2, step=False, axis=False, return_val='mpv_quant', limits=[-1, 3]):
        
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
            
        for i in range(len(self.red_header)):
            if i in self.imf_indices:
                if not step:
                    final_params = self.final_params(return_val=return_val)
                    fparameters = [final_params[list(final_params.keys())[j]] for j in range(len(final_params)) if j in self.imf_indices]
                else:
                    step_params = self.select_step(step, return_val=return_val)[1]
                    fparameters = [step_params[list(step_params.keys())[j]] for j in range(len(step_params)) if j in self.imf_indices]
        
        imf = [i[1] for i in fparameters]
        imf_quant1 = [i[0] for i in fparameters]
        imf_quant2 = [i[2] for i in fparameters]
        
        ms_parameters = parameters.ms_parameters
        alpha1_ms = ms_parameters['alpha1_ms'].value
        alpha2_ms = ms_parameters['alpha2_ms'].value
        alpha3_ms = ms_parameters['alpha3_ms'].value
        
        if 'alpha1' in self.fixed_params.keys():
            ms_params_imf = [alpha2_ms, alpha3_ms]
            imf_ranges = self.imf_ranges[1:]
        else:
            ms_params_imf = [alpha1_ms, alpha2_ms, alpha3_ms]
        
        if prior_imf_means!=False:
            if prior_imf_means=='ms_values':
                prior_imf_means = ms_params_imf
        
        if not step:
            fig, ax = plt.subplots()
        else:
            ax = axis
            ax.clear()
            ax.set_title('Step ' + str(step), fontsize=15)
            
        for i in range(len(self.imf_indices)):
            if i==0:
                x = np.linspace(imf_ranges[i], imf_ranges[i+1])
                ax.plot(x, np.full_like(x, imf[i]), label='BGM FASt', linestyle='-', color='blue', zorder=3)
                ax.fill_between(x, imf_quant1[i], imf_quant2[i], color='yellow', alpha=0.3)
                if prior_imf_means==False:
                    ax.plot(x, np.full_like(x, prior_imf_means[i]), label='Priors', linestyle='--', color='red', zorder=1)
                if ms_imf==True:
                    ax.plot(x, np.full_like(x, ms_params_imf[i]), label='MS/Priors', linestyle='--', color='green', zorder=2)
                ax.axvline(imf_ranges[i+1], alpha=0.5, linestyle='--', color='black', linewidth=1)
            else:
                x = np.linspace(imf_ranges[i], imf_ranges[i+1])
                ax.plot(x, np.full_like(x, imf[i]), linestyle='-', color='blue')
                ax.fill_between(x, imf_quant1[i], imf_quant2[i], color='yellow', alpha=0.3)
                if prior_imf_means==False:
                    ax.plot(x, np.full_like(x, prior_imf_means[i]), linestyle='--', color='red', zorder=1)
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
            
            
    def build_real_imf(self, output=False, show=False, ms_imf=True, prior_imf_means='ms_values', prior_imf_std=2, step=False, axis=False, return_val='mpv_quant'):
        
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
            
        for i in range(len(self.red_header)):
            if i in self.imf_indices:
                if not step:
                    final_params = self.final_params(return_val=return_val)
                    fparameters = [final_params[list(final_params.keys())[j]] for j in range(len(final_params)) if j in self.imf_indices]
                else:
                    step_params = self.select_step(step, return_val=return_val)[1]
                    fparameters = [step_params[list(step_params.keys())[j]] for j in range(len(step_params)) if j in self.imf_indices]
        
        imf = [i[1] for i in fparameters]
        
        ms_parameters = parameters.ms_parameters
        alpha1_ms = ms_parameters['alpha1_ms'].value
        alpha2_ms = ms_parameters['alpha2_ms'].value
        alpha3_ms = ms_parameters['alpha3_ms'].value
        ms_params_imf = [alpha1_ms, alpha2_ms, alpha3_ms]
        
        if prior_imf_means!=False:
            if prior_imf_means=='ms_values':
                prior_imf_means = ms_params_imf
        
        if 'alpha1' in self.fixed_params.keys():
            imf = [alpha1_ms, imf[0], imf[1]]
            
        general_parameters = parameters.general_parameters
        ps_parameters = parameters.ps_parameters
        x1 = general_parameters['x1'].value
        x2 = ps_parameters['x2_ps'].value
        x3 = ps_parameters['x3_ps'].value
        x4 = general_parameters['x4'].value
        
        K = Continuity_Coeficients_func(imf[0], imf[1], imf[2], x1, x2, x3, x4)
        K_priors = Continuity_Coeficients_func(prior_imf_means[0], prior_imf_means[1], prior_imf_means[2], x1, x2, x3, x4)
        K_ms = Continuity_Coeficients_func(ms_params_imf[0], ms_params_imf[1], ms_params_imf[2], x1, x2, x3, x4)
        
        if not step:
            fig, ax = plt.subplots()
        else:
            ax = axis
            ax.clear()
            ax.title('Step ' + str(step), fontsize=15)
            
        for i in range(len(imf)):
            if i==0:
                x = np.linspace(self.imf_ranges[i], self.imf_ranges[i+1])
                fi = lambda m: K[i]*m**(-imf[i])
                real_imf = [fi(m) for m in x]
                ax.plot(x, real_imf, label='BGMFASt', linestyle='-', color='blue')
                if prior_imf_means!=False:
                    fi = lambda m: K_priors[i]*m**(-prior_imf_means[i])
                    real_imf_priors = [fi(m) for m in x]
                    ax.plot(x, real_imf_priors, label='Priors', linestyle='--', color='red', zorder=1)
                if ms_imf==True:
                    fi = lambda m: K_ms[i]*m**(-ms_params_imf[i])
                    real_imf_ms = [fi(m) for m in x]
                    ax.plot(x, real_imf_ms, label='MS', linestyle='--', color='green', zorder=2)
                ax.axvline(self.imf_ranges[i+1], alpha=0.5, linestyle='--', color='black', linewidth=1)
            else:
                x = np.linspace(self.imf_ranges[i], self.imf_ranges[i+1])
                fi = lambda m: K[i]*m**(-imf[i])
                real_imf = [fi(m) for m in x]
                ax.plot(x, real_imf, linestyle='-', color='blue')
                if prior_imf_means!=False:
                    fi = lambda m: K_priors[i]*m**(-prior_imf_means[i])
                    real_imf_priors = [fi(m) for m in x]
                    ax.plot(x, real_imf_priors, linestyle='--', color='red', zorder=1)
                if ms_imf==True:
                    fi = lambda m: K_ms[i]*m**(-ms_params_imf[i])
                    real_imf_ms = [fi(m) for m in x]
                    ax.plot(x, real_imf_ms, linestyle='--', color='green', zorder=2)
                ax.axvline(self.imf_ranges[i+1], alpha=0.5, linestyle='--', color='black', linewidth=1)
                
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(self.imf_ranges[0], self.imf_ranges[-1])
        ax.set_xlabel(r'Stellar mass ($M/M_\odot$)')
        ax.set_ylabel(r'IMF')
        ax.legend(loc=4)

        if output!=False:
            fig.savefig(output, dpi=300)
        if show:
            plt.show()
            plt.close()
            
    
    def distance_evolution(self, output=False, show=False):
        
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
        for distance in self.data[self.header[self.dist_index]]:
            if counter%self.num_acc_sim==0:
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
    
    
    def parameter_evolution(self, index, output=False, show=True, param_name=False, plot_single_step={'median': True, 'median_quant': True, 'mpv': False, 'mpv_quant': False, 'min': False, 'max': False}, plot_acc={'median': False, 'median_quant': False, 'mpv': True, 'mpv_quant': True, 'min': False, 'max': False}):
        
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

        if not param_name==False:
            print('\nBuilding %s evolution...\n' %param_name)
        else:
            print('\nBuilding parameter evolution...\n')

        fig, ax = plt.subplots()

        if plot_single_step['median']==True or plot_single_step['median_quant']==True:
            single_parameters = []
            single_mins = []
            single_maxs = []
            for step in range(self.num_steps):
                single_step_params = self.single_step(step, return_val='median_quant')[1]
                single_parameters.append(single_step_params[self.red_header[index]])
                single_mins.append(min(self.single_step_data[self.red_header[index]]))
                single_maxs.append(max(self.single_step_data[self.red_header[index]]))

            single_medians = [i[1] for i in single_parameters]

            if plot_single_step['median_quant']==True:
                single_quant1 = [i[1] - i[0] for i in single_parameters]
                single_quant2 = [i[2] - i[1] for i in single_parameters]
                single_quantiles = np.array([single_quant1, single_quant2])
                ax.errorbar(range(self.num_steps), single_medians, yerr=single_quantiles, fmt='o--', capsize=5, label='Single step (median)')

            elif plot_single_step['median_quant']==False:
                ax.scatter(range(self.num_steps), single_medians, marker='o', s=5, label='Single step (median)')

            else:
                print('plot_single_step has to be a boolean [True/False]')
                sys.exit()

        if plot_single_step['mpv']==True or plot_single_step['mpv_quant']==True:
            single_parameters = []
            single_mins = []
            single_maxs = []
            for step in range(self.num_steps):
                single_step_params = self.single_step(step, return_val='mpv_quant')[1]
                single_parameters.append(single_step_params[self.red_header[index]])
                single_mins.append(min(self.single_step_data[self.red_header[index]]))
                single_maxs.append(max(self.single_step_data[self.red_header[index]]))

            single_mpv = [i[1] for i in single_parameters]

            if plot_single_step['mpv_quant']==True:
                single_quant1 = [i[1] - i[0] for i in single_parameters]
                single_quant2 = [i[2] - i[1] for i in single_parameters]
                single_quantiles = np.array([single_quant1, single_quant2])
                ax.errorbar(range(self.num_steps), single_mpv, yerr=single_quantiles, fmt='o--', capsize=5, label='Single step (mpv)')

            elif plot_single_step['mpv_quant']==False:
                ax.scatter(range(self.num_steps), single_mpv, marker='o', s=5, label='Single step (mpv)')

            else:
                print('plot_single_step has to be a boolean [True/False]')
                sys.exit()

        if plot_single_step['min']==True:
            ax.scatter(range(self.num_steps), single_mins, marker='o', s=5, label='Single step (min)')
        if plot_single_step['max']==True:
            ax.scatter(range(self.num_steps), single_maxs, marker='o', s=5, label='Single step (max)')

        if plot_acc['median']==True or plot_acc['median_quant']==True:
            aparameters = []
            acc_mins = []
            acc_maxs = []
            for step in range(self.num_steps):
                step_params = self.up_to_step(step, return_val='median_quant')[1]
                aparameters.append(step_params[self.red_header[index]])

                acc_mins.append(min(self.step_data[self.red_header[index]]))
                acc_maxs.append(max(self.step_data[self.red_header[index]]))

            acc_medians = [i[1] for i in aparameters]

            if plot_acc['median_quant']==True:
                acc_quant1 = [i[1] - i[0] for i in aparameters]
                acc_quant2 = [i[2] - i[1] for i in aparameters]
                acc_quantiles = np.array([single_quant1, single_quant2])
                ax.errorbar(range(self.num_steps), acc_medians, yerr=acc_quantiles, fmt='o--', capsize=5, label='Accumulated (median)')

            elif plot_acc['median_quant']==False:
                ax.scatter(range(self.num_steps), acc_medians, marker='o', s=5, label='Accumulated (median)')

            else:
                print('plot_single_step has to be a boolean [True/False]')
                sys.exit()

        if plot_acc['mpv']==True or plot_acc['mpv_quant']==True:
            aparameters = []
            acc_mins = []
            acc_maxs = []
            for step in range(self.num_steps):

                step_params = self.up_to_step(step, return_val='mpv_quant')[1]
                aparameters.append(step_params[self.red_header[index]])

                acc_mins.append(min(self.step_data[self.red_header[index]]))
                acc_maxs.append(max(self.step_data[self.red_header[index]]))

            acc_mpv = [i[1] for i in aparameters]

            if plot_acc['mpv_quant']==True:
                acc_quant1 = [i[1] - i[0] for i in aparameters]
                acc_quant2 = [i[2] - i[1] for i in aparameters]
                acc_quantiles = np.array([single_quant1, single_quant2])
                ax.errorbar(range(self.num_steps), acc_mpv, yerr=acc_quantiles, fmt='o--', capsize=5, label='Accumulated (mpv)')

            elif plot_acc['mpv_quant']==False:
                ax.scatter(range(self.num_steps), acc_mpv, marker='o', s=5, label='Accumulated (mpv)')

            else:
                print('plot_single_step has to be a boolean [True/False]')
                sys.exit()

        if plot_acc['min']==True:
            ax.scatter(range(self.num_steps), acc_mins, marker='o', s=5, label='Accumulated (min)')
        if plot_acc['max']==True:
            ax.scatter(range(self.num_steps), acc_maxs, marker='o', s=5, label='Accumulated (max)')

        ax.set_xlabel('Steps')
        ax.legend()
        if param_name:
            ax.set_ylabel(param_name)
        else:
            ax.set_ylabel(self.red_header[index])

        if output!=False:
            fig.savefig(output, dpi=300)
        if show:
            plt.show()
            
        
    def cornerplot(self, output=False, show=False, ranges=[(-1.5, 4.5), (-1, 10), (-1, 6), (-1, 6), (-1, 10), (-1, 14), (-1, 12), (-1, 12), (-1, 10), (-1, 14), (-1, 12), (-1, 12), (-1, 16)], corner_num_params=None, labels=[r'$\alpha_2$', r'$\alpha_3$', r'SFH$_1$', r'SFH$_2$', r'SFH$_3$', r'SFH$_4$', r'SFH$_5$', r'SFH$_6$', r'SFH$_7$', r'SFH$_8$', r'SFH$_9$', r'SFH$_{10}$', r'SFH$_{11}$', r'SFH$_{12}$'], num_decimals=2, dpi=300):
        
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
        
        if corner_num_params==None:
            self.corner_num_params = self.num_params
        else:
            self.corner_num_params = corner_num_params
            
        self.corner_ndim, self.nsamples = self.corner_num_params, self.num_datapoints
        self.samples = []
        for datapoint in range(self.num_datapoints):
            for param in self.red_header[:self.corner_num_params]: #discard dist and wgt
                self.samples.append(self.data[param][datapoint])
        xy_sample = np.array(self.samples).reshape(-1, self.corner_num_params).T.tolist()
        self.samples = np.array(self.samples).reshape([self.nsamples, self.corner_ndim])
        
        if ranges=='auto':
            figure = corner.corner(self.samples, labels=labels, show_titles=True, color='darkgreen', verbose=True, title_kwargs={"fontsize": 8}, title_fmt="0.3f", hist_kwargs={"density": True, "linewidth":2})
        else:
            figure = corner.corner(self.samples, range=ranges, labels=labels, show_titles=True, color='darkgreen', verbose=True, title_kwargs={"fontsize": 8}, title_fmt="0.3f", hist_kwargs={"density": True, "linewidth":2})
            
        axes = np.array(figure.axes).reshape((self.corner_ndim, self.corner_ndim))
        params_mpv = self.final_params(return_val='mpv_quant_dist')
        params_quant = self.final_params(return_val='median_quant')
        
        ms_parameters = parameters.ms_parameters
        ms_values = [ms_parameters['alpha1_ms'].value, ms_parameters['alpha2_ms'].value, ms_parameters['alpha3_ms'].value]
        ms_values_sfh = [ms_parameters['SigmaParam_ms'].value[0], ms_parameters['SigmaParam_ms'].value[1], ms_parameters['SigmaParam_ms'].value[2], ms_parameters['SigmaParam_ms'].value[3], ms_parameters['midpopbin_ms'].value[0], ms_parameters['midpopbin_ms'].value[1], ms_parameters['midpopbin_ms'].value[2], ms_parameters['midpopbin_ms'].value[3],
        ms_parameters['lastpopbin_ms'].value[0],
        ms_parameters['lastpopbin_ms'].value[1],
        ms_parameters['lastpopbin_ms'].value[2]]
        ms_values_sfh = [sfh_param/(self.sfh_ranges[i+1] - self.sfh_ranges[i]) for sfh_param, i in zip(ms_values_sfh, range(len(self.sfh_ranges)))]
        ms_values.extend(ms_values_sfh)
        if 'alpha1' in self.fixed_params.keys():
            ms_values = ms_values[1:]
        ms_values = ms_values[:self.corner_num_params]
        
        mpv_values = []
        for i, param_name in zip(range(self.corner_ndim), self.red_header):
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

            title = labels[i] + "$ = {}^{{+{}}}_{{-{}}}$".format(mpv_rounded, up_error_rounded, down_error_rounded)

            ax.set_title(title, fontsize=10)

            for j in range(i):
                ax2 = axes[i, j]
                ax2.scatter(ms_values[j], ms_values[i], marker='x', s=50, color='magenta', lw=3)
                ax2.scatter(mpv_values[j], mpv_values[i], marker='*', s=50, color='orange', lw=3)
                coef, pvalue = pearsonr(xy_sample[i], xy_sample[j])
                ax2.text(0.85, 0.92, str(round(float(coef), 2)), ha='center', va='center', transform=ax2.transAxes, color="black", fontweight='bold')
                
        if output!=False:
            figure.savefig(output, dpi=dpi)
        if show:
            plt.show()

        #self.medians = np.median(self.samples, axis=0)

        #return self.medians


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

        pass


    def generate_catalog_hess_diagram(self, filename_catalog, colnames=['G','GRp','longitude','latitude', 'Mvarpi', 'parallax'], Gmax=13.0):
        '''
        Generate catalog Hess diagram
        
        Input parameters 
        ----------------
        filename_catalog : str --> directory of the catalog file
        colnames : list --> list with the name of the columns in the catalog with the following order: G, G-Rp, longitude, latitude, M_G' and parallax 
        Gmax : int or float --> limitting magnitude
        
        Output parameters
        -----------------
        catalog_cmd : numpy array --> 4-dimensional numpy array (Hess diagram + latitude + longitude) containing the complete Hess diagram
        catalog_data : numpy array --> 4-dimensional numpy array with the Hess diagrams corresponding to each one of the longitude and latitude ranges
        '''
        
        self.bgmfast_sim.read_catalog(filename_catalog, colnames, Gmax)
        self.bgmfast_sim.generate_catalog_cmd()
        catalog_cmd = self.bgmfast_sim.return_cmd()[0]
        catalog_data = self.bgmfast_sim.return_cmd()[1]
        
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
    
    
    def build_hess_diagrams_plots(self, catalog1_cmd, catalog2_cmd, distance_cmd, difference_cmd, quocient_cmd, output=False, show=True, titles=['Catalog 1', 'Catalog 2', r'$\delta_P$(Catalog 1, Catalog 2)', 'Catalog 1 - Catalog 2', 'Catalog 1/Catalog 2'], limits='auto'):
        '''
        Build the Hess diagrams of two catalogs and their differences
        
        Input parameters
        ----------------
        catalog1_cmd : numpy array --> 4-dimensional numpy array (Hess diagram + latitude + longitude) containing the complete Hess diagram of the first catalog
        catalog2_cmd : numpy array --> 4-dimensional numpy array (Hess diagram + latitude + longitude) containing the complete Hess diagram of the second catalog
        Note: when comparing simulations and observed data, catalog 1 refers to the simulation and catalog 2 to the data according to Eq. (58) from Mor et al. 2018 for the computation of the distance
        distance_cmd : numpy array --> 4-dimensional numpy array (Hess diagram + latitude + longitude) containing the distance per bin of the complete Hess diagrams between catalogs
        difference_cmd : numpy array --> 4-dimensional numpy array (Hess diagram + latitude + longitude) containing the absolute difference in number of stars per bin of the complete Hess diagrams between catalogs
        quocient_cmd : numpy array --> 4-dimensional numpy array (Hess diagram + latitude + longitude) containing the quocient of the number of stars per bin of the complete Hess diagrams between catalogs
        output : str or False --> directory of the output file of the plot
        show : boolean --> wether you want the plot to be displayed or not
        titles : list --> titles of the five different columns of the plot
        limits : list or 'auto' --> upper limits of the plots in each one of the four colour bars or set to 'auto'
        '''

        acc_parameters = parameters.acc_parameters
        nLonbins = acc_parameters['nLonbins'].value
        nLatbins = acc_parameters['nLatbins'].value

        binning_parameters = parameters.binning_parameters
        Xmin = binning_parameters['Xmin'].value
        Xmax = binning_parameters['Xmax'].value
        Ymin = binning_parameters['Ymin'].value
        Ymax = binning_parameters['Ymax'].value

        extent = [Xmin, Xmax, Ymax, Ymin]

        fig = plt.figure(figsize=(22, 9))
        fig.tight_layout()
        axs = fig.subplots(nLatbins, len(titles), gridspec_kw={'width_ratios': [1, 1, 1.25, 1.25, 1.25]})

        if limits=='auto':
            limits_hess = max(np.array(catalog1_cmd).max(), np.array(catalog2_cmd).max())
            limits_dist = max([abs(np.quantile(distance_cmd, 0.01)), abs(np.quantile(distance_cmd, 0.99))])
            limits_diff = max([abs(np.quantile(difference_cmd, 0.01)), abs(np.quantile(difference_cmd, 0.99))])
            max_lim = max([abs(np.log10(np.quantile(quocient_cmd, 0.01))), abs(np.log10(np.quantile(quocient_cmd, 0.99)))])
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
            if limits[3]=='auto':
                max_lim = max([abs(np.log10(np.quantile(quocient_cmd, 0.01))), abs(np.log10(np.quantile(quocient_cmd, 0.99)))])
            else:
                max_lim = np.log10(limits[3])
        limits_quoc = [10**(-max_lim), 10**(max_lim)]

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

                    cmap2 = plt.cm.get_cmap('BuPu')
                    cmap2.set_bad(color="white")

                    cmap3 = plt.cm.get_cmap('RdYlGn')
                    cmap3.set_bad(color='white')

                    axs[lat, col].set_xlim(Xmin, Xmax)
                    axs[lat, col].set_ylim(10, Ymin)
                    axs[lat, col].axhline(5, color='black', lw=0.5)

                    if col==0:
                        CMD = np.log10(catalog1_cmd[lon][lat]).T
                        norm_hess = colors.Normalize(vmin=0, vmax=np.log10(limits_hess))
                        hess_catalog = axs[lat, col].imshow(CMD, extent=extent, interpolation="nearest", cmap=cmap, aspect="auto", norm=norm_hess)
                        hess_catalog.set_clim(0, np.log10(limits_hess))

                    elif col==1:
                        CMD = np.log10(catalog2_cmd[lon][lat]).T
                        norm_hess = colors.Normalize(vmin=0, vmax=np.log10(limits_hess))
                        hess_bgmfast = axs[lat, col].imshow(CMD, extent=extent, interpolation="nearest", cmap=cmap, aspect="auto", norm=norm_hess)
                        hess_bgmfast.set_clim(0, np.log10(limits_hess))

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

                    elif col==4:
                        CMD = quocient_cmd[lon][lat].T
                        norm_quoc = colors.LogNorm(vmin=limits_quoc[0], vmax=limits_quoc[1])
                        hess_quoc = axs[lat, col].imshow(CMD, extent=extent, interpolation="nearest", cmap=cmap3, aspect="auto", norm=norm_quoc)
                        hess_quoc.set_clim(limits_quoc[0], limits_quoc[1])

            cax = fig.add_axes([0.065, 0.130, 0.015, 0.75])
            cb = fig.colorbar(hess_bgmfast, cax=cax, norm=norm_hess)
            cb.set_label(r"$\log(N_\star)$")
            cax.yaxis.set_label_position("left")
            cax.yaxis.set_ticks_position("left")

            cb2 = fig.colorbar(hess_sum, ax=axs[:, 2], norm=norm_sum, aspect=30)
            #cb2.set_label(r"$q|1 - R + \ln(R)|$")

            cb3 = fig.colorbar(hess_diff, ax=axs[:, 3], norm=norm_diff, aspect=30, ticklocation='left')
            #cb3.set_label(r"$N_\star$")

            cb4 = fig.colorbar(hess_quoc, ax=axs[:, 4], norm=norm_quoc, aspect=30)
            cb4.set_label(r'$R$')

        if output!=False:
            fig.savefig(output, dpi=300)
        if show:
            plt.show()


    def compute_distance(self, catalog2_cmd, catalog1_cmd):
        '''
        Compute the metric distance between two catalogs
        
        Input parameters
        ----------------
        catalog1_cmd : numpy array --> 4-dimensional numpy array with the Hess diagrams corresponding to each one of the longitude and latitude ranges for the first catalog
        catalog2_cmd : numpy array --> 4-dimensional numpy array with the Hess diagrams corresponding to each one of the longitude and latitude ranges for the second catalog
        Note: when comparing simulations and observed data, catalog 1 refers to the simulation and catalog 2 to the data according to Eq. (58) from Mor et al. 2018 for the computation of the distance
        
        Output parameters
        -----------------
        distance : float --> value of the metric distance between the catalogs
        '''

        distance = dist_metric_gdaf2(catalog2_cmd, catalog1_cmd)
        print('\nDistance between catalogs: %f\n' %distance)

        return distance
    
    
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


