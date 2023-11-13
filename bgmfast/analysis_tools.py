from bgmfast.bgmfast_simulation_class import bgmfast_simulation
from bgmfast.auxiliary_functions import *
from bgmfast import parameters
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd


class compare_hess_diagrams:

    def __init__(self):
        '''
        Initialize the compare_hess_diagrams class
        '''

        self.bgmfast_sim = bgmfast_simulation()
        self.bgmfast_sim.open_spark_session()

        self.bgmfast_sim.set_acc_parameters()
        self.bgmfast_sim.set_binning_parameters()

        pass


    def generate_catalog_hess_diagram(self, filename_catalog, colnames=['G','BpRp','longitude','latitude', 'Mvarpi', 'parallax'], Gmax=13.0):
        '''
        Generate catalog Hess diagram
        
        Input parameters 
        ----------------
        filename_catalog : str --> directory of the catalog file
        colnames : list --> list with the name of the columns in the catalog with the following order: G, Bp-Rp, longitude, latitude, M_G' and parallax 
        Gmax : int or float --> limitting magnitude
        
        Output parameters
        -----------------
        catalog_cmd : numpy array --> 4-dimensional numpy array (Hess diagram + latitude + longitude) containing the complete Hess diagram
        catalog_data : numpy array --> 4-dimensional numpy array with the Hess diagrams corresponding to each one of the longitude and latitude ranges
        '''
        
        self.bgmfast_sim.read_catalog(filename_catalog, colnames, Gmax)
        self.bgmfast_sim.generate_catalog_cmd()
        catalog_cmd = self.bgmfast_sim.return_cmd()[0]
        catalog_data = self.bgmfast_sim.return_cmd()[3]
        
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
        
        quocient_cmd = np.array([[[[catalog1_cmd[lon][lat][i][j]/catalog2_cmd[lon][lat][i][j] if catalog1_cmd[lon][lat][i][j]!=0 and catalog2_cmd[lon][lat][i][j]!=0 else (catalog1_cmd[lon][lat][i][j] + 1)/(catalog2_cmd[lon][lat][i][j] + 1) for j in range(len(catalog1_cmd[lon][lat][i]))] for i in range(len(catalog1_cmd[lon][lat]))] for lat in range(len(catalog1_cmd[lon]))] for lon in range(len(catalog1_cmd))])
        distance_cmd = np.array([[[[catalog2_cmd[lon][lat][i][j]*(1 - quocient_cmd[lon][lat][i][j] + np.log(quocient_cmd[lon][lat][i][j])) for j in range(len(quocient_cmd[lon][lat][i]))] for i in range(len(quocient_cmd[lon][lat]))] for lat in range(len(quocient_cmd[lon]))] for lon in range(len(quocient_cmd))])
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
                        axs[lat, col].set_xlabel("$Bp-Rp$")

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
    bgmfast_cmd : numpy array --> 4-dimensional numpy array with the Hess diagrams corresponding to each one of the longitude and latitude ranges
    output_file : str --> directory of the output file of the plot
    
    Output parameters
    -----------------
    df : Pandas DataFrame --> table with the values of the bins
    '''
    
    binning_parameters = parameters.binning_parameters
    bprp_steps = binning_parameters['Ylims_Xsteps'].value[0]
    bprp_min = binning_parameters['Xmin'].value
    mvarpi_steps = binning_parameters['Ylims_Ysteps'].value[0]
    mvarpi_min = binning_parameters['Ymin'].value

    longitudes = []
    latitudes = []
    bprps = []
    mvarpis = []
    counts = []
    for lon in range(len(bgmfast_cmd)):
        for lat in range(len(bgmfast_cmd[lon])):
            for bprp in range(len(bgmfast_cmd[lon][lat])):
                for mvarpi in range(len(bgmfast_cmd[lon][lat][bprp])):
                    longitudes.append(lon)
                    latitudes.append(lat)
                    bprps.append(bprp_min + bprp*bprp_steps + bprp_steps/2)
                    mvarpis.append(mvarpi_min + mvarpi*mvarpi_steps + mvarpi_steps/2)
                    counts.append(bgmfast_cmd[lon][lat][bprp][mvarpi])
                    
    data = {'longitude_bin': longitudes, 'latitude_bin': latitudes, 'bprp_bin': bprps, 'mvarpi_bin': mvarpis, 'counts': counts}
    df = pd.DataFrame(data)
    
    if output_file!=False:
        df.to_csv(output_file, index=False)
    
    return df


