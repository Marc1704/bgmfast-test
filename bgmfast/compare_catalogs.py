from bgmfast.bgmfast_simulation_class import bgmfast_simulation
from bgmfast.auxiliary_functions import *


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


    def compute_distance(self, catalog1_data, catalog2_data):
        '''
        Compute the metric distance between two catalogs
        
        Input parameters
        ----------------
        catalog1_cmd : numpy array --> 4-dimensional numpy array with the Hess diagrams corresponding to each one of the longitude and latitude ranges for the first catalog
        catalog2_cmd : numpy array --> 4-dimensional numpy array with the Hess diagrams corresponding to each one of the longitude and latitude ranges for the second catalog
        
        Output parameters
        -----------------
        distance : float --> value of the metric distance between the catalogs
        '''

        distance = dist_metric_gdaf2(catalog1_data, catalog2_data)
        print('\nDistance between catalogs: %f\n' %distance)

        return distance


