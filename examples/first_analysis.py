from bgmfast import compare_catalogs
from bgmfast import parameters

filename_catalog = "./input_data/catalog/Gaia_DR3_G13.csv"
colnames_catalog = parameters.catalog_file_parameters['sel_columns_catalog'].value
Gmax_catalog = parameters.catalog_file_parameters['Gmax_catalog'].value

filename_ms = "./input_data/ms/ms_G13_err.csv"
colnames_ms = ['Gerr', 'GRperr', 'longitude', 'latitude', 'Mvarpi', 'parallaxerr']
Gmax_ms = parameters.ms_file_parameters['Gmax_ms'].value

comparison = compare_catalogs.compare_hess_diagrams()

catalog_cmd, catalog_data = comparison.generate_catalog_hess_diagram(filename_catalog, colnames_catalog, Gmax_catalog)

ms_cmd, ms_data = comparison.generate_catalog_hess_diagram(filename_ms, colnames_ms, Gmax_ms)

distance_cmd, difference_cmd, quocient_cmd = comparison.generate_difference_hess_diagram(catalog_cmd, ms_cmd)

distance = comparison.compute_distance(catalog_data, ms_data)

comparison.build_hess_diagrams_plots(catalog_cmd, ms_cmd, distance_cmd, difference_cmd, quocient_cmd)
