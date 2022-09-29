from roost.airspace import *
from roost.pptp import *
from roost.wxtex import *
from roost.geoplot import *
from roost.bada4 import BADA4_jet_CR
from roost.plot import *
import xarray as xr



""" Inputs """

# Input a dataset, which includes the required meteorological variables
ds = xr.open_dataset("test/samples/data/20_12_2018_main.nc")

# Filter the dataset to the required variables
dsc = ds[['t', 'u', 'v', 'z', 'r', 'pv', 'C1', 'C2', 'olr', 'aCCF_CH4']]

# Information of the considered aircraft (notice that the user needs to input s .xml file)
apm = BADA4_jet_CR('A320-214', full_path='A320-214.xml')

# New coordinates for processing the input meteorological data
cfl = CoordinatesFromLimits({'latitude': (22, 73), 'longitude': (-28, 58)})

# Process the input data (i.e., dsc) and save it (i.e., 'processed.nc'). Notice that if 'processed.nc' already exists, the processing will be ignored. 
wc = WeatherCache('test/samples/data/20_12_2018_processed.nc', dsc, cfl.axes)

# The directory of the route graph from the end of SIDs (Standard Instrument Departures) to the beginning of the STARs (Standard Instrument Arrivals)
route_graph = RouteGraph.init_from_gexf('test/samples/graph/TESGA_KEDUB_d1.03.gexf')


""" Scenario Configuration """

# Load default configurations
problem_config = ProblemConfig()

# Origin waypoint (end of SID)
problem_config['origin'] = 'TESGA'

# Destination waypoint (Beginning of STAR)
problem_config['destination'] = 'KEDUB'

# Departure time
problem_config['departure_time'] = wc.ds.coords['time'].values[4] 

# Maximum flight level during the cruise phase
problem_config['FL_max']= 410 

# Minimum flight level during the cruise phase
problem_config['FL_min']= 260

# Initial flight level
problem_config['FL0'] = 100

# initial true airpeed
problem_config['tas0'] = 150


""" objective funciton """

# If yes, the climate impact is considered as an objective in the objective function to be optimized
problem_config['compute_accf'] = True

# # If yes, the NOx emission is considered as an objective in the objective function to be optimized
problem_config['compute_emissions'] = True


# Day-time contrails?
problem_config['dCon_index'] = 1

# Night-time contrails?
problem_config['nCon_index'] = 0

# Weights ATR of O3 in the objective function 
problem_config['CI_nox'] = 1

# Weights ATR of CO2 in the objective function 
problem_config['CI_co2'] = 1

# Weights ATR of H2O in the objective function 
problem_config['CI_h2o'] = 1

# Weights ATR of Contrails in the objective function 
problem_config['CI_contrail'] = 1

# Weights distance flown in persistent contrails formation areas in the objective function 
problem_config['CI_contrail_dis'] = 0


# Weights flight time in the objective function 
problem_config['CI'] = 0.75

# Weights fuel consumption in the objective function 
problem_config['CI_fuel'] = 0.51

# Weights operating cost in the objective function (cost_index * [CI * flight time + CI_fuel * fuel consumption])
problem_config['cost_index'] = 1

# Weights NOx emission in the objective function
problem_config['emission_index'] = 0

# Weights climate impact in the objective function
problem_config['climate_index'] = 1


""" Optimization configuration """ 
# Notice that more settings can be found in roost/pptp.py

# Loads default configurations
ccfg = ComputationalConfig()

# Number of search directions
ccfg['n_plans'] = 10

# Number of scenarios
ccfg['n_scenarios'] = 10

# Augmented Random Search step size
ccfg['ars_step_size'] = 0.25

# Nesterov velocity factor
ccfg['nesterov_velocity_factor'] = 0.75 

""" Run """ 

# Number of iterations
n_iters = 4000

C_ATR = [0.0, 1e11,  1e12]

dfs = {} 

# Performing optimization for different values weighting ATR in the objective function
sfpp = StructuredFlightPlanningProblem(apm, wc, route_graph, setup=problem_config, cconfig=ccfg)
for C in C_ATR:
    sfpp.pcfg['climate_index'] = C
    J, t = sfpp.timed_run(n_iters, noise_scaling=4, explo_exp=0.0, seq_offset=1000, exec_exp=0.25)
    dfs[C] = sfpp.get_profiles_dataframes()

# Save the optimized trajectories in .JSON file format    
generate_json (dfs, problem_config['departure_time'], C_ATR, problem_config['origin'], problem_config['destination'],  'test/Results/')

# Save the figure  depicting aircraft profile and climate impacts in .pdf file format    
plot_pprof(dfs, C_ATR,  'test/Results/')    
