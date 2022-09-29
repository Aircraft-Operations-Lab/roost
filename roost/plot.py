
import json
import numpy as np
from .geoplot import *
import matplotlib as mpl
import matplotlib.pyplot as plt

def get_json_dict(jpath):
    """
    Get dict from json file.
    :param jpath: json file path
    :return: dictionary from json
    """
    with open(jpath, 'rb') as fp:
        d = json.load(fp)
    fp.close()
    return d

def save_json_dict(jpath, rdict):
    """
    Save dict in json file.
    :param jpath: json file path.
    :param rdict: result dictionary.
    :return: save as json.
    """
    with open(jpath, 'w') as fp:
        json.dump(rdict, fp)
    fp.close()

def plot_pprof(dfs, CIs, path_save):
    
    cmap_dp = mpl.cm.plasma
    norm_dp = mpl.colors.Normalize(vmin=0, vmax=float(len(CIs))-1)
    col = []
    for i in range (0, len(CIs)):
        col.append ( cmap_dp(norm_dp(float(i))))
    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(12,8), sharex=False)
    n = len(dfs[CIs[0]])
    
    
    ## Flight Level
    
    for i, CI in enumerate(CIs):
        ax = axes[0,0]
        FL = [df.FL.values for df in dfs[CI]]
        median_FL = np.median(FL, axis=0)
        d = (np.cumsum(dfs[CI][0].d2n.values)/1852)
        ax.fill_between(d, np.min(FL, axis=0)+ 0.4*i, np.max(FL, axis=0)+ 0.8*i, alpha=0.2, color = col[i])
        ax.plot(d, (median_FL + 0.8*i), color = col[i])
        ax.set_ylabel("Flight level")
        ax.set_xlabel("Distance flown (nm)")   

        ## Lateral path
        ax = axes[0,1]
        lon = dfs[CI][0]['λ']
        lat = dfs[CI][0]['Φ']
        ax.plot(lon, lat, color = col[i])
        ax.set_xlim(-20,40)
        ax.set_ylim(25,60)

        ax.set_ylabel("latitude (deg)")
        ax.set_xlabel("longitude (deg)")

    
        ## True Airspeed
        ax = axes[0,2]
        TAS = [df.TAS.values for df in dfs[CI]]
        median_TAS = np.mean(TAS, axis=0)
        d = np.cumsum(dfs[CI][0].d2n.values)/1852
        ax.plot(d, median_TAS, color = col[i])
        ax.set_ylabel("True airspeed (m/s)")
        ax.set_xlabel("Distance flown (nm)")

    
        d = np.cumsum(dfs[CI][0].d2n.values)/1852
        m = np.array([df.m.values for df in dfs[CI]])

        ATR_CH4 = np.array([df.ch4.values for df in dfs[CI]])
        ATR_O3 = np.array([df.o3.values for df in dfs[CI]])

        ATR_CH4_seg = np.zeros(ATR_CH4.shape)
        ATR_O3_seg  = np.zeros(ATR_CH4.shape)
        fuel_burn = np.zeros(m.shape)
        for i1 in range(len(m[0, :])-1):
            fuel_burn[:, i1 + 1] = m[:, i1] - m[:, i1 + 1]
            ATR_CH4_seg[:, i1 + 1] = ATR_CH4[:, i1+1] - ATR_CH4[:, i1]
            ATR_O3_seg[:, i1 + 1] = ATR_O3[:, i1+1] - ATR_O3[:, i1]

        NOx_emiss_EI = np.array([df.nox_emission_c.values for df in dfs[CI]])
        NOx_emiss = NOx_emiss_EI * fuel_burn
        
        ## Fuel Consumption
        ax = axes[1,0]
        median_fuel_burn = np.median(fuel_burn, axis=0)
        ax.plot(d, median_fuel_burn, color = col[i])
        ax.fill_between(d, np.min(fuel_burn, axis=0), np.max(fuel_burn, axis=0), alpha=0.2, color = col[i])
        ax.set_xlabel("Distance flown (nm)")
        ax.set_ylabel('Fuel consumption (kg(fuel)/t)')



        ## NOx EI
        ax = axes[1,1]
        ax.plot(d, np.median(NOx_emiss_EI, axis=0), color = col[i])
        ax.fill_between(d, np.min(NOx_emiss_EI, axis=0), np.max(NOx_emiss_EI, axis=0), alpha=0.2, color = col[i])
        ax.set_ylabel("NOx emission index (g(NO2)/Kg(fuel))")
        ax.set_xlabel("Distance flown (nm)")


    
        ax = axes[2,0]
        d = np.cumsum(dfs[CI][0].d2n.values)/1852
        m = [df.n_contrail.values for df in dfs[CI]]
        median_m = np.median(m, axis=0)
        max_m = np.max(m, axis=0)
        min_m = np.min(m, axis=0)
        ax.fill_between(d, min_m, max_m, alpha=0.2, color = col[i])
        ax.plot(d, median_m, label=f"$\\alpha$ = {CIs[i]} USD/K", color = col[i])
        ax.set_ylabel("ATR of Contrails [K]")
        ax.legend(fontsize='small', ncol=1)


        ax = axes[2,1]
        d = np.cumsum(dfs[CI][0].d2n.values)/1852
        m = [df.ch4.values+df.o3.values for df in dfs[CI]]
        median_m = np.median(m, axis=0)
        max_m = np.max(m, axis=0)
        min_m = np.min(m, axis=0)
        ax.fill_between(d, min_m, max_m, alpha=0.2, color = col[i])
        ax.plot(d, median_m, color = col[i])
        ax.set_ylabel("ATR of NO$_{x}$ emis. [K]")

        ax = axes[1,2]
        d = np.cumsum(dfs[CI][0].d2n.values)/1852
        m = [df.h20.values for df in dfs[CI]]
        median_m = np.median(m, axis=0)
        max_m = np.max(m, axis=0)
        min_m = np.min(m, axis=0)
        ax.fill_between(d, min_m, max_m, alpha=0.2, color = col[i])
        ax.plot(d, median_m, color = col[i])
        ax.set_ylabel("ATR of Water Vapour [K]")
        ax.set_xlabel("Distance flown (nm)")


        ax = axes[2,2]

        d = np.cumsum(dfs[CI][0].d2n.values)/1852
        m = [df.n_contrail.values+df.o3.values+df.ch4.values+df.h20.values for df in dfs[CI]]
        median_m = np.median(m, axis=0)
        max_m = np.max(m, axis=0)
        min_m = np.min(m, axis=0)
        ax.fill_between(d, min_m, max_m, alpha=0.2, color = col[i])
        ax.plot(d, median_m, color = col[i])
        ax.set_ylabel("Total non-CO$_{2}$ ATR [K]")

    
    for ii in range (3):
        for jj in range (3):
            axes[ii,jj].grid()
    fig.savefig( path_save + 'profile.pdf', bbox_inches='tight')

    plt.close()


def generate_json (dfs, departure_time, CIs, org_sid, des_star,  path_save):
    
    objectives = {}
    Obj = {}
    optimized_trj = {}
    flight_data = {}
    flight_data ['ac_type'] = 'A320-214'
    flight_data ['Departure_time (EXP.)'] = str(departure_time)
    flight_data ['SID'] = org_sid
    flight_data ['STAR'] = des_star
    

    for i, CI in enumerate(CIs):
        Objectives = {}
        ens_trj = {}

        d = np.cumsum(dfs[CI][0].d2n.values)/1852
        lat = np.array([df.Φ.values for df in dfs[CI]])
        lon = np.array([df.λ.values for df in dfs[CI]])
        M = np.array([df.M.values for df in dfs[CI]])
        FL = np.array([df.FL.values for df in dfs[CI]])

        m = np.array([df.m.values for df in dfs[CI]])
        t = np.array([df.t.values for df in dfs[CI]])
        v_gs = np.array([df.gs.values for df in dfs[CI]])
        v_tas = np.array([df.TAS.values for df in dfs[CI]])


        ATR_Cont = np.array([df.n_contrail.values for df in dfs[CI]])
        ATR_H2O = np.array([df.h20.values for df in dfs[CI]])
        ATR_CH4 = np.array([df.ch4.values for df in dfs[CI]])
        ATR_O3 = np.array([df.o3.values for df in dfs[CI]])

        ATR_Cont_seg = np.zeros(ATR_Cont.shape)
        ATR_H2O_seg = np.zeros(ATR_Cont.shape)
        ATR_CH4_seg = np.zeros(ATR_Cont.shape)
        ATR_O3_seg  = np.zeros(ATR_Cont.shape)

        fuel_burn = np.zeros(m.shape)
        for i1 in range(len(m[0, :])-1):
            fuel_burn[:, i1 + 1] = m[:, i1] - m[:, i1 + 1]
            ATR_Cont_seg[:, i1 + 1] = ATR_Cont[:, i1+1] - ATR_Cont[:, i1]
            ATR_H2O_seg[:, i1 + 1] = ATR_H2O[:, i1+1] - ATR_H2O[:, i1]
            ATR_CH4_seg[:, i1 + 1] = ATR_CH4[:, i1+1] - ATR_CH4[:, i1]
            ATR_O3_seg[:, i1 + 1] = ATR_O3[:, i1+1] - ATR_O3[:, i1]

        ATR_CO2_seg = 10.1 * 6.94e-16 * fuel_burn
        ATR_CO2 = 10.1 * 6.94e-16 * (m[:,0]-m[:,-1])

        NOx_emiss_EI = np.array([df.nox_emission_c.values for df in dfs[CI]])
        NOx_emiss = NOx_emiss_EI * fuel_burn
        ATR_t = ATR_Cont_seg + ATR_H2O_seg + ATR_CH4_seg + ATR_O3_seg + ATR_CO2_seg
        ATR_t_final = ATR_Cont[:,-1] + ATR_H2O[:,-1] + ATR_CH4[:,-1] + ATR_O3[:,-1] + ATR_CO2

        ## Objectives
        ATR_f_min = np.min (ATR_t_final)
        ATR_f_mean= np.mean (ATR_t_final)
        ATR_f_max = np.max  (ATR_t_final)

        mass_cons = m[:,0]-m[:,-1]
        time_fli  = t[:,-1] - t[:,0]

        cost_min  = 1e-3 * (0.51 * np.min (mass_cons)  +  0.75 * np.min (time_fli))
        cost_mean = 1e-3 * (0.51 * np.mean (mass_cons) +  0.75 * np.mean (time_fli))
        cost_max  = 1e-3 * (0.51 * np.max (mass_cons)  +  0.75 * np.max (time_fli))

        Cost = {}
        Cost['unit'] = 'tUSD'
        Cost['min']  =  float(cost_min)
        Cost['mean'] =  float(cost_mean)
        Cost['max']  =  float(cost_max)
        ATR  = {}
        ATR ['unit']  = 'K'
        ATR ['min']  = float(ATR_f_min)
        ATR ['mean'] = float(ATR_f_mean)
        ATR ['max']  = float(ATR_f_max)

        Objectives ['ATR']  = ATR
        Objectives ['SOC'] = Cost

        f_plan = len (m[:,0])
        int_points = len (m[0,:])

        for j in range (f_plan):
                scenario = {}
                for k in range (int_points):
                            scenario['WP{}'.format(k)] = {'lat' : float(lat[j,k]),
                                                        'lon' : float(lon[j,k]),
                                                        'FL'  : float(FL[j,k]),
                                                        'distance' : float(d[k]),
                                                        'Mach': float(M[j,k]),
                                                        'mass': float (m[j,k]),
                                                        'fuel_cons.': float(fuel_burn[j,k]),
                                                        'true airspeed': float (v_tas[j,k]),
                                                        'groundspeed': float (v_gs[j,k]),
                                                        'ATR_Cont': float(ATR_Cont_seg[j,k]),
                                                        'ATR_H2O': float(ATR_H2O_seg[j,k]),
                                                        'ATR_CH4': float(ATR_CH4_seg[j,k]),
                                                        'ATR_NOx' : float (ATR_CH4_seg[j,k] + ATR_O3_seg[j,k]),
                                                        'ATR_O3': float(ATR_O3_seg[j,k]),
                                                        'ATR_CO2' : float(ATR_CO2_seg[j,k]),
                                                        'ATR_tot': float(ATR_t[j,k]),
                                                        'NOx_emission': float(NOx_emiss[j,k]),
                                                        'NOx_EI' : float (NOx_emiss_EI[j,k]),
                                                        'time': str(departure_time + t[j,k] * np.timedelta64(1, 's'))}
                ens_trj ['Opt_trajectory_{}'.format(j)] = scenario
        objectives    ['objectives_alpha_{}'.format(CI)] = Objectives
        optimized_trj ['Opt_trj_alpha_{}'.format(CI)]    = ens_trj

    optimized_trj ['Env_indices'] = CIs
    Obj['flight_data'] = flight_data
    Obj['objectives'] = objectives
    Obj['optimized_trajectories'] = optimized_trj
    path = path_save + 'optimized_profile'
    save_json_dict (path +'.json', Obj)
    
    return Obj
