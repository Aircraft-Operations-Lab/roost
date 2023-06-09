#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pytz
import datetime as dt
import numpy as np
import tempfile

import pycuda.driver as drv
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curand
from pycuda.compiler import SourceModule, DEFAULT_NVCC_FLAGS

# from scipy.interpolate import RectBivariateSpline

from roost.cudatools import *
import roost.codegen as codegen
import roost.airspace

d2r = np.pi / 180
r2d = 1 / d2r
nm = 1852.0

home = os.path.expanduser("~")
fdir = os.path.dirname(os.path.realpath(__file__))


class AttrDict(dict):
    pass

    # def __getattr__(self, attr_name):
    #     try:
    #         return self[attr_name]
    #     except KeyError as err:
    #         raise AttributeError(err)


class ComputationalConfig(AttrDict):
    def __init__(self):
        super().__init__()
        self.update(
            {
                "n_plans": 20,
                "n_scenarios": 32,
                "exploration_noise_std": 1,
                "top_performing": 10,
                "ars_version": "V1-t",
                "horizontal_resolution_nm": 10,
                "ars_step_size": 0.6,
                "nu_cr": 0.1,
                "nu_tas": 2.5 / 4,
                "nu_M": 0.003,
                "nu_fl": 5 / 4,
                "nu_d2g": 2,
                "nesterov_acceleration": True,
                "nesterov_velocity_factor": 0.5,
                "T_texture_offset": -150,
                "U_texture_offset": 200,
                "V_texture_offset": 200,
                "cr_execution_noise_mode": "common",
                "FL_penalty_coeff": 0.01,
            }
        )


class ProblemConfig(AttrDict):
    def __init__(self):
        super().__init__()
        self.update(
            {
                "perform_descent": True,
                "initial_phase": "CLIMB",
                "origin": "PINAR",
                "destination": "MILGU",
                "FL0_cruise": 200,  # 200, ##300,
                "FL0": 50,
                "dateref": dt.datetime(2018, 5, 21, 3, tzinfo=pytz.utc),
                "m0": 61600,
                "tas0": 120,
                "tas0_cruise": 220,
                "M0_cruise": 0.73,
                "FL_min": 200,  # 300,
                "FL_max": 410,  # 360,
                "FL_STAR": 100.0,
                "tas_low_FL_min": 160,
                "tas_high_FL_min": 245,
                "tas_low_FL_max": 182,
                "tas_high_FL_max": 232,
                "M_low_FL_min": 0.6,
                "M_high_FL_min": 0.82,
                "M_low_FL_max": 0.68,
                "M_high_FL_max": 0.82,
                "CI": 0.5,
                "gamma_max": 0.15,
                "gamma_min": -0.1,
                "FL_change_threshold_nm": 200,
                "FL_change_variability_nm": 50,
                "tas_regularization_distance_nm": 45,
                "FL_change_regularization_distance_nm": 100,
                "M_guidance_regularization_nm": 50,
                "M_change_threshold_nm": 260,
                "M_change_variability_nm": 10,
                "M_change_climb_offset": 0,
                "m0_stdev": 0,  # 10,
                "t0_stdev": 0,  # 10,
                # 'n_members': 32,
                "climb_descent_profile_coeffs": 8,
                "top_cd_profile": 45000,
                "local_d2g_delta": 15,
                "d2gdesc_min": 50,
                "d2gdesc_max": 150,
                "local_d2g_delta": 15,
                "compute_accf": False,
                "compute_emissions": False,
                "climate_index": 0,
                "emission_index": 0,
                "cost_index": 1,
                "CI_fuel": 1,
                "CI_nox": 1,
                "CI_co2": 1,
                "CI_contrail": 1,
                "CI_h2o": 1,
                "CI_contrail_dis": 0,
                "DI_contrail": 0,
                "nCon_index": 1,
                "dCon_index": 0,
                "avoid_negative_thrust": True,
            }
        )


class CUDAFunction(object):
    def __init__(self, fcn, grid, block, args):
        self.fcn = fcn
        self.grid = grid
        self.block = block
        self.fcn.prepare(args)

    def __call__(self, *args):
        self.fcn.prepared_call(self.grid, self.block, *args)


class StructuredFlightPlanningProblem(object):
    @classmethod
    def init_from_nav_network(
        cls,
        apm,
        weather_cache,
        navigation_network: roost.airspace.NavigationNetwork,
        setup=None,
        cconfig=None,
    ):
        if cconfig is None:
            cconfig = ComputationalConfig()
        if setup is None:
            setup = ProblemConfig()
        route_graph = cls.get_graph(navigation_network, setup)
        return cls(apm, weather_cache, route_graph, setup, cconfig)

    def __init__(
        self,
        apm,
        weather_cache,
        route_graph: roost.airspace.RouteGraph,
        setup=None,
        cconfig=None,
    ):
        if cconfig is None:
            cconfig = ComputationalConfig()
        if setup is None:
            setup = ProblemConfig()
        self.apm = apm
        self.wc = weather_cache
        self.ccfg = cconfig
        self.pcfg = setup
        self.route_graph = route_graph
        self.fpe = self.route_graph.generate_flight_plan_encoding(
            setup["origin"],
            setup["destination"],
            cconfig["horizontal_resolution_nm"],
            climb_descent_coefficients=setup["climb_descent_profile_coeffs"],
        )
        blended_dict = self.fpe.get_constants_dictionary()
        blended_dict.update(self.ccfg)
        blended_dict.update(self.pcfg)
        blended_dict.update(self.wc.get_constants_dictionary())
        n_plans = self.ccfg["n_plans"]
        n_scenarios = self.ccfg["n_scenarios"]
        n_fp_vars = blended_dict["n_flight_plan_vars"]
        assert not n_plans % 2
        n_directions = int(n_plans / 2)
        blended_dict["n_directions"] = n_directions
        n_directions_c2 = get_next_power_of_two(n_directions)
        n_plans_c2 = get_next_power_of_two(n_plans)
        blended_dict["n_directions_c2"] = n_directions_c2
        blended_dict["n_plans_c2"] = n_directions_c2
        breadth = blended_dict["fp_vars_sweep_breadth"]
        max_block = 1024
        blended_dict["fp_vars_per_thread"] = vars_per_thread = int(
            np.ceil(n_fp_vars * 2 / max_block)
        )
        rollout_threads = int(np.ceil(n_fp_vars / vars_per_thread))
        blended_dict["aircraft"] = apm.ac_label
        blended_dict["apm"] = apm
        # Because of limited 32-bit float precision, we reference time with respect to
        # the beginning of the weather forecast to prevent underflows.
        blended_dict["t0"] = (
            blended_dict["departure_time"] - self.wc.start_time
        ) / np.timedelta64(1, "s")
        self.src = codegen.get_pptp_source(**blended_dict)

        tmphandle, tmppath = tempfile.mkstemp(text=True, suffix="cu")
        with os.fdopen(tmphandle, "w") as tmpfile:
            tmpfile.write(self.src)
        print(f"Saving formatted CUDA code at {tmppath}")
        flags = ["-lineinfo", "--generate-line-info"]
        try:
            self.module = SourceModule(
                self.src,
                include_dirs=[os.path.join(fdir, "cuda/")],
                no_extern_c=True,
                options=DEFAULT_NVCC_FLAGS + flags,
            )
        except pycuda.driver.CompileError:
            raise

        # Get CUDA functions ------------------------------------------

        self.update_fp_vars = self.get_function(
            "update_fp_vars", (n_fp_vars, 1), (n_directions_c2, 1, 1), "PPPPPff"
        )

        self.get_objectives = self.get_function(
            "get_objectives", (n_plans, 1), (n_scenarios, 1, 1), "PPff"
        )

        self.average_objectives = self.get_function(
            "objectives_mean", (1, 1), (n_plans_c2, 1, 1), "PP"
        )

        self.clamp_fp_vars = self.get_function(
            "clamp_fp_vars",
            (breadth // max_block + 1, 1),
            (min(breadth, max_block), 1, 1),
            "PP",
        )

        self.initialize_fp_vars = self.get_function(
            "initialize_fp_vars",
            (n_fp_vars // max_block + 1, 1),
            (max_block, 1, 1),
            "PP",
        )

        self.calc_objectives_stdev = self.get_function(
            "objectives_stdev", (1, 1), (n_plans_c2, 1, 1), "PP"
        )

        self.generate_rollouts = self.get_function(
            "generate_rollouts", (n_directions, 1), (rollout_threads, 2, 1), "PPPPPff"
        )

        self.clear_guidance = self.get_function(
            "clear_guidance", (n_plans, 1), (4, n_scenarios, 1), "PP"
        )

        self.clear_profiles = self.get_function(
            "clear_profiles", (n_plans, 36), (n_scenarios, 1, 1), "P"
        )

        self.generate_guidance_matrix = self.get_function(
            "generate_guidance_matrix", (n_plans, 1), (n_scenarios, 1, 1), "PPPPPPPP"
        )

        self.generate_4D_profile = self.get_function(
            "generate_4D_profile", (n_plans, 1), (n_scenarios, 1, 1), "PPPPPP"
        )

        self.integrate_profile = self.get_function(
            "integrate_profile", (n_plans, 1), (n_scenarios, 1, 1), "PPPPi"
        )

        self.create_climb_descent = self.get_function(
            "create_cd_curves",
            (n_plans, 1),
            (setup["climb_descent_profile_coeffs"], 2, 2),
            "PP",
        )
        #
        self.evaluate_weather_grib = self.get_function(
            "evaluate_weather_grib_1block", (1, 1), (32, 32, 1), "PPffP"
        )
        self.evaluate_tex = self.get_function(
            "evaluate_tex", (1, 1), (32, 1, 1), "ifffP"
        )

        # Create arrays --------------------------------------------

        def float_array(*shape):
            return gpuarray.GPUArray(shape=shape, dtype=np.float32)

        def int_array(*shape):
            return gpuarray.GPUArray(shape=shape, dtype=np.int32)

        self.objectives_means = float_array(n_plans)
        self.objectives_mean_and_std = float_array(2)
        self.exploration_samples = float_array(n_directions, n_fp_vars)
        self.flight_plan_variables = float_array(n_fp_vars)
        self.velocity = float_array(n_fp_vars)
        self.velocity.fill(0.0)
        self.decision_variables = float_array(n_plans, n_fp_vars)
        self.summaries = float_array(n_plans, n_scenarios, 10)
        self.nn2e = a2gpu(self.fpe.nn2e, np.int32)
        self.cr_table = a2gpu(self.fpe.crt, np.int32)
        self.edge_lengths = a2gpu(self.fpe.e_len)
        self.guidance_int = int_array(
            n_plans, self.fpe.max_nodes_per_path, n_scenarios, 4
        )
        self.guidance_float = float_array(
            n_plans, self.fpe.max_nodes_per_path, n_scenarios, 4
        )
        self.edge_n_points = a2gpu(self.fpe.edge_n_points, np.int32)
        self.profiles = float_array(
            n_plans, self.fpe.max_subnodes_per_path, 36, n_scenarios
        )
        self.edge_subnode_coordinates = a2gpu(self.fpe.ep)
        self.climbdescent_curves = float_array(
            n_plans, 2, 2, self.pcfg["climb_descent_profile_coeffs"]
        )
        upper_lims = 175 + 50 / 8 * np.array(range(8))
        self.vlims = a2gpu(
            np.array(
                [[90.0, 102.85, 112.86, 127.8, 152.8, 175.57, 198.7, 205.0], upper_lims]
            )
        )

        # Transfer weather arrays --------------------------------------

        self.wc.bind_textures_to_module(self.module)
        # self.weather_arrays_gpu = {}
        # self.weather_tex = {}
        #
        # # arrs_list = [(weather_arrays.u, weather_arrays.v, weather_arrays.T,),
        # #              (weather_arrays_lower.u, weather_arrays_lower.v, weather_arrays_lower.T,)]
        #
        # # arrs = (weather_arrays.u, weather_arrays.v, weather_arrays.T, )
        # var_codes = [('U', 'V', 'T'), ('U_lower', 'V_lower', 'T_lower')]
        #
        # for arrs, variable_codes in zip(arrs_list, var_codes):
        #     for arr, var_name in zip(arrs, variable_codes):
        #         for i in range(self.pcfg['n_members']):
        #             # The offset seeks to move the variables to a positive range
        #             # The reason is that, apparently, the CUDA trilinear filter for 3D texture fetches
        #             # may overflow when interpolating between values of different sign.
        #             # Interpolating between same sign values (thanks to the offset)
        #             # represents, thus, a useful workaround.
        #             tex_ref = f"{var_name}_{i}"
        #             offset = self.ccfg[tex_ref[:2] + 'texture_offset']
        #             tex_array = flip_weather_array_for_texture(arr[:, :, i, :, :].astype(np.float32) + offset)
        #             self.weather_arrays_gpu[tex_ref] = np_to_array_4d_f4(tex_array)
        #             self.weather_tex[tex_ref] = self.module.get_texref(tex_ref)
        #             self.weather_tex[tex_ref].set_array(self.weather_arrays_gpu[tex_ref])
        #             self.weather_tex[tex_ref].set_flags(drv.TRSF_NORMALIZED_COORDINATES)
        #             self.weather_tex[tex_ref].set_filter_mode(drv.filter_mode.LINEAR)  # LINEAR)
        #             self.weather_tex[tex_ref].set_address_mode(0, drv.address_mode.CLAMP)
        #             self.weather_tex[tex_ref].set_address_mode(1, drv.address_mode.CLAMP)
        #             self.weather_tex[tex_ref].set_address_mode(2, drv.address_mode.CLAMP)

        # Create the RNG and random arrays --------------------------------

        self.rng = curand.MRG32k3aRandomNumberGenerator()
        self.exploration_noise_shape = (n_directions, n_fp_vars)
        self.execution_noise_shape = (
            n_plans,
            n_scenarios,
            blended_dict["n_random_vars"],
        )
        self.initial_conditions_noise_shape = (n_plans, n_scenarios, 2)
        self.exploration_noise = float_array(*self.exploration_noise_shape)
        self.execution_noise = float_array(*self.execution_noise_shape)
        self.ic_noise = float_array(*self.initial_conditions_noise_shape)

    @staticmethod
    def get_graph(navigation_network, problem_setup):
        navigation_network = navigation_network.filter_route_type("AR")
        origin_coords = navigation_network.get_point_coords(problem_setup["origin"])
        destin_coords = navigation_network.get_point_coords(
            problem_setup["destination"]
        )
        od_v = destin_coords - origin_coords
        navigation_network = navigation_network.filter_geo(
            [origin_coords - 0.05 * od_v, destin_coords + 0.05 * od_v]
        )
        navigation_network = navigation_network.filter_connected()
        rg = navigation_network.create_graph()
        return rg.reduce_k_shortest_paths(
            problem_setup["origin"], problem_setup["destination"]
        )

    def get_function(self, func_name, grid, block, args):
        print(func_name, grid, block)
        return CUDAFunction(self.module.get_function(func_name), grid, block, args)

    def generate_exploration_noise(self):
        self.rng.fill_normal(self.exploration_noise)

    def generate_execution_noise(self):
        self.rng.fill_uniform(self.execution_noise)
        self.rng.fill_normal(self.ic_noise)

    def execute_flight_plans(
        self, i, execution_noise_scaling=1, write_profile=np.int32(0)
    ):
        self.generate_execution_noise()
        if execution_noise_scaling:
            self.execution_noise = 0.5 + execution_noise_scaling * (
                self.execution_noise - 0.5
            )
        else:
            self.execution_noise.fill(np.float32(0.5))
        # self.guidance_int.fill(0)
        # self.guidance_float.fill(0.0)
        # self.profiles.fill(0.0)
        self.clear_guidance(self.guidance_int.gpudata, self.guidance_float.gpudata)
        self.clear_profiles(self.profiles.gpudata)
        self.generate_guidance_matrix(
            self.decision_variables.gpudata,
            self.execution_noise.gpudata,
            self.nn2e.gpudata,
            self.cr_table.gpudata,
            self.edge_lengths.gpudata,
            self.edge_n_points.gpudata,
            self.guidance_int.gpudata,
            self.guidance_float.gpudata,
        )
        self.generate_4D_profile(
            self.guidance_int.gpudata,
            self.execution_noise.gpudata,
            self.edge_n_points.gpudata,
            self.guidance_float.gpudata,
            self.edge_subnode_coordinates.gpudata,
            self.profiles.gpudata,
        )
        self.create_climb_descent(
            self.decision_variables.gpudata, self.climbdescent_curves.gpudata
        )
        self.integrate_profile(
            self.profiles.gpudata,
            self.climbdescent_curves.gpudata,
            self.ic_noise.gpudata,
            self.summaries.gpudata,
            write_profile,
        )
        #                        # i)

    def sample_new_decision_variables(self, noise_scaling=1.0, beta_velocity=None):
        if beta_velocity is None:
            beta_velocity = self.ccfg["nesterov_velocity_factor"]
        self.generate_exploration_noise()
        self.generate_rollouts(
            self.flight_plan_variables.gpudata,
            self.exploration_noise.gpudata,
            self.decision_variables.gpudata,
            self.velocity.gpudata,
            self.vlims.gpudata,
            beta_velocity,
            noise_scaling,
        )

    def compute_objectives(self):
        self.get_objectives(
            self.summaries.gpudata,
            self.objectives_means.gpudata,
            np.float32(self.pcfg["CI"]),
            np.float32(self.pcfg["climate_index"]),
        )
        self.average_objectives(
            self.objectives_means.gpudata, self.objectives_mean_and_std.gpudata
        )
        self.calc_objectives_stdev(
            self.objectives_means.gpudata, self.objectives_mean_and_std.gpudata
        )

    def update_decision_variables(self, ars_factor=1.0):
        self.update_fp_vars(
            self.objectives_means.gpudata,
            self.exploration_noise.gpudata,
            self.objectives_mean_and_std.gpudata,
            self.flight_plan_variables.gpudata,
            self.velocity.gpudata,
            self.ccfg["ars_step_size"] * ars_factor,
            self.ccfg["nesterov_velocity_factor"],
        )
        self.clamp_fp_vars(self.flight_plan_variables.gpudata, self.vlims.gpudata)

    def timed_run(
        self,
        N=500,
        initialize=True,
        noise_scaling=4,
        blank_steps=50,
        execution_noise_factor=1,
        seq_offset=1000,
        explo_exp=0.5,
        exec_exp=0.5,
    ):
        import time

        if initialize:
            self.initialize_fp_vars(
                self.flight_plan_variables.gpudata, self.vlims.gpudata
            )
        J = []
        t = []
        nsc = noise_scaling
        self.mean_stdev = np.zeros((N, 2))
        # seq_offset = 5000
        for i in range(N):
            f = seq_offset / (seq_offset + i)
            execution_noise_scaling = execution_noise_factor * f**exec_exp
            if i <= N - blank_steps:
                self.sample_new_decision_variables(nsc * f**explo_exp)
            else:
                self.sample_new_decision_variables(0.0, 0.0)
                execution_noise_scaling = 0
            self.execute_flight_plans(i, execution_noise_scaling)
            self.compute_objectives()
            self.mean_stdev[i, :] = self.objectives_mean_and_std.get()
            if i <= N - blank_steps:
                if i > 0:
                    if self.mean_stdev[i, 0] > self.mean_stdev[i - 1, 0]:
                        self.velocity.fill(0.0)
                self.update_decision_variables(f)
            if i == 0:
                t0 = time.time()
            elif i > 0:
                ti = time.time()
                t.append(ti)
                J.append(self.objectives_means.get())

        t2 = time.time()
        self.sample_new_decision_variables(0.0, 0.0)
        print("Sample: ", time.time() - t2)
        t2 = time.time()
        self.execute_flight_plans(i, 0, write_profile=np.int32(1))
        print("Exec: ", time.time() - t2)
        t2 = time.time()
        self.compute_objectives()
        print("Obj comp: ", time.time() - t2)
        t2 = time.time()
        self.mean_stdev[i, :] = self.objectives_mean_and_std.get()
        self.update_decision_variables(f)
        print("Update: ", time.time() - t2)

        return J, [t_ - t0 for t_ in t]

    def get_profiles_dataframes(self, clip_to_first_scenario=True):
        import pandas as pd

        if clip_to_first_scenario:
            dfs = [
                pd.DataFrame(
                    data=self.profiles.get()[i, :, :35, 0],
                    columns=[
                        "Φ",
                        "λ",
                        "v_T",
                        "FL_T",
                        "χ",
                        "d2n",
                        "m",
                        "rho",
                        "FL",
                        "γ",
                        "TAS",
                        "TASn",
                        "gs",
                        "gsn",
                        "M",
                        "Thr",
                        "D",
                        "t",
                        "Δt",
                        "tn",
                        "fc",
                        "fcn",
                        "a",
                        "p",
                        "Mg",
                        "CT",
                        "eta",
                        "n_contrail",
                        "ch4",
                        "h20",
                        "o3",
                        "contrail_c",
                        "nox_emission",
                        "nox_emission_c",
                        "contrail_dis",
                    ],
                )
                for i in range(self.ccfg["n_plans"])
            ]

            return [dfs[i][dfs[i]["v_T"] > 0] for i in range(self.ccfg["n_plans"])]
        else:
            dfs = [
                [
                    pd.DataFrame(
                        data=self.profiles.get()[i, :, :35, j],
                        columns=[
                            "Φ",
                            "λ",
                            "v_T",
                            "FL_T",
                            "χ",
                            "d2n",
                            "m",
                            "rho",
                            "FL",
                            "γ",
                            "TAS",
                            "TASn",
                            "gs",
                            "gsn",
                            "M",
                            "Thr",
                            "D",
                            "t",
                            "Δt",
                            "tn",
                            "fc",
                            "fcn",
                            "a",
                            "p",
                            "Mg",
                            "CT",
                            "eta",
                            "n_contrail",
                            "ch4",
                            "h20",
                            "o3",
                            "contrail_c",
                            "nox_emission",
                            "nox_emission_c",
                            "contrail_dis",
                        ],
                    )
                    for j in range(self.ccfg["n_scenarios"])
                ]
                for i in range(self.ccfg["n_plans"])
            ]

            return [
                [
                    dfs[i][j][dfs[i][j]["v_T"] > 0]
                    for j in range(self.ccfg["n_scenarios"])
                ]
                for i in range(self.ccfg["n_plans"])
            ]

    def test_one_iterate(self, init=True):
        if init:
            self.initialize_fp_vars(
                self.flight_plan_variables.gpudata, self.vlims.gpudata
            )
        print("Flight Plan Variables\n=======")
        print(self.flight_plan_variables.get())
        self.sample_new_decision_variables()
        print("Exploration Noise\n=======")
        print(self.exploration_noise.get())
        print("Decision variables\n=======")
        print(self.decision_variables.get())
        self.execute_flight_plans(0)
        print("Guidance Int\n=========")
        print(self.guidance_int.get()[0, :10, 0, :])
        print("Guidance Float\n=========")
        print(self.guidance_float.get()[0, :, 0, :])
        print("Profiles\n=========")
        for k in [0]:  # (0, 1):
            print("....")
            for j in [0]:  # range(self.ccfg['n_scenarios']):
                prf = self.profiles.get()[k, :, :16, j]
                for i in range(self.profiles.shape[1]):
                    row = prf[i, :]
                print("---")
        s = self.summaries.get()
        print("Times\n=========")
        print(s[:, :, 1])
        print("Burns\n=========")
        print(s[:, :, 0])
        self.update_decision_variables()
        print("Objectives means\n=========")
        print(self.objectives_means.get())
        print("Objectives mean and std\n=========")
        print(self.objectives_mean_and_std.get())
        print("New vars \n=============")
        print(self.flight_plan_variables.get())

    def execute_one_flight_plan(
        self, m_list, h_list, execution_noise_scaling=1, write_profile=np.int32(0)
    ):
        self.generate_execution_noise()
        self.execution_noise = 0.5 + execution_noise_scaling * (
            self.execution_noise - 0.5
        )
        # self.guidance_int.fill(0)
        # self.guidance_float.fill(0.0)
        # self.profiles.fill(0.0)
        self.generate_guidance_matrix(
            self.decision_variables.gpudata,
            self.execution_noise.gpudata,
            self.nn2e.gpudata,
            self.cr_table.gpudata,
            self.edge_lengths.gpudata,
            self.edge_n_points.gpudata,
            self.guidance_int.gpudata,
            self.guidance_float.gpudata,
        )

        # Matrix to CPU
        guidance_int_cpu = self.guidance_int.get()
        # Overwrite guidance_int matrix
        it = 0
        for m, h in zip(m_list, h_list):
            guidance_int_cpu[:, it, :, 1] = m
            guidance_int_cpu[:, it, :, 2] = h
            it += 1

        # Matrix to GPU
        guidance_int_new = gpuarray.to_gpu(guidance_int_cpu)

        self.generate_4D_profile(
            guidance_int_new.gpudata,
            self.execution_noise.gpudata,
            self.edge_n_points.gpudata,
            self.guidance_float.gpudata,
            self.edge_subnode_coordinates.gpudata,
            self.profiles.gpudata,
        )
        self.create_climb_descent(
            self.decision_variables.gpudata, self.climbdescent_curves.gpudata
        )
        self.integrate_profile(
            self.profiles.gpudata,
            self.climbdescent_curves.gpudata,
            self.ic_noise.gpudata,
            self.summaries.gpudata,
            write_profile,
        )

    def test_one_iterate_new(self, m_list, h_list, init=True):
        if init:
            self.initialize_fp_vars(
                self.flight_plan_variables.gpudata, self.vlims.gpudata
            )
        print("Flight Plan Variables\n=======")
        print(self.flight_plan_variables.get())
        self.sample_new_decision_variables()
        print("Exploration Noise\n=======")
        print(self.exploration_noise.get())
        print("Decision variables\n=======")
        print(self.decision_variables.get())
        self.execute_one_flight_plan(m_list, h_list)
        print("Guidance Int\n=========")
        print(self.guidance_int.get()[0, :10, 0, :])
        print("Guidance Float\n=========")
        print(self.guidance_float.get()[0, :, 0, :])
        print("Profiles\n=========")
        for k in [0]:  # (0, 1):
            print("....")
            for j in [0]:  # range(self.ccfg['n_scenarios']):
                prf = self.profiles.get()[k, :, :16, j]
                for i in range(self.profiles.shape[1]):
                    row = prf[i, :]
                    print(row)
                print("---")
        s = self.summaries.get()
        print("Times\n=========")
        print(s[:, :, 1])
        print("Burns\n=========")
        print(s[:, :, 0])
        self.update_decision_variables()
        print("Objectives means\n=========")
        print(self.objectives_means.get())
        print("Objectives mean and std\n=========")
        print(self.objectives_mean_and_std.get())
        print("New vars \n=============")
        print(self.flight_plan_variables.get())
