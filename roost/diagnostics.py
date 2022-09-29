#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from pycuda import gpuarray
from .pptp import StructuredFlightPlanningProblem


def check_arrays_for_nan(*arrays):
    for arr in arrays:
        assert not np.isnan(gpuarray.sum(arr).get())


class SFPPNanCheck(StructuredFlightPlanningProblem):

    def execute_flight_plans(self, i):
        self.generate_execution_noise()
        self.guidance_int.fill(0)
        self.guidance_float.fill(0.0)
        self.profiles.fill(0.0)
        self.generate_guidance_matrix(self.decision_variables.gpudata,
                                      self.execution_noise.gpudata,
                                      self.nn2e.gpudata,
                                      self.cr_table.gpudata,
                                      self.edge_lengths.gpudata,
                                      self.edge_n_points.gpudata,
                                      self.guidance_int.gpudata,
                                      self.guidance_float.gpudata)
        check_arrays_for_nan(self.decision_variables)
        check_arrays_for_nan(self.execution_noise)
        check_arrays_for_nan(self.guidance_int)
        check_arrays_for_nan(self.guidance_float)
        self.generate_4D_profile(self.guidance_int.gpudata,
                                 self.execution_noise.gpudata,
                                 self.edge_n_points.gpudata,
                                 self.guidance_float.gpudata,
                                 self.edge_subnode_coordinates.gpudata,
                                 self.profiles.gpudata)
        check_arrays_for_nan(self.profiles)
        self.create_climb_descent(self.decision_variables.gpudata,
                                  self.climbdescent_curves.gpudata)
        check_arrays_for_nan(self.climbdescent_curves)
        self.integrate_profile(self.profiles.gpudata,
                               self.climbdescent_curves.gpudata,
                               self.ic_noise.gpudata,
                               self.summaries.gpudata)
        check_arrays_for_nan(self.profiles)
        check_arrays_for_nan(self.ic_noise)
        check_arrays_for_nan(self.summaries)
        #                        # i)

    def sample_new_decision_variables(self, noise_scaling=1, beta_velocity=None):
        if beta_velocity is None:
            beta_velocity = self.ccfg['nesterov_velocity_factor']
        self.generate_exploration_noise()
        check_arrays_for_nan(self.exploration_noise)

        self.generate_rollouts(self.flight_plan_variables.gpudata,
                               self.exploration_noise.gpudata,
                               self.decision_variables.gpudata,
                               self.velocity.gpudata,
                               self.vlims.gpudata,
                               beta_velocity,
                               noise_scaling)
        check_arrays_for_nan(self.decision_variables)
        check_arrays_for_nan(self.flight_plan_variables)
        check_arrays_for_nan(self.decision_variables)
        check_arrays_for_nan(self.velocity)

    def update_decision_variables(self, ars_factor=1, beta_velocity=None):
        if beta_velocity is None:
            beta_velocity = self.ccfg['nesterov_velocity_factor']
        self.get_objectives(self.summaries.gpudata,
                            self.objectives_means.gpudata,
                            np.float32(self.pcfg['CI']))
        check_arrays_for_nan(self.summaries)
        check_arrays_for_nan(self.objectives_means)
        self.average_objectives(self.objectives_means.gpudata,
                                self.objectives_mean_and_std.gpudata)
        check_arrays_for_nan(self.objectives_mean_and_std)
        self.calc_objectives_stdev(self.objectives_means.gpudata,
                                   self.objectives_mean_and_std.gpudata)
        check_arrays_for_nan(self.objectives_mean_and_std)
        self.update_fp_vars(self.objectives_means.gpudata,
                            self.exploration_noise.gpudata,
                            self.objectives_mean_and_std.gpudata,
                            self.flight_plan_variables.gpudata,
                            self.velocity.gpudata,
                            self.ccfg['ars_step_size'] * ars_factor,
                            beta_velocity)
        check_arrays_for_nan(self.flight_plan_variables)
        check_arrays_for_nan(self.velocity)
        self.clamp_fp_vars(self.flight_plan_variables.gpudata,
                           self.vlims.gpudata)
        check_arrays_for_nan(self.flight_plan_variables)


class SFPPWithExtraOutput(StructuredFlightPlanningProblem):

    def timed_run_with_directions(self, N=500, initialize=True, noise_scaling=4, ars_factor=1,
                                  blank_steps=50, ex_factor=0.1, directions_at=None):
        import time
        if initialize:
            self.initialize_fp_vars(self.flight_plan_variables.gpudata, self.vlims.gpudata)
        if directions_at is None:
            directions_at = range(N)
            #directions_at = [1, 10, 50, 100, 1000, 2000]
        J = []
        t = []
        directions = []
        nsc = noise_scaling
        rel_step_size = 1
        execution_noise_factor = 1
        step_size_factor = ex_factor ** (2 / (N))
        for i in range(N):
            if i in directions_at:
                directions.append(self.velocity.get())
            if i <= N - blank_steps:
                self.sample_new_decision_variables(nsc)
            else:
                self.sample_new_decision_variables(0.0, 0.0)
            self.execute_flight_plans(i, execution_noise_factor)
            if i <= N - blank_steps:
                self.update_decision_variables(ars_factor)
            if i == 0:
                t0 = time.time()
            elif i > 0:
                if i > N / 2:
                    execution_noise_factor *= step_size_factor
                rel_step_size *= step_size_factor

                ti = time.time()
                t.append(ti)
                J.append(self.objectives_means.get())
        return J, [t_ - t0 for t_ in t], directions_at, directions


def cosine_similarity(A, B):
    return (A @ B) / (np.linalg.norm(A, 2) * np.linalg.norm(B, 2))
