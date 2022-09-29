import unittest
from roost.airspace import *
from roost.pptp import *
from roc3.bada4 import BADA4_jet_CR
from roc3.weather import WeatherStore_4D
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

test_dir = Path().absolute()


class TestNetwork(unittest.TestCase):

    def test_create_sfpp(self):
        route_graph = RouteGraph.init_from_gexf(test_dir / 'samples' / 'pinar_to_milgu.gexf')
        apm = BADA4_jet_CR('A320-231')
        ws = WeatherStore_4D(test_dir / 'samples' / 'weather_data-wu-2018-06-07_12.npz', ll_resolution=4.0)
        wa = ws.get_weather_arrays()
        problem_config = ProblemConfig()
        problem_config['t0'] = wa.get_constants_dictionary()['times_min'] + 1800
        ccfg = ComputationalConfig()
        one_iterate = False
        if one_iterate:
            ccfg['n_scenarios'] = 4
            ccfg['n_plans'] = 2
        sfpp = StructuredFlightPlanningProblem(apm, wa, route_graph,
                                               setup=problem_config,
                                               cconfig=ccfg)
        if one_iterate:
            sfpp.test_one_iterate()
        else:
            J, t = sfpp.timed_run(3000)
            Jarr = np.array(J)
            J_mean = Jarr.mean(axis=1)

            plt.plot(np.array(t)*1e3, J_mean, '.-')
            plt.plot(np.array(t) * 1e3, Jarr.min(axis=1), '--')
            plt.plot(np.array(t) * 1e3,  Jarr.max(axis=1), '--')
            plt.xlabel("Time (ms)")
            plt.ylabel("Objective (Kg)")
            plt.show()
            sfpp.test_one_iterate(False)


if __name__ == '__main__':
    unittest.main()
