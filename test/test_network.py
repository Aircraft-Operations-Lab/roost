import unittest
from roost.airspace import *
from pathlib import Path

test_dir = Path().absolute()


class TestNetwork(unittest.TestCase):
    def test_read_network_plot(self):
        csv_path = test_dir / 'samples' / 'network_jun2018.csv'
        nw = NavigationNetwork.init_from_csv(csv_path)
        nw.plot()

    def test_create_graph(self):
        csv_path = test_dir / 'samples' / 'network_jun2018.csv'
        nw = NavigationNetwork.init_from_csv(csv_path)
        nw = nw.filter_geo([(40, -2.6), (52.5, 13.4)])
        nw = nw.filter_route_type('AR')
        nw = nw.filter_connected()
        route_net = nw.create_graph()
        N = 5000
        rnr = route_net.reduce_k_shortest_paths('PINAR', 'MILGU', 5000)
        rnr.save_to_gexf(test_dir / 'samples' / f'p2m_{N}.gexf')

    def test_create_graph_from_path(self):
        path = ['PINAR', 'BRITO', 'LARDA', 'RONNY', 'TOPTU', 'TOU', 'GAI', 'MEN', 'MINDI',
                'VEROT', 'MURRO', 'LSE', 'DEPUL', 'ARGIS', 'TOKDO', 'PAS', 'GVA', 'SPR',
                'REVLI', 'FRI', 'WIL', 'ZUE', 'SONOM', 'NELLI', 'NOTGA', 'LAMPU', 'NIKUT',
                'TOSTU', 'RASPU', 'GUDOM', 'GORKO', 'BOKNI', 'PILAM', 'VAGAB', 'BAMKI',
                'NENAN', 'ERF', 'WEMAR', 'GALMA', 'OSKAT', 'TADUV', 'MILGU']
        csv_path = test_dir / 'samples' / 'network_jun2018.csv'
        nw = NavigationNetwork.init_from_csv(csv_path)
        route_net = nw.create_graph_from_path(path)
        route_net.draw()
        route_net.save_to_gexf(test_dir / 'samples' / f'example_path.gexf')

    def test_reduce_d_factor(self):
        csv_path = test_dir / 'samples' / 'network_jun2018.csv'
        nw = NavigationNetwork.init_from_csv(csv_path)
        nw = nw.filter_geo([(40, -2.6), (52.5, 13.4)])
        nw = nw.filter_route_type('AR')
        nw = nw.filter_connected()
        route_net = nw.create_graph()
        d = 1.025
        rnr = route_net.reduce_d_factor('PINAR', 'MILGU', d)
        rnr.draw()
        rnr.save_to_gexf(test_dir / 'samples' / f'p2m_d{d}.gexf')


class TestRouteGraph(unittest.TestCase):

    def test_plot(self):
        rnr = RouteGraph.init_from_gexf(test_dir / 'samples' / 'p2m_5000.gexf')
        rnr.draw(show_shortest_path=('PINAR', 'MILGU'), show_node_ids=True)

    def test_plot_generated_graph(self):
        rnr = RouteGraph.init_from_gexf(test_dir / 'samples' / 'pinar_to_milgu.gexf')
        rnr.draw(show_node_ids=True)


if __name__ == '__main__':
    unittest.main()
