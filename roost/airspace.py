#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from typing import Sequence
from roost.geometry import *
import networkx as nx
import matplotlib.pyplot as plt
import networkx.readwrite as rw
from networkx.algorithms import shortest_simple_paths, shortest_path, shortest_path_length
from networkx.exception import NetworkXNoPath


class FlightPlanEncoding(object):

    def __init__(self, origin, destination, n_nodes, n_edges, node_indexes, node_index_table, node_coordinates,
                 nn2e, e2nn, edge_lengths, edge_n_points, edge_points, crossroad_count, max_nodes_per_path,
                 max_subnodes_per_path, climb_descent_coefficients):
        self.o = origin
        self.d = destination
        self.ni = node_indexes  
        self.crt = node_index_table
        self.nll = node_coordinates
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.nn2e = nn2e
        self.e2nn = e2nn
        self.e_len = edge_lengths
        self.edge_n_points = edge_n_points
        self.ep = edge_points
        self.cross_road_count = crossroad_count
        # self.n_flight_plan_vars = crossroad_count + 2 * self.n_edges
        self.n_flight_plan_vars = crossroad_count + 3 * self.n_edges + 2 * climb_descent_coefficients + 1
        self.fp_vars_sweep_breadth = max(crossroad_count, self.n_edges, climb_descent_coefficients)
        # crossroads + edges * (TAS, FL, d2g_desc_delta) + d2g_desc + 2*cd_coeffs
        self.n_random_vars = crossroad_count + self.n_edges * 2
        self.max_nodes_per_path = max_nodes_per_path
        self.max_subnodes_per_path = max_subnodes_per_path

    def get_decision_array_shape(self, n_fp):
        return n_fp, self.n_flight_plan_vars

    def get_constants_dictionary(self):
        return {'n_nodes': self.n_nodes,
                'n_edges': self.n_edges,
                'origin_idx': self.ni[self.o],
                'destination_idx': self.ni[self.d],
                'origin_lat': self.nll[self.ni[self.o], 0],
                'origin_lon': self.nll[self.ni[self.o], 1],
                'max_nodes_per_path': self.max_nodes_per_path,
                'max_subnodes_per_path': self.max_subnodes_per_path,
                'n_flight_plan_vars': self.n_flight_plan_vars,
                'n_random_vars': self.n_random_vars,
                'max_subpoints_per_edge': max(self.edge_n_points),
                'crt_row_length': self.crt.shape[1],
                'n_crossroad_vars': self.cross_road_count,
                'fp_vars_sweep_breadth': self.fp_vars_sweep_breadth,
                }


class RouteGraph(nx.DiGraph):

    def __init__(self):
        super().__init__()

    @classmethod
    def init_from_gexf(cls, file_or_path):
        rg = cls()
        rg.update(rw.read_gexf(file_or_path))
        return rg

    def save_to_gexf(self, file_path):
        rw.write_gexf(self, file_path)

    @staticmethod
    def remove_loops_from_path(path):
        # note: in general there might be several ways to break loops, leading to
        # different paths. Ideally
        while len(path) > len(set(path)):
            for i, node in enumerate(path):
                if path.count(node) > 1:
                    last_index = len(path) - 1 - path[::-1].index(node)
                    path = path[:i] + path[last_index:]
                    break  # break the for, not the while, once a simplication is made
        return path

    def get_subgraph_origin_to_destination(self, origin, destination):
        descendants = nx.algorithms.descendants(self, origin)
        ancestors = nx.algorithms.ancestors(self, destination)
        nodes = descendants.intersection(ancestors)
        return self.subgraph(nodes)

    def draw(self, show=True, show_distances=False, show_node_ids=False, show_shortest_path=None):
        pos = {name: (node['lon'], node['lat']) for name, node in self.nodes.items()}
        nx.draw(self, pos, node_size=10)
        if show_distances:
            edge_labels = {e: f"{self.edges[e]['distance_nm']:.1f}" for e in self.edges}
            nx.draw_networkx_edge_labels(self, pos, edge_labels=edge_labels)
        if show_node_ids:
            node_labels = {n: f"{i}: {n}" for i, n in enumerate(self.nodes)}
            nx.draw_networkx_labels(self, pos, labels=node_labels)
        if show_shortest_path:
            print(shortest_path_length(self,
                                       source=show_shortest_path[0],
                                       target=show_shortest_path[1],
                                       weight='distance_nm'))
            path = shortest_path(self,
                                 source=show_shortest_path[0],
                                 target=show_shortest_path[1],
                                 weight='distance_nm')
            nodes = [self.nodes[node] for node in path]
            lats = [node['lat'] for node in nodes]
            lons = [node['lon'] for node in nodes]
            plt.plot(lons, lats, color='C1', linewidth=1.5)

        if show:
            plt.show()

    def remove_u_turns(self, origin, destination):

        edges_to_remove = []
        for edge in self.edges:
            if edge[1] == origin or edge[0] == destination:
                edges_to_remove.append(edge)
        for edge in edges_to_remove:
            self.remove_edge(*edge)

        def _test_node(node):
            out_edges = self.out_edges(node)
            in_edges = self.in_edges(node)
            if len(out_edges) != 1:
                return None
            if len(in_edges) != 1:
                return None
            in_edge = list(in_edges)[0]
            out_edge = list(out_edges)[0]
            if in_edge[0] == out_edge[1]:
                self.remove_edge(*in_edge)
                self.remove_edge(*out_edge)
                self.remove_node(node)
                _test_node(in_edge[0])

        for node in list(self.nodes.keys()):
            if node in self.nodes and node != origin and node != destination:
                _test_node(node)

    def reduce_d_factor(self, origin, destination, d_factor=1.2):

        new_graph = type(self)()

        shortest_length = shortest_path_length(self,
                                               source=origin,
                                               target=destination,
                                               weight='distance_nm')

        def _add_node(label):
            if label not in new_graph:
                old_node = self.nodes[label]
                new_graph.add_node(label,
                                   lat=old_node['lat'],
                                   lon=old_node['lon'],
                                   comment=old_node['comment'])

        def _add_edge(node1, node2):
            edge = (node1, node2)
            if edge not in new_graph.edges:
                new_graph.add_edge(node1, node2, distance_nm=self[node1][node2]['distance_nm'])

        def _add_path(path):
            path = self.remove_loops_from_path(path)
            _add_node(path[0])
            for node_prev, node_next in zip(path[:-1], path[1:]):
                _add_node(node_next)
                _add_edge(node_prev, node_next)

        for i, edge in enumerate(self.edges):
            if edge not in new_graph.edges and edge[::-1] not in new_graph.edges:
                try:
                    d_o = shortest_path_length(self,
                                               source=origin,
                                               target=edge[0],
                                               weight='distance_nm')
                    d_d = shortest_path_length(self,
                                               source=edge[1],
                                               target=destination,
                                               weight='distance_nm')

                except NetworkXNoPath:
                    continue
                if edge[::-1] in self.edges: # check whether the reverse edge is actually in a shorter path
                    try:
                        d_o_alt = shortest_path_length(self,
                                                   source=origin,
                                                   target=edge[1],
                                                   weight='distance_nm')
                        d_d_alt = shortest_path_length(self,
                                                   source=edge[0],
                                                   target=destination,
                                                   weight='distance_nm')
                        if d_d_alt + d_o_alt < d_o + d_d:
                            edge = edge[::-1]
                            d_o = d_o_alt
                            d_d = d_d_alt
                    except NetworkXNoPath:
                        continue
                d_edge = self.edges[edge]['distance_nm']
                if d_o + d_d + d_edge <= d_factor * shortest_length:
                    _add_node(edge[0])
                    _add_node(edge[1])
                    _add_edge(*edge)
                    _add_path(shortest_path(self,
                                            source=origin,
                                            target=edge[0],
                                            weight='distance_nm'))
                    _add_path(shortest_path(self,
                                            source=edge[1],
                                            target=destination,
                                            weight='distance_nm'))

        new_graph.remove_u_turns(origin, destination)
        return new_graph

    def reduce_k_shortest_paths(self, origin, destination, number_of_paths=100):

        paths = shortest_simple_paths(self, source=origin, target=destination, weight="distance_nm")
        new_graph = type(self)()

        def _add_node(label):
            if label not in new_graph:
                old_node = self.nodes[label]
                new_graph.add_node(label,
                                   lat=old_node['lat'],
                                   lon=old_node['lon'],
                                   comment=old_node['comment'])

        def _add_edge(node1, node2):
            edge = (node1, node2)
            if edge not in new_graph.edges:
                new_graph.add_edge(node1, node2, distance_nm=self[node1][node2]['distance_nm'])

        for k, path in enumerate(paths):
            if k == number_of_paths:
                break
            path = self.remove_loops_from_path(path)
            _add_node(path[0])
            for node_prev, node_next in zip(path[:-1], path[1:]):
                _add_node(node_next)
                _add_edge(node_prev, node_next)

        return new_graph


    def generate_flight_plan_encoding(self, origin, destination, horizontal_res, max_outgoing_edges_per_node=8,
                                      climb_descent_coefficients=5):
        required_crossroad_vars = {0: 0}  # dict instead of function with log2 for extra performance LOL
        n = 1
        low = 1
        while max_outgoing_edges_per_node not in required_crossroad_vars.keys():
            max_range = 2 ** n
            for i in range(low, max_range + 1):
                required_crossroad_vars[i] = n
            n += 1
            low = max_range + 1
        required_crossroad_vars[1] = 0

        n_nodes = self.number_of_nodes()
        n_edges = self.number_of_edges()
        moepn = max_outgoing_edges_per_node
        number_random_vars = int(np.ceil(np.log2(moepn)))
        row_length = 2 + moepn + number_random_vars
        row_length = 2 ** (int(np.ceil(np.log2(row_length))))

        node_indexes = {}

        for i, node in enumerate(self):
            node_indexes[node] = i

        node_index_table = np.zeros((n_nodes, row_length), dtype=np.int32)
        crossroad_count = 0

        for node, i in node_indexes.items():
            succesors = self[node].keys()
            successor_indexes = [node_indexes[s] for s in succesors]
            l = len(succesors)
            cr = required_crossroad_vars[l]
            node_index_table[i, 0] = l
            node_index_table[i, 1:(cr + 1)] = range(crossroad_count, crossroad_count + cr)
            crossroad_count += cr
            node_index_table[i, (cr + 1):(cr + 1 + l)] = successor_indexes

        node_coordinates = np.zeros((n_nodes, 2), dtype=np.float32)
        for node, i in node_indexes.items():
            node_coordinates[i, :] = (self.nodes[node]['lat'], self.nodes[node]['lon'])

        nn2e = -1 * np.ones((n_nodes, n_nodes), dtype=np.int32)
        e2nn = -1 * np.ones((n_edges, 2), dtype=np.int32)
        edge_lengths = np.zeros(n_edges, dtype=np.float32)
        edge_n_points = np.zeros(n_edges, dtype=np.int32)

        min_edge_length = 99999999
        min_segment_length = 99999999

        for j, edge in enumerate(self.edges):
            data = self.get_edge_data(*edge)
            start_idx = node_indexes[edge[0]]
            end_idx = node_indexes[edge[1]]
            nn2e[start_idx, end_idx] = j
            e2nn[j, :] = (start_idx, end_idx)
            edge_lengths[j] = data['distance_nm']
            edge_n_points[j] = int(np.ceil(data['distance_nm'] / horizontal_res)) + 1
            min_edge_length = min(data['distance_nm'], min_edge_length)
            min_segment_length = min(data['distance_nm']/(edge_n_points[j] - 1), min_segment_length)

        edge_points = np.zeros((n_edges, edge_n_points.max(), 2), dtype=np.float32)
        for j, edge in enumerate(self.edges):
            o = node_coordinates[node_indexes[edge[0]], :]
            d = node_coordinates[node_indexes[edge[1]], :]
            point_arr = generate_gcircle_point_list_n(o, d, edge_n_points[j] - 1)
            edge_points[j, :edge_n_points[j], :] = point_arr

        def path_subsegments(path):
            total_subsegments = 0
            for node1, node2 in zip(path[:-1], path[1:]):
                idx1 = node_indexes[node1]
                idx2 = node_indexes[node2]
                edge_index = nn2e[idx1, idx2]
                total_subsegments += edge_n_points[edge_index] - 1
            return total_subsegments

        spl = shortest_path_length(self,
                                   source=origin,
                                   target=destination,
                                   weight='distance_nm')

        max_nodes_per_path = int((np.ceil(spl * 1.05 / min_edge_length)))
        max_subnodes_per_path = int((np.ceil(spl * 1.05 / min_segment_length)))
        #max_nodes_per_path = len(max(nx.all_simple_paths(self, origin, destination), key=lambda x: len(x)))
        #path_with_max_subnodes = max(nx.all_simple_paths(self, origin, destination), key=path_subsegments)
        #max_subnodes_per_path = path_subsegments(path_with_max_subnodes)

        return FlightPlanEncoding(origin, destination, n_nodes, n_edges, node_indexes, node_index_table,
                                  node_coordinates, nn2e, e2nn, edge_lengths * 1852, edge_n_points, edge_points,
                                  crossroad_count, max_nodes_per_path + 1, max_subnodes_per_path + 1,
                                  climb_descent_coefficients)


class NavigationNetwork(object):

    def __init__(self, dataframe):
        self.df = dataframe

    @classmethod
    def init_from_csv(cls, path):
        df = pd.read_csv(path, ',')
        return cls(df)

    def create_graph(self) -> RouteGraph:
        graph = RouteGraph()
        point_labels = set()
        route_labels = set()
        for index, point in self.df.iterrows():
            label = point['Point']
            lat = point['Lat']
            lon = point['Lon']
            com = point['Comment']
            if label not in point_labels:
                graph.add_node(label, lat=lat, label=label, lon=lon, comment=com)
                point_labels.add(label)
            route_labels.add(point['RouteName'])
        for route in route_labels:
            df_r = self.df[self.df['RouteName'] == route]
            df_r.sort_values('PointNumber')
            labels = df_r['Point']
            for p0, p1 in zip(labels[:-1], labels[1:]):
                w = RQCoords([graph.nodes[p0]['lat'],
                              graph.nodes[p0]['lon']],
                             [graph.nodes[p1]['lat'],
                              graph.nodes[p1]['lon']]).distance_o2d_nm
                graph.add_edge(p0, p1, distance_nm=w)
        return graph

    def create_graph_from_path(self, path) -> RouteGraph:
        graph = RouteGraph()
        segment_list = zip(path[:-1], path[1:])
        for node in path:
            rows = self.df[self.df['Point'] == node]
            lat = rows['Lat'].iloc[0]
            lon = rows['Lon'].iloc[0]
            com = rows['Comment'].iloc[0]
            graph.add_node(node, lat=lat, label=node, lon=lon, comment=com)
        for segment in segment_list:
            p0, p1 = segment
            w = RQCoords([graph.nodes[p0]['lat'],
                          graph.nodes[p0]['lon']],
                         [graph.nodes[p1]['lat'],
                          graph.nodes[p1]['lon']]).distance_o2d_nm
            graph.add_edge(p0, p1, distance_nm=w)
        return graph

    def get_point_coords(self, point_label):
        return self.df[self.df['Point'] == point_label].iloc[0][['Lat', 'Lon']].to_numpy()

    def filter_geo(self, points: Sequence[Sequence[float]], aperture: float = 1):
        p0 = points[0]
        p1 = points[1]
        rq_od = RQCoords(p0, p1)
        pA = rq_od.rq2ll(0, - 2 * aperture)
        pB = rq_od.rq2ll(0, 2 * aperture)
        rq_OA = RQCoords(p0, pA)
        rq_OB = RQCoords(p0, pB)
        rq_AD = RQCoords(pA, p1)
        rq_BD = RQCoords(pB, p1)
        eps = 1e-6

        def is_in_area(lat, lon):
            # print(rq_OA.get_distance_to_plane(lat, lon),
            #       rq_OB.get_distance_to_plane(lat, lon),
            #       rq_AD.get_distance_to_plane(lat, lon),
            #       rq_BD.get_distance_to_plane(lat, lon))
            return rq_OA.get_distance_to_plane(lat, lon) > -eps and \
                   rq_OB.get_distance_to_plane(lat, lon) < eps and \
                   rq_AD.get_distance_to_plane(lat, lon) > -eps and \
                   rq_BD.get_distance_to_plane(lat, lon) < eps

        def row_in_area(row):
            return is_in_area(row['Lat'], row['Lon'])

        new_df = self.df[self.df.apply(row_in_area, axis=1)]
        return type(self)(new_df)

    def filter_route_type(self, route_type='AR'):
        new_df = self.df[self.df['RouteType'] == route_type]
        return type(self)(new_df)

    def filter_connected(self):
        counts = {}
        for index, p in self.df.iterrows():
            loc = p['Lat'], p['Lon']
            if loc in counts:
                counts[loc] += 1
            else:
                counts[loc] = 1
        new_df = self.df[self.df.apply(lambda row: counts[(row['Lat'], row['Lon'])] > 1, axis=1)]
        return type(self)(new_df)

    def plot(self, lw=0.5, alpha=0.5):
        airways = {}
        for index, p in self.df.iterrows():
            route = p['RouteName']
            ll = (p['Lat'], p['Lon'])
            if route in airways:
                airways[route][0].append(ll)
            else:
                airways[route] = [[ll], p['RouteType']]
        import matplotlib.pyplot as plt
        d_color = {'AR': 'C0', 'AP': 'C2', 'DP': 'C1', 'OT': 'C3'}
        for route_name, route_points in airways.items():
            points = np.array(route_points[0])
            color = d_color[route_points[1]]
            plt.plot(points[:, 1], points[:, 0], '.-', color=color, alpha=alpha, linewidth=lw)
        plt.show()
