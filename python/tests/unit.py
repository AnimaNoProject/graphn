import unittest
import graphn
import networkx as nx
import numpy as np
from functools import cmp_to_key


class TestGraphnFunctions(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def dist(self, dA, dB):
        return np.linalg.norm(dA - dB)

    def dist_fun(self, dA, dB):
        return self.dist(dA['pos'], dB['pos'])

    def angle(self, dCenter, dA):
        x, y = dA[0] - dCenter
        angle = np.arctan2(x, y)
        if angle < 0:
            angle = 2 * np.pi + angle
        return angle

    def compare_points(self, dA, dB):
        relative_center = np.asarray([0, 0])
        angle_a = self.angle(relative_center, dA[1])
        angle_b = self.angle(relative_center, dB[1])
        if angle_a < angle_b:
            return 1
        if angle_a == angle_b and self.dist(relative_center, dA[1]) < self.dist(relative_center, dB[1]):
            return 1
        return -1

    def ordered_neighbours(self, graph: nx.Graph, center_id, start_id):
        sorted_neighbours = []
        positions = nx.get_node_attributes(graph, 'pos')

        for n, cur in graph.edges(center_id):
            pos = positions[cur]
            sorted_neighbours.append((cur, pos - positions[center_id]))

        if len(sorted_neighbours) < 2:
            return [start_id]
        sorted(sorted_neighbours, key=cmp_to_key(self.compare_points))

        id_to_start = 0
        for neighbour_id in range(len(sorted_neighbours)):
            if sorted_neighbours[neighbour_id][0] == start_id:
                id_to_start = neighbour_id

        ordered_neighbours = []
        for neighbour_id in range(len(sorted_neighbours)):
            ordered_neighbours.append(sorted_neighbours[(id_to_start + neighbour_id) % len(sorted_neighbours)][0])

        return ordered_neighbours

    def testGraphEncodingShouldGiveCorrectCode(self) -> None:
        a = nx.Graph()
        a.add_node(0, pos=np.asarray([1, 0]))
        a.add_node(1, pos=np.asarray([2, 0]))
        a.add_node(2, pos=np.asarray([3, 0]))
        a.add_node(3, pos=np.asarray([4, 0]))
        a.add_node(4, pos=np.asarray([5, 0]))
        a.add_node(5, pos=np.asarray([6, 0]))
        a.add_node(6, pos=np.asarray([7, 0]))
        a.add_node(7, pos=np.asarray([8, 0]))
        a.add_edge(4, 0)
        a.add_edge(0, 3)
        a.add_edge(3, 7)
        a.add_edge(3, 2)
        a.add_edge(2, 6)
        a.add_edge(2, 1)
        a.add_edge(1, 5)

        self.assertEqual("2314536758", graphn.generate_graph_code(a, self.ordered_neighbours))

    def testGGDReturns0ForIsomorphicGraphs(self) -> None:
        a = nx.Graph()
        a.add_node(0, pos=np.asarray([1, 0]))
        a.add_node(1, pos=np.asarray([2, 0]))
        a.add_edge(0, 1)

        b = nx.Graph()
        b.add_node(0, pos=np.asarray([1, 0]))
        b.add_node(1, pos=np.asarray([2, 0]))
        b.add_edge(0, 1)

        self.assertEqual(graphn.ggd(a, b, self.dist_fun), 0)
        self.assertEqual(graphn.ggd(a, a, self.dist_fun), 0)

        c = nx.Graph()
        c.add_node(0, pos=np.asarray([2, 0]))
        c.add_node(1, pos=np.asarray([1, 0]))
        c.add_edge(1, 0)

        self.assertEqual(graphn.ggd(a, c, self.dist_fun), 0)

    def testGGDReturnCorrectDistance(self) -> None:
        a = nx.Graph()
        a.add_node(0, pos=np.asarray([1, 0]))
        a.add_node(1, pos=np.asarray([2, 0]))
        a.add_node(2, pos=np.asarray([0, 0]))
        a.add_edge(0, 1)
        a.add_edge(0, 2)

        b = nx.Graph()
        b.add_node(0, pos=np.asarray([1, 0]))
        b.add_node(1, pos=np.asarray([2, 0]))
        b.add_edge(0, 1)

        self.assertEqual(1, graphn.ggd(a, b, self.dist_fun))
        self.assertEqual(1, graphn.ggd(a, b, self.dist_fun, 2, 1))
        self.assertEqual(2, graphn.ggd(b, a, self.dist_fun, 1, 2))
        self.assertEqual(3, graphn.ggd(b, a, self.dist_fun, 1, 3))

        c = nx.Graph()
        c.add_node(0, pos=np.asarray([2, 0]))
        c.add_node(1, pos=np.asarray([1, 0]))
        c.add_node(2, pos=np.asarray([0, 0]))
        c.add_node(3, pos=np.asarray([3, 0]))
        c.add_node(4, pos=np.asarray([4, 0]))
        c.add_edge(2, 0)
        c.add_edge(2, 1)
        c.add_edge(2, 3)
        c.add_edge(2, 4)

        self.assertEqual(4, graphn.ggd(b, c, self.dist_fun))

        d = nx.Graph()
        d.add_node(0, pos=np.asarray([2, 0]))
        d.add_node(2, pos=np.asarray([0, 0]))
        d.add_edge(2, 0)
        self.assertEqual(1, graphn.ggd(a, d, self.dist_fun, 1))
        self.assertEqual(2, graphn.ggd(a, d, self.dist_fun, 2))
        self.assertEqual(2, graphn.ggd(a, d, self.dist_fun, 4, 1))   # cost smaller to exchange edges
        self.assertEqual(4, graphn.ggd(a, d, self.dist_fun, 4, 10))  # exchange edge cost too high

    def testGGDReturnsMaxCostIfOneGraphEmpty(self) -> None:
        a = nx.Graph()
        a.add_node(0, pos=np.asarray([1, 0]))
        a.add_node(1, pos=np.asarray([2, 0]))
        a.add_node(2, pos=np.asarray([0, 0]))
        a.add_edge(0, 1)
        a.add_edge(0, 2)

        b = nx.Graph()
        self.assertEqual(np.inf, graphn.ggd(a, b, self.dist_fun))

