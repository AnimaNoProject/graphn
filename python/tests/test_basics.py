import unittest
from graphn import graphn
import networkx as nx
import numpy as np
from functools import cmp_to_key


class TestGraphnFunctions(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def dist(self, dA, dB):
        return int(abs(np.linalg.norm(dA - dB)))

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

    def test_GraphEncodingShouldGiveCorrectCode(self) -> None:
        a = nx.Graph()
        a.add_node(11, pos=np.asarray([1, 1]))
        a.add_node(12, pos=np.asarray([2, 1]))
        a.add_node(13, pos=np.asarray([2, 2]))
        a.add_node(14, pos=np.asarray([1, 2]))
        a.add_node(15, pos=np.asarray([0, 0]))
        a.add_node(16, pos=np.asarray([3, 0]))
        a.add_node(17, pos=np.asarray([3, 3]))
        a.add_node(18, pos=np.asarray([0, 3]))
        a.add_edge(15, 11)
        a.add_edge(11, 14)
        a.add_edge(14, 18)
        a.add_edge(14, 13)
        a.add_edge(13, 17)
        a.add_edge(13, 12)
        a.add_edge(12, 16)

        self.assertEqual("2314536758", graphn.generate_graph_code(a, self.ordered_neighbours)[0])

    def test_MarkedSubGraphIsomorphismShouldGiveCorrectResult(self) -> None:
        a = nx.Graph()
        a.add_node(11, pos=np.asarray([1, 1]))
        a.add_node(12, pos=np.asarray([2, 1]))
        a.add_node(13, pos=np.asarray([2, 2]))
        a.add_node(14, pos=np.asarray([1, 2]))
        a.add_node(15, pos=np.asarray([0, 0]))
        a.add_node(16, pos=np.asarray([3, 0]))
        a.add_node(17, pos=np.asarray([3, 3]))
        a.add_node(18, pos=np.asarray([0, 3]))
        a.add_edge(15, 11)
        a.add_edge(11, 14)
        a.add_edge(14, 18)
        a.add_edge(14, 13)
        a.add_edge(13, 17)
        a.add_edge(13, 12)
        a.add_edge(12, 16)

        self.assertTrue(graphn.marked_subgraph_isomorphism(a, a, self.ordered_neighbours))

        b = nx.Graph()
        b.add_node(21, pos=np.asarray([1, 1]))
        b.add_node(22, pos=np.asarray([2, 1]))
        b.add_node(23, pos=np.asarray([2, 2]))
        b.add_node(24, pos=np.asarray([1, 2]))
        b.add_node(25, pos=np.asarray([0, 0]))
        b.add_node(26, pos=np.asarray([3, 0]))
        b.add_node(27, pos=np.asarray([3, 3]))
        b.add_node(28, pos=np.asarray([0, 3]))
        b.add_edge(25, 21)
        b.add_edge(21, 24)
        b.add_edge(24, 28)
        b.add_edge(24, 23)
        b.add_edge(23, 27)
        b.add_edge(23, 22)
        b.add_edge(22, 26)
        b.add_edge(21, 22)

        self.assertFalse(graphn.marked_subgraph_isomorphism(b, a, self.ordered_neighbours))

        c = nx.Graph()
        c.add_node(31, pos=np.asarray([1, 1]))
        c.add_node(32, pos=np.asarray([2, 1]))
        c.add_node(33, pos=np.asarray([2, 2]))
        c.add_node(34, pos=np.asarray([1, 2]))
        c.add_node(35, pos=np.asarray([0, 0]))
        c.add_node(36, pos=np.asarray([3, 0]))
        c.add_node(37, pos=np.asarray([3, 3]))
        c.add_node(38, pos=np.asarray([0, 3]))
        c.add_edge(35, 31)
        c.add_edge(31, 34)
        c.add_edge(34, 38)
        c.add_edge(34, 33)
        c.add_edge(33, 37)
        c.add_edge(33, 32)
        c.add_edge(32, 36)
        c.add_edge(31, 32)
        c.add_edge(35, 36)
        c.add_edge(36, 37)
        c.add_edge(37, 38)
        c.add_edge(38, 35)

        self.assertTrue(graphn.marked_subgraph_isomorphism(b, c, self.ordered_neighbours))

    def test_GGDReturns0ForIsomorphicGraphs(self) -> None:
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

    def test_GGDReturnCorrectDistance(self) -> None:
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

        self.assertEqual(0, graphn.ggd(a, a, self.dist_fun))
        self.assertEqual(1, graphn.ggd(a, b, self.dist_fun))
        self.assertEqual(1, graphn.ggd(b, a, self.dist_fun))
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

        self.assertEqual(10, graphn.ggd(c, b, self.dist_fun))

        d = nx.Graph()
        d.add_node(0, pos=np.asarray([2, 0]))
        d.add_node(1, pos=np.asarray([0, 0]))
        d.add_edge(1, 0)
        self.assertEqual(3, graphn.ggd(a, d, self.dist_fun, 1))
        self.assertEqual(4, graphn.ggd(a, d, self.dist_fun, 2))
        self.assertEqual(4, graphn.ggd(a, d, self.dist_fun, 4, 1))
        self.assertEqual(24, graphn.ggd(a, d, self.dist_fun, 4, 10))

        t = nx.Graph()

        t.add_node(0, pos=np.asarray([0, 0], dtype=np.int))
        t.add_node(1, pos=np.asarray([10, 0], dtype=np.int))
        t.add_node(2, pos=np.asarray([5, 10], dtype=np.int))
        t.add_node(3, pos=np.asarray([5, 20], dtype=np.int))
        t.add_node(4, pos=np.asarray([5, 25], dtype=np.int))
        t.add_node(5, pos=np.asarray([0, 28], dtype=np.int))
        t.add_node(6, pos=np.asarray([10, 28], dtype=np.int))

        t.add_edge(0, 2)
        t.add_edge(1, 2)
        t.add_edge(2, 3)
        t.add_edge(3, 4)
        t.add_edge(3, 5)
        t.add_edge(3, 6)

        self.assertEqual(0, graphn.ggd(t, t, self.dist_fun))



    def test_GGDReturnsMaxCostIfOneGraphEmpty(self) -> None:
        a = nx.Graph()
        a.add_node(0, pos=np.asarray([1, 0]))
        a.add_node(1, pos=np.asarray([2, 0]))
        a.add_node(2, pos=np.asarray([0, 0]))
        a.add_edge(0, 1)
        a.add_edge(0, 2)

        b = nx.Graph()
        self.assertEqual(np.inf, graphn.ggd(a, b, self.dist_fun))
