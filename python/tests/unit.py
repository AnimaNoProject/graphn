import unittest
import graphn
import networkx as nx
import numpy as np


class TestGraphnFunctions(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def dist_fun(self, dA, dB):
        return np.linalg.norm(np.asarray([dA['pos'][0], dA['pos'][1]]) - np.asarray([dB['pos'][0], dB['pos'][1]]))

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

    def testGGDReturnsMaxCostIfOneGraphEmpty(self) -> None:
        a = nx.Graph()
        a.add_node(0, pos=np.asarray([1, 0]))
        a.add_node(1, pos=np.asarray([2, 0]))
        a.add_node(2, pos=np.asarray([0, 0]))
        a.add_edge(0, 1)
        a.add_edge(0, 2)

        b = nx.Graph()
        self.assertEqual(np.inf, graphn.ggd(a, b, self.dist_fun))

