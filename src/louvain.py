import numpy as np
from collections import defaultdict
from itertools import combinations
from copy import deepcopy

class Vertex(object):
    '''
    Class for a weighted vertex in a community graph.

    Attributes:
    parent: Vertex
        this Vertex's parent, i.e. representative of the community this Vertex 
        is in. Default is self.
    label: object
        the label of the vertex. Must be a hashable object.
    weighted_degree: float, int
        the weight of the vertex.

    Methods:
        __init__(label, weighted_degree)
        __hash__()
        __eq__()
    '''
    def __init__(self, label, weighted_degree):
        '''
        Init class.
        
        Arguments:
        label: object
            the label of the vertex. Must be a hashable object.
        weighted_degree: float, int
            the weight of the vertex.
        
        Raise:
            TypeError if label is unhashable.
        '''
        self.parent = self
        if '__hash__' not in dir(label):
            raise TypeError('label must be hashable!')
        self.label = label
        self.weighted_degree = weighted_degree

    def __hash__(self):
        '''
        Creates hash by assuming self.label's hash.
        '''
        return self.label.__hash__()

    def __eq__(self, other):
        '''
        Checks equality with other Vertex by comparing self.label with the 
        other's label.
        '''
        return self.label == other.label

    def __repr__(self):
        '''
        Creates Vertex representation using self.label.
        '''
        return self.label.__repr__()


class Graph(object):
    '''
    Class for a community graph.

    Attributes:
    adj_map: defaultdict(defaultdict), {Vertex: {Vertex: float}}
        adjacency map of the graph. Each Vertex v maps to a dictionary where 
        adjecent Vertices are keys and edge weights (floats) are values.
    m: float
        the sum of all edge weights.

    Methods:
    __init__(A, cutoff)
    __repr__()
    '''
    def __init__(self, A, cutoff=None):
        '''
        Init class.

        Arguments:
        A: numpy.ndarray
            adjacency matrix.
        cutoff: int, float
            the number of standard deviations from the mean edge weight by which
            a lowpass point is marked on the distribution of edge weights.
            For example, if cutoff = 1, every edge whose weight is below the 
            mean weight minus 1 standard distribution is disregarded. 
            If cutoff = None, no edge is disregarded. Default is None.
        '''
        self.adj_map = defaultdict(defaultdict)
        A_trimmed = np.triu(A, k=1) # upper-triangle of A, 
                                    # excluding the diagonal line
        self.m = A_trimmed.sum()
        mean = A_trimmed[np.where(A_trimmed > 0)].mean()
        std = A_trimmed[np.where(A_trimmed > 0)].std()
        if cutoff == None:
            low = 0
        else:
            low = mean - cutoff * std
        for r in range(A_trimmed.shape[0]):
            for c in range(A_trimmed.shape[0]):
                if (A_trimmed[r][c] < 1e+3) & (A_trimmed[r][c] >= low):
                    degree_r = A_trimmed[r].sum() + A_trimmed[:, r].sum()
                    degree_c = A_trimmed[c].sum() + A_trimmed[:, c].sum()
                    self.adj_map[Vertex(r, degree_r)][Vertex(c, degree_c)] = \
                    A_trimmed[r][c]
    
    def __repr__(self):
        '''
        Generates class representation by assuming adj_map.__repr__().
        '''
        return self.adj_map.__repr__()


class Louvain(object):
    '''
    Class for Louvain community detection.

    Attributes:
    G: Graph
        a Graph object whose communities are to be detected.
    epsilon: float
        threshold level for algorithm stopping.
    delta: np.inf, float
        ammount of change incurred after each step of algorithm. 
        Initiated at infinity (np.inf).
    communities: dict {Vertex: {Vertex}}
        a dictionary whose keys are Vertices and whose values are sets of 
        Vertices within the communities represented by corresponding keys.
        Is originally a defaultdict(set) on initialization.

    Methods:
    __init__(G, epsilon)
        init class.
    _get_w(v_1, v_2)
        gets weight of the edge incident to 2 Vertices v_1 and v_2.
    _find_roots(v)
        finds the root, i.e. "ultimate parent", of a Vertex.
    _k_in(parent, v)
        gets the sum of all edge weights within a community.
    _sigma_in(parent, v)
        gets sum of all the weights of the links inside the community v is
        moving into.
    _sigma_total(parent, v, sigma_in)
        gets sum of all the weights of the links to nodes in the community 
        v is moving into.
    _get_delta(v_in, v_comm)
        gets the change in modularity once a Vertex is moved into a community.
    _delta_update()
        performs the core of Louvain algorithm while updating the optimal delta.
    find()
        performs the Louvain algorithm.
    '''
    def __init__(self, G, epsilon=1e-16):
        '''
        Init class.

        Arguments:
        G: Graph
            a Graph object.
        epsilon:
            threshold level for algorithm stopping. Default at 1e-16.
        '''
        self.G = G
        self.epsilon = epsilon
        self.delta = np.inf
        self.communities = defaultdict(set)
        for v in self.G.adj_map:
            self.communities[v].add(v)

    def _get_w(self, v_1, v_2):
        '''
        Gets weight of the edge incident to 2 Vertices v_1 and v_2.
        '''
        return self.G.adj_map[v_1][v_2]

    def _find_root(self, v):
        '''
        Finds the root, i.e. "ultimate parent", of a Vertex.
        '''
        if v.parent != v:
            v.parent = self._find_root(v.parent)
        return v.parent

    def _k_in(self, parent, v):
        '''
        Gets the sum of all edge weights within a community.
        '''
        k_in = 0
        for v_1 in self.communities[parent]:
            k_in += self._get_w(v, v_1)
        return k_in

    def _sigma_in(self, parent, v):
        '''
        Gets sum of all the weights of the links inside the community v is
        moving into.
        '''
        sigma = 0
        for v_1, v_2 in combinations(self.communities[parent], 2):
            sigma += self._get_w(v_1, v_2)
        return sigma 

    def _sigma_total(self, parent, v, sigma_in):
        '''
        Gets sum of all the weights of the links to nodes in the community 
        v is moving into.
        '''
        sigma = 0
        for v_1 in self.communities[parent]:
            sigma += v_1.weighted_degree
        return sigma - sigma_in

    def _get_delta(self, v_in, v_comm):
        '''
        Gets the change in modularity once a Vertex is moved into a community.
        '''
        parent = v_comm.parent
        k_in = self._k_in(parent, v_in)
        sigma_in = self._sigma_in(parent, v_in)
        sigma_total = self._sigma_total(parent, v_in, sigma_in)
        m = self.G.m
        k = v_in.weighted_degree
        return (sigma_in + 2*k_in)/(2*m) - ((sigma_total + k)/(2*m))**2 \
             - (sigma_in/(2*m)) + (sigma_total/(2*m))**2 + (k/(2*m))**2

    def _delta_update(self):
        '''
        Performs the core of Louvain algorithm while updating the optimal delta.
        '''
        while self.delta > self.epsilon:

            for v in self.G.adj_map.keys():
                delta_list = []
                candidates = []
                # gets all hypothetical deltas
                for visited in self.G.adj_map[v]:
                    delta_list.append(self._get_delta(v, visited))
                    candidates.append(visited)
                # actually moves v to the community which hypothetically yields
                # the largest delta
                i_max = np.argmax(delta_list)
                i_min = np.argmin(delta_list)
                if delta_list[i_max] - delta_list[i_min] > self.epsilon:
                    self.delta = delta_list[i_max]
                    old_p = self._find_root(v)
                    v.parent = self._find_root(candidates[i_max])
                    self.communities[v.parent].add(v)
                    # removes v from old community
                    self.communities[old_p].remove(v) 

    def find(self):
        '''
        Performs the Louvain algorithm.

        Arguments: None

        Return:
        A dictionary whose keys are community representatives (type: Vertex) and 
        whose values are sets of Vertices within such communities 
        (type: {Vertex}).
        '''
        self._delta_update()
        comm_copy = deepcopy(self.communities)
        
        # corrects remaining Vertices whose parents are not those of their
        # community representatives.
        for key, values in comm_copy.items():
            for v in values:
                if v.parent != key.parent:
                    v.parent = key.parent
                    self.communities[v.parent].add(v)
                    self.communities[key].remove(v)

        self.communities ={k:v for k,v in self.communities.items() if len(v)!=0}
        return self.communities