import numpy as np

class KMeansSpectral(object):
    '''
    Class for performing spectral clustering with K-means++ on a Bhattacharyya
    coefficient matrix of HMMs. 

    Attributes:
    k: int
        number of clusters
    epsilon: float
        threshold level. If the sum of norm-1 distances between old and updated 
        centroids gets below epsilon, stop fitting before undergoing all epochs.
    epochs: int
        number of epochs.
    centroids: list of (int, numpy.ndarray) tuples
        list of 2-tuples representing centroids, whose 0th element is a number 
        representing its name, and the other is a numpy array of its 
        k-dimensional coordinate.
    classes: dict, {int: (int, numpy.ndarray) tuple}
        dictionary whose keys are indices of k classes and values are 2-tuples 
        of points. Each tuple's 0th element is the name of the point
        corresponding to the index of the neuron HMM it represents, and the 
        other one is a numpy array of its k-dimensional coordinate.
    classes_idx: dict, {int: int}
        dictionary whose keys are indices of k classes and values are names of  
        points.
    D2: numpy.ndarray
        an array of squared distances between every pair of initial centroid and
        data point.
    cumsum_D2: numpy.ndarray
        array of cumulative sums of ratios deduced from D2 / sum of D2.

    Methods:
    __init__(A, k, epsilon=1e-5, epochs=500)
        init class.
    _reduced_laplacian(A)
        creates an (N, N) reduced graph Laplacian from a non-negative 
        Bhattacharyya coefficient matrix.
    _eigvec_matrix(A)
        creates a (k, N) matrix by column-stacking k eignevectors associated 
        with k smallest eigenvalues of the (N, N) Laplacian.
    _cal_D_squared(data)
        computes D2.
    _find_next_centroid(data)
        finds another centroid after one initial centroid has been found.
    _init_centroids(data)
        finds all initial centroids.
    _fit_one_round(data)
        performs K-means++ with a random batch of initial centroids.
    fit(num_votes=1)
        performs K-means++ for several times then returns the result that
        repeats the most times.
    '''
    def __init__(self, k, epsilon=1e-5, epochs=500):
        '''
        Init class.

        Arguments:
        k: int
            the number of clusters.
        epsilon: float
            the epsilon threshold for K-means++. Default at 1e-5.
        epochs: int
            the number of epochs for each round of K-means++. Default at 500.
        '''
        self.k = k
        self.epsilon = epsilon
        self.epochs = epochs
        self.centroids = []
        self.classes = None
        self.classes_idx = None
        self.D2 = None
        self.cumsum_D2 = None

    def _reduced_laplacian(self, A):
        '''
        Creates an (N, N) reduced graph Laplacian from a non-negative 
        Bhattacharyya coefficient matrix.
        '''
        D = np.diag(np.reciprocal(np.sqrt(np.sum(A, axis=0))))
        return D @ A @ D

    def _eigvec_matrix(self, A):
        '''
        Creates a (k, N) matrix by column-stacking k eignevectors associated 
        with k smallest eigenvalues of the (N, N) Laplacian.
        '''
        laplacian = self._reduced_laplacian(A)
        eigval, eigvec = np.linalg.eig(laplacian)
        eigpair = list(zip(eigval, eigvec))
        eigpair.sort()
        k_matrix = np.array([pair[1] / np.linalg.norm(pair[1]) \
                            for pair in eigpair[:self.k]]).T
        return list(zip(range(len(k_matrix)), k_matrix))

    def _cal_D_squared(self, data):
        '''
        Computes D2.
        '''
        self.D2 = np.array([np.min([np.linalg.norm(d[1]-c[1])**2 for c in self.centroids]) \
                            for d in data])
    
    def _find_next_centroid(self, data):
        '''
        Finds another centroid after one initial centroid has been found.
        '''
        self.cumsum_D2 = np.cumsum(self.D2 / self.D2.sum())
        mu = np.random.rand(1)[0]
        return data[np.where(self.cumsum_D2 >= mu)[0][0]]

    def _init_centroids(self, data):
        '''
        Finds all initial centroids.
        '''
        self.centroids = []
        self.centroids.append(data[np.random.choice(len(data), 1)[0]])
        for i in range(self.k - 1):
            self._cal_D_squared(data)
            self.centroids.append(self._find_next_centroid(data))
    
    def _fit_one_round(self, data):
        '''
        Performs K-means++ with a random batch of initial centroids.
        '''
        self._init_centroids(data)
        classes = {}
        classes_idx = {}

        for j in range(self.epochs):

            for i in range(self.k):
                classes[i] = []
                classes_idx[i] = set()
            
            # puts each data point into a class whose centroid is closest to it
            # relative to other centroids.
            for d in data:
                dist = [np.linalg.norm(d[1] - centroid[1]) for centroid in self.centroids]
                classify = np.argmin(dist)
                classes[classify].append(d)
                classes_idx[classify].add(d[0])
            
            # updates centroid coordinates 
            previous = dict(zip(range(len(self.centroids)), self.centroids))
            for c in classes:
                mean = np.mean([point[1] for point in classes[c]], axis = 0)
                self.centroids[c] = (c, mean)

            # sees if updated centroids are significantly different from old
            # ones by comparing the sum of differences with epsilon.
            optimal = True
            for c in range(len(self.centroids)):
                prev_centroid = previous[c]
                curr = self.centroids[c]
            if np.sum(curr[1] - prev_centroid[1]) > self.epsilon:
                optimal = False

            # breaks early if already optimal, i.e. the updated centroids
            # are not significantly different from their previous ones
            if optimal:
                break
        
        return classes, classes_idx
        
    def fit(self, A, num_votes=1):
        '''
        Performs K-means++ for several times then returns the result that
        repeats the most times. Eventually updates the fields "classes" and
        "classes_idx" of this class.

        Note that num_votes should be > 1 only when data points are conjectured
        to be easily separated into k distinct clusters, which is not always 
        the case.

        Arguments:
        A: numpy.ndarray
            an (N, N) non-negative Gram matrix of Bhattacharyya coefficients 
            of N HMMs.
        num_votes: int
            number of rounds to perform K-means++. Default at 1.
        '''
        # init eigenvector matrix, voting board and list of results
        data = self._eigvec_matrix(A)
        wins = np.zeros(num_votes)
        models = []

        # finds the results from all rounds
        for i in range(num_votes):
            models.append(self._fit_one_round(data))

        # finds the result that repeats the most time throughout all rounds.
        for i, model in enumerate(models):
            for other in (models[:i] + models[i+1:]):
                if model[1] == other[1]:
                    wins[i] += 1

        winner = models[np.argmax(wins)]
        self.classes = winner[0]
        self.classes_idx = winner[1]