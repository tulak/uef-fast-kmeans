import matplotlib.pyplot as plt
import numpy as np
from random import randint
from IPython import embed as pry
import time

class FastKMeans:
    data = None
    data_size = None
    centroids = None
    nClusters = None
    vectors_register = None
    def __init__(self, data, nClusters):
        self.data = np.array(data)
        self.vectors_register = []
        for vID, vector in enumerate(self.data):
            self.vectors_register.append(VectorRegister(vID, vector))

        self.data_size = len(self.data)
        self.nClusters = nClusters
        self.partitions = [None] * self.data_size
        self.distances = [None] * self.data_size

    def calculate(self):
        start = time.clock()
        self.set_initial_centroids()

        iterations = 0
        while True:
            self.repartition()
            self.calculate_centroids()

            iterations += 1
            if self.stable_centroids(): break

        finish = time.clock()

        return {
            'iteration': iterations,
            'centroids': [c.vector for c in self.centroids],
            'totalSquaredError': self.totalSquaredError(),
            'time': (finish - start)
        }

    def repartition(self):
        for vID, vector in enumerate(self.data):
            vr = self.vectors_register[vID]

            distances = [float('inf')] * self.nClusters
            if vr.centroid: distances[vr.centroid.id] = vr.distance

            for centroid in self.centroids:
                # Calculate distance if the centroid is not static, or the
                # current centroid of the vector has moved
                if not centroid.static or vr.current_centroid_moved:
                    distances[centroid.id] = self.distance(vector, centroid.vector)

            min_dist = min(distances)
            closests_vector_index = distances.index(min_dist)

            vr.set_centroid(self.centroids[closests_vector_index])
            vr.distance = min_dist

    def calculate_centroids(self):
        for centroid in self.centroids:
            cluster_vectors = np.array([vr.vector for vr in self.vectors_register if vr.centroid == centroid])
            new_centroid_vector = np.mean(cluster_vectors, 0)
            if np.array_equal(centroid.vector, new_centroid_vector):
                centroid.static = True
            else:
                centroid.static = False
                centroid.vector = new_centroid_vector


    def set_initial_centroids(self):
        centroids = []
        for cID in range(self.nClusters):
            centroid_vector = self.data[randint(0, self.data_size-1)]
            centroid = Centroid(cID, centroid_vector)
            centroids.append(centroid)
        self.centroids = centroids

    def distance(self, v1, v2):
        return np.linalg.norm(v1 - v2)

    def draw(self):
        plt.scatter(self.data[:, 0], self.data[:, 1])

        centroids = np.array([c.vector for c in self.centroids])
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='d', s=150)

        plt.draw()
        plt.show()

    def stable_centroids(self):
        moved = [vr.centroid_moved() for vr in self.vectors_register]
        return set(moved) == set([False])

    def totalSquaredError(self):
        error = 0
        for vr in self.vectors_register:
            error += self.distance(vr.vector, vr.centroid.vector)**2
        return error


class VectorRegister:
    id = None
    vector = None
    centroid = None
    prev_centroid = None
    distance = None

    def __init__(self, id, vector):
        self.id = id
        self.vector = vector

    def current_centroid_moved(self):
        if not centroid: return False
        return not centroid.static

    def set_centroid(self, centroid):
        self.old_centroid = self.centroid
        self.centroid = centroid

    def centroid_moved(self):
        return self.old_centroid != self.centroid

class Centroid:
    id = None
    vector = None
    static = False
    def __init__(self, id, vector):
        self.id = id
        self.vector = vector

    def __eq__(self, other):
        if other == None: return False
        return self.id == other.id
