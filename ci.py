import numpy as np

def calculate_ci(centroids, gths):
    centroids = np.array(centroids)
    gths = np.array(gths)

    nCentroids = len(centroids)
    nGths = len(gths)

    map_centroids = [None] * nCentroids
    map_gths = [None] * nGths

    for cID, centroid in enumerate(centroids):
        lengths = [0] * nGths
        for gID, gth in enumerate(gths):
            lengths[gID] = np.linalg.norm(centroid - gth)

        min_length = min(lengths)
        closest_gth_id = lengths.index(min_length)
        map_centroids[cID] = closest_gth_id

    for gID, gth in enumerate(gths):
        lenths = [0] * nCentroids
        for cID, centroid in enumerate(centroids):
            lengths[cID] = np.linalg.norm(gth - centroid)
        min_length = min(lengths)
        closest_centroid_id = lengths.index(min_length)
        map_gths[gID] = closest_centroid_id

    centroid_ophrams = abs(len(set(map_centroids)) - nCentroids)
    gth_ophrams = abs(len(set(map_gths)) - nGths)

    return max(centroid_ophrams, gth_ophrams)
