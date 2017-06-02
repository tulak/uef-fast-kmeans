# Usage
```python
# Import the class
from fast_kmeans import FastKMeans

data = [
  [15,16],[14,16],[15,14],[16,14],
  [0,0], [0,1],[1,0],[1,1],[2,1]
]
nClusters = 2

# Create kmeans object
kmeans = FastKMeans(data, nClusters)
# Run the algorithm
results = kmeans.calculate()
# results => {'centroids': [array([ 15.,  15.]), array([ 0.8,  0.6])],
#             'iteration': 3,
#             'time': 0.0009430000000001382,
#             'totalSquaredError': 9.9999999999999982}
```
The result of `kmeans.calculate()` is a dictionary with appropriate result values
