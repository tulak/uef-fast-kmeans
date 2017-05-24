import sys
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed as pry
from fast_kmeans import FastKMeans
from ci import calculate_ci
import datetime
import csv
from io import StringIO

def loadData(filename):
    data = []
    with open(filename,'r') as csvfile:
        io = StringIO(csvfile.read().replace('\t', ' '))
        reader = csv.reader(io, delimiter=' ')
        for row in reader:
            data.append(row)
    data = [[float(cell) for cell in row if cell != ''] for row in data]
    return data

def test_dataset(name, nClusters):
    data_file = "datasets/" + str(name) + ".txt"
    gths_file = "datasets/" + str(name) + "_gt.txt"

    data = loadData(data_file)
    gths = loadData(gths_file)

    km = FastKMeans(data, nClusters)
    result = km.calculate()

    ci = calculate_ci(result['centroids'], gths)
    result['ci'] = ci
    result['km'] = km

    return result

datasets = {
    'a1': 20,
    'a2': 35,
    'a3': 50,
    's1': 15,
    's2': 15,
    's3': 15,
    's4': 15,
    'dim032': 16,
    'unbalance': 8
}

results = {}
repeat = 100
# pry()
for name, clusters in datasets.items():
    print('Dataset: '+ name + ' TIME:', datetime.datetime.now())
    set_results = []
    for i in range(repeat):
        print(i, end=' ')
        sys.stdout.flush()
        r = test_dataset(name, clusters)
        set_results.append([r['totalSquaredError'], r['ci'], r['time']])
    result = np.mean(set_results, 0)
    successful = sum([1 for r in set_results if r[1] == 0])
    result = np.append(result, successful)
    results[name] = result
    print('')
    print('Name: ', name, round(result[0]), result[1], result[2], result[3])

pry()
