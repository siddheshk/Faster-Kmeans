# Faster-Kmeans

This repository contains the code and the datasets for running the experiments for the following paper:

>Siddhesh Khandelwal and Amit Awekar, *"Faster K-Means Cluster Estimation"*. To appear in Proceedings of European
Conference on Information Retrieval (ECIR), 2017.

## Code

The code is present in the **Code** folder.
* **kmeans.py** : Python implementation of the Lloyd's Algorithm [1]
* **heuristic_kmeans.py** : Python implementation of Lloyd's Algorithm [1] augmented with our heuristic
* **triangleInequality.py** : Python implementation of the K-means with Triangle Inequality Algorithm [2]
* **heuristic_triangleinequality.py** : Python implementation of the K-means with Triangle Inequality Algorithm [2] augmented with our heuristic

## Datasets

The datasets used in our paper are present in the **Datasets** folder.

## Running the Code

There are two types of files:
* Algorithms not augmented with our heuristic.
* Algorithms augmented with our heuristic

If the code if of the first type, it can be run by calling the following function
```python
Kmeans(k, pointList, kmeansThreshold, initialCentroids=None)
# k = Number of Clusters
# pointList = List of n-dimensional points (Every point should be a list)
# kmeansThreshold = Percentage Change in Mean Squared Error (MSE) below which the algorithm should stop. Used as a stopping criteria
# initialCentroids (optional) = Provide initial seeds for centroids (List of points)
```

If the code if of the second type, it can be run by calling the following function
```python
Kmeans(k, pointList, kmeansThreshold, centroidsToRemember, initialCentroids=None)
# k = Number of Clusters
# pointList = List of n-dimensional points (Every point should be a list)
# kmeansThreshold = Percentage Change in Mean Squared Error (MSE) below which the algorithm should stop. Used as a stopping criteria
# centroidsToRemember = The value of k'. This value is the percentage of k to be used as the Candidate Cluster List (CCL)
# initialCentroids (optional) = Provide initial seeds for centroids (List of points)
```

## Approach

Major bottleneck of K-means clustering is the computation of data point to cluster centroid distance. For a dataset with `n` data points and `k` clusters, each iteration of K-means performs `n x k` such distance computations. To overcome this bottleneck, we maintain a list of candidate clusters for each data point. Let size of this list be `k'`. We assume that `k'` is significantly smaller than `k`. We build this candidate cluster list based on top `k'` nearest clusters to the data point after first iteration of K-means. Now each iteration of K-means will perform only `n x k'` distance computations.

Motivation for this approach comes from the observation that data points have tendency to go to clusters that were closer in the previous iteration. In k-means, we compute distance of a data point to every cluster even though the point has extremely little chance of being assigned to it. The figure below shows an example execution of k-means for a synthetic dataset of 100,000 points in 10 dimensions which needs to be partitioned into 100 clusters. X axis represents iteration of the algorithm and Y axis represents percentage of points that get assigned to a particular cluster. For example, in iteration 2, about 75 percent points got reassigned to same cluster as in iteration 1 and about 10 percent points got assigned to a cluster that was second closest cluster in previous iteration. As we progress through the algorithm, more points get assigned to same cluster or clusters that were close in previous iteration. 
<p align="center">
  <img src="Images/table.png" alt="Point Distribution per Iteration" width="500" align="middle"/>
</p>

## Experiments

### Comparing with K-means with Triangle Inequality [2]
|           |                     | **k' = 20** | **k' = 30** | **k' = 40** | **k' = 50** | **k' = 60** |
|-----------|---------------------|:-------:|:-------:|:-------:|:-------:|:-------:|
| **Birch**     | % Difference in MSE |  -0.11  |   0.04  |    0    |    0    |    0    |
|           | Speedup             |   3.05  |   2.48  |   2.01  |   1.68  |   1.41  |
| **Covtype**   | % Difference in MSE |   0.21  |   0.02  |    0    |    0    |    0    |
|           | Speedup             |   2.32  |   1.81  |   1.61  |   1.55  |   1.42  |
| **Mnist**     | % Difference in MSE |   1.30  |   0.72  |   0.51  |   0.43  |   0.37  |
|           | Speedup             |   1.90  |   1.68  |   1.59  |   1.48  |   1.47  |
| **KDDCup**    | % Difference in MSE |   0.81  |   0.11  |   0.08  |  -0.18  |    0    |
|           | Speedup             |   1.44  |   1.33  |   1.42  |   0.88  |   1.18  |
| **Synthetic** | % Difference in MSE |   0.19  |   0.11  |   0.06  |   0.03  |   0.01  |
|           | Speedup             |   2.90  |   2.28  |   1.87  |   1.51  |   1.36  |

### Comparing with K-means++ [2] 

Note that as K-means++ is just a seeding technique, K-means with Triangle Inequality is used as the algorithm after seeding

|           |                     | **k' = 20** | **k' = 30** | **k' = 40** | **k' = 50** | **k' = 60** |
|-----------|---------------------|:-------:|:-------:|:-------:|:-------:|---------|
| **Birch**     | % Difference in MSE |    0    |    0    |    0    |    0    |    0    |
|           | Speedup             |   3.14  |   2.26  |   1.93  |   1.67  |   1.31  |
| **Covtype**   | % Difference in MSE |   0.03  |    0    |    0    |    0    |    0    |
|           | Speedup             |   2.02  |   1.82  |   1.63  |   1.38  |   1.20  |
| **Mnist**     | % Difference in MSE |   1.36  |   0.71  |   0.36  |   0.18  |   0.09  |
|           | Speedup             |   1.47  |   1.44  |   1.26  |   1.19  |   1.15  |
| **KDDCup**    | % Difference in MSE |   0.70  |   0.15  |   0.02  |  -0.01  |    0    |
|           | Speedup             |   1.60  |   1.15  |   1.02  |   0.99  |   1.02  |
| **Synthetic** | % Difference in MSE |   0.15  |   0.08  |   0.04  |   0.01  |   0.01  |
|           | Speedup             |   2.45  |   1.97  |   1.71  |   1.35  |   1.17  |


## References
[1] S. P. Lloyd. *Least squares quantization in pcm*. Information Theory, IEEE Trans. on, 28(2):129–137, 1982.

[2] C. Elkan. *Using the triangle inequality to accelerate k-means*. In International Conference om Machine Learning, pages 147–153, 2003.
