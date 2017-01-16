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

## References
[1] S. P. Lloyd. *Least squares quantization in pcm*. Information Theory, IEEE Trans. on, 28(2):129–137, 1982.

[2] C. Elkan. *Using the triangle inequality to accelerate k-means*. In International Conference om Machine Learning, pages 147–153, 2003.
