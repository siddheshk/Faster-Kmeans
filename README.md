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

## Running the Code



## References
[1] S. P. Lloyd. *Least squares quantization in pcm*. Information Theory, IEEE Trans. on, 28(2):129–137, 1982.

[2] C. Elkan. *Using the triangle inequality to accelerate k-means*. In International Conference om Machine Learning, pages 147–153, 2003.
