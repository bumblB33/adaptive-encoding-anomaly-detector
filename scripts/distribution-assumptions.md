# Measuring Assumptions regarding IEEE-CIS Data

## LOF

1. Parameter Testing
   a. number of neighbors (n_neighbors : int = 20)
   - Reporting options - iterative scaling to determine impact of n_neighbors > 20, and sample fallback impact

   b. n_neighbors algorithm (Minkowski, Euclidean)
   - defaults to brute force if data is too sparse

2. Distribution Assumptions & Testing

- Gaussian? Power Law? Sparse? Skewed? Scatter? Linear? Clustered?

For EEG data, DTW for time series?

- https://dynamictimewarping.github.io/
- pip install dtw-python
