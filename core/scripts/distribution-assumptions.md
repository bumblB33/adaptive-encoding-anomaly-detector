# Measuring Assumptions regarding IEEE-CIS Data

## LOF

1. Parameter Testing
    a. n_neighbors : int = 20
    * Reporting options - iterative scaling to determine impact of n_neighbors > 20, and sample fallback impact

    b. n_neighbors algorithm
    * defaults to brute force if data is too sparse
    * override with 

