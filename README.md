# futhark-mixtures
This library currently implements three clustering algorithms for sparse datasets.
1. k-means
2. Spherical k-means
3. Gaussian mixture model (diagonal covariance matrix)

It also contains code for von-Mises-Fisher mixtures, however due to a missing implementation of the modified Bessel function in Futhark, this implementation is incomplete.

## Running small examples

1. In the futhark folder, run `futhark pkg sync` to load external dependencies.
2. In tests, compile the test_(imp).fut file for the implementation of choice with the futhark backend of choice: `futhark cuda test_gaussian_diag.fut`
3. Pipe the example dataset to the executable: `cat example_dataset_binary | ./test_gaussian_diag -e test_gaussian_diag`.