import "../futhark/em"
import "../futhark/sparse_dataset"
import "../futhark/k_means_mixture"
import "k"

module dataset = sparse_dataset f32 i64
module mixture = k_means_mixture dataset
module k_means_em = em mixture

-- ==
-- entry: test_bbc_em_k_means
-- compiled input @ ../../skmeans/data/futhark/bbc_binary
entry test_bbc_em_k_means [nnz][np1] (values: [nnz]f32) (indices: [nnz]i64) (pointers: [np1]i64) = 
    let values = map dataset.V.f32 values
    let indices = map dataset.I.i64 indices
    let pointers = map dataset.I.i64 pointers
    let data = dataset.mk_from_csr values indices pointers (dataset.I.i64 29126i64)
    let (t, i, _, hist) = k_means_em.fit (mixture.C.i64 0i64) k 30 data
    let means = mixture.get_means t
    let weights = mixture.get_weights t
    in (i, f64.sum weights, means, hist)
 