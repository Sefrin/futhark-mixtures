import "../futhark/em"
import "../futhark/sparse_dataset"
import "../futhark/gaussian_diag"
import "k"

module dataset = sparse_dataset f64 i64
module mixture = gaussian_diag dataset
module gaussian_em = em mixture


-- = =
-- entry: test_gaussian_diag_custom
-- nobench input { [1,2
       --  ,1,4
       --  ,1,0
       --  ,10,2
       --  ,10,4
       --  ,10,0] [0,1,0,1,0,1,0,1,0,1,0,1] [0,2,4,6,8,10,12]  }
entry test_gaussian_diag_custom [nnz][np1] (values: [nnz]f32) (indices: [nnz]i64) (pointers: [np1]i64) = 
    let values = map dataset.V.f32 values
    let indices = map dataset.I.i64 indices
    let pointers = map dataset.I.i64 pointers
    let data = dataset.mk_from_csr values indices pointers (dataset.I.i64 2)
    let (t, i, _, hist) = gaussian_em.fit (mixture.C.f32 0.0001f32) k 200 data
    let means = mixture.get_means t
    let weights = mixture.get_weights t
    in (i, f64.sum weights, means, hist)

 
-- ==
-- entry: test_gaussian_diag_bbc
-- compiled input @ ../../skmeans/data/futhark/bbc_binary
entry test_gaussian_diag_bbc [nnz][np1] (values: [nnz]f32) (indices: [nnz]i64) (pointers: [np1]i64) = 
    let values = map dataset.V.f32 values
    let indices = map dataset.I.i64 indices
    let pointers = map dataset.I.i64 pointers
    let data = dataset.mk_from_csr values indices pointers (dataset.I.i64 29126i64)
    let (t, i, _, hist) = gaussian_em.fit (mixture.C.f32 0.001f32) k 10 data
    let means = mixture.get_means t
    let weights = mixture.get_weights t
    in (i, f64.sum weights, means, hist)

 
-- ==
-- entry: test_gaussian_diag_movielens
-- nobench compiled input @ ../../skmeans/data/futhark/movielens_binary
entry test_gaussian_diag_movielens [nnz][np1] (values: [nnz]f32) (indices: [nnz]i64) (pointers: [np1]i64) = 
    let values = map dataset.V.f32 values
    let indices = map dataset.I.i64 indices
    let pointers = map dataset.I.i64 pointers
    let data = dataset.mk_from_csr values indices pointers (dataset.I.i64 131263i64)
    let (t, i, _, hist) = gaussian_em.fit (mixture.C.f32 0.001f32) k 1000 data
    let means = mixture.get_means t
    let weights = mixture.get_weights t
    in (i, f64.sum weights, means, hist)

 
-- ==
-- entry: test_gaussian_diag_nytimes
-- nobench compiled input @ ../../skmeans/data/futhark/nytimes_binary
entry test_gaussian_diag_nytimes [nnz][np1] (values: [nnz]f32) (indices: [nnz]i64) (pointers: [np1]i64) = 
    let values = map dataset.V.f32 values
    let indices = map dataset.I.i64 indices
    let pointers = map dataset.I.i64 pointers
    let data = dataset.mk_from_csr values indices pointers (dataset.I.i64 102661i64)
    let (t, i, _, hist) = gaussian_em.fit (mixture.C.f32 0.0000000001f32) k 1000 data
    let means = mixture.get_means t
    let weights = mixture.get_weights t
    in (i, f64.sum weights, means, hist)

 
-- ==
-- entry: test_gaussian_diag_scrna
-- nobench compiled input @ ../../skmeans/data/futhark/scrna_binary
entry test_gaussian_diag_scrna [nnz][np1] (values: [nnz]f32) (indices: [nnz]i64) (pointers: [np1]i64) = 
    let values = map dataset.V.f32 values
    let indices = map dataset.I.i64 indices
    let pointers = map dataset.I.i64 pointers
    let data = dataset.mk_from_csr values indices pointers (dataset.I.i64 26485i64)
    let (t, i, _, hist) = gaussian_em.fit (mixture.C.f32 0.0001f32) k 1000 data
    let means = mixture.get_means t
    let weights = mixture.get_weights t
    in (i, f64.sum weights, means, hist)
 