import "../futhark/em"
import "../futhark/dataset/sparse_dataset"
import "../futhark/mixture/gaussian_diag"
import "k"

module dataset = sparse_dataset f64 i64
module mixture = gaussian_diag dataset
module gaussian_em = em mixture


-- ==
-- entry: test_gaussian_diag
-- nobench input { [1,2
       --  ,1,4
       --  ,1,0
       --  ,10,2
       --  ,10,4
       --  ,10,0] [0,1,0,1,0,1,0,1,0,1,0,1] [0,2,4,6,8,10,12]  }
entry test_gaussian_diag [nnz][np1] (values: [nnz]f32) (indices: [nnz]i64) (pointers: [np1]i64) = 
    let values = map dataset.V.f32 values
    let indices = map dataset.I.i64 indices
    let pointers = map dataset.I.i64 pointers
    let data = dataset.mk_from_csr values indices pointers (dataset.I.i64 2)
    let (t, i, _, hist) = gaussian_em.fit (mixture.C.f32 0.0001f32) k 200 data
    let means = mixture.get_means t
    let weights = mixture.get_weights t
    in (i, f64.sum weights, means, hist)