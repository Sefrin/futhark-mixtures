import "../futhark/em"
import "../futhark/dataset/sparse_dataset"
import "../futhark/mixture/spherical_k_means"
import "k"

module dataset = sparse_dataset f32 i64
module mixture = spherical_k_means dataset
module k_means_em = em mixture

-- ==
-- entry: test_spherical_k_means
-- nobench input { [1,2
       --  ,1,4
       --  ,1,0
       --  ,10,2
       --  ,10,4
       --  ,10,0] [0,1,0,1,0,1,0,1,0,1,0,1] [0,2,4,6,8,10,12]  }
entry test_spherical_k_means [nnz][np1] (values: [nnz]f32) (indices: [nnz]i64) (pointers: [np1]i64) = 
    let values = map dataset.V.f32 values
    let indices = map dataset.I.i64 indices
    let pointers = map dataset.I.i64 pointers
    let data = dataset.mk_from_csr values indices pointers (dataset.I.i64 2)
    let normalized_data = dataset.normalize data
    let (t, i, _, hist) = k_means_em.fit (mixture.C.i64 0i64) k 30 normalized_data
    let means = mixture.get_means t
    let weights = mixture.get_weights t
    in (i, f64.sum weights, means, hist)