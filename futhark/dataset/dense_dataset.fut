import "mixture_dataset"
import "data_types"

module dense_dataset (R: real) (I: integral): (mixture_dataset 
                                                    with val_t = R.t 
                                                    with index_t = I.t
                                                ) = {
    module V = R
    module IND = I
    type val_t = R.t
    type index_t = I.t
    type~ data_rep = [][]val_t

    let get_d [n][d] (_: [n][d]val_t) = d
    let get_n [n][d] (_: [n][d]val_t) = n
    let ind_to_i64 (i: index_t) = IND.to_i64 i
    let mk_from_dense [n][d] (data: [n][d]val_t) = data

    let mk_from_csr [nnz][np1] (vals: [nnz]val_t) (_: [nnz]index_t) (_: [np1]index_t) (columns: index_t): data_rep =
        -- DUMMY IMPLEMENTATION
        let n =  np1 -1 
        let d = ind_to_i64 columns in
        unflatten n d (replicate (n*d) vals[0])

    let take (X: data_rep) (k: i64): [k][]val_t = take k X

    let apply_k_semirings 't [k][d] (X: data_rep)
                                (mixture_vectors: [k][d]t)
                                (pairwise_op: val_t -> t -> val_t) 
                                (add_op: val_t -> val_t -> val_t)
                                (add_ne: val_t) : [][k]val_t =
        let X = X :> [k][d]val_t
        in map (\row ->
            map (\vector -> 
                reduce add_op add_ne 
                    (map2 pairwise_op row vector)
            ) mixture_vectors
        ) X
}