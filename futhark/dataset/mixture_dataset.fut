module type mixture_dataset = {
    module V : real
    module I : integral
    type val_t = V.t
    type index_t = I.t
    type~ data_rep

    val mk_from_dense [n][d]: [n][d]val_t -> data_rep
    val mk_from_csr [nnz][np1]: (values: [nnz]val_t) -> 
                                (indices: [nnz]index_t) ->
                                (pointers: [np1]index_t) ->
                                (columns: index_t)
                                -> data_rep 

    -- for each vector, for each row of data_rep:
        -- apply (mult_op a b) to the data where a is in data and b in the vector
        -- then reduce with (add_op ne) the results of mult  
    val apply_k_semirings 't [k][d]: (X: data_rep)  
                            -> (k_vectors: [k][d]t)
                            -> (pairwise_op: val_t -> t -> val_t) 
                            -> (add_op: val_t -> val_t -> val_t)
                            -> (add_ne: val_t)
                            -> [][k]val_t

    val map_index_reduce_by_key [n]: (X: data_rep) 
                            -> (keys: [n]index_t) 
                            -> (n_distict_keys: i64) -- how many bins do we have for keys
                            -> (map_f: val_t -> index_t -> index_t -> val_t) -- map value row_index col_index
                            -> (red_op: val_t -> val_t -> val_t)
                            -> (ne: val_t)
                            ->  [n_distict_keys][]val_t

    val map_index_reduce: (X: data_rep) 
                            -> (d: i64) 
                            -> (map_f: val_t -> index_t -> index_t -> val_t) -- map value row_index col_index
                            -> (red_op: val_t -> val_t -> val_t)
                            -> (ne: val_t) 
                            -> [d]val_t

    val normalize: (X: data_rep)
                            -> data_rep

    val take: (X: data_rep) 
                -> (k: i64) 
                -> [k][]val_t

    val get_d: data_rep -> i64
    val get_n: data_rep -> i64
    val ind_to_i64: index_t -> i64
}