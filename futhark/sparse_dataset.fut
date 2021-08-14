import "mixture_dataset"
import "utils"
import "lib/github.com/diku-dk/sorts/radix_sort"

module sparse_dataset (R: real) (I: integral): (mixture_dataset 
                                                    with val_t = R.t 
                                                    with index_t = I.t
                                                ) = {
    module V = R
    module I = I
    type val_t = R.t
    type index_t = I.t
    type~ data_rep = {n: index_t, d: index_t, coo: [](val_t, index_t, index_t), pointers: []index_t}
     
    let ind_to_i64 (index: index_t) = I.to_i64 index
    let get_d (X: data_rep) = ind_to_i64 X.d
    let get_n (X: data_rep) = ind_to_i64 X.n

    let mk_from_csr [nnz][np1] (vals: [nnz]val_t) (indices: [nnz]index_t) (pointers: [np1]index_t) (columns: index_t): data_rep =
        let n = np1 - 1
        let vector_starts = init pointers :> [n]index_t
        let vector_ends = tail pointers :> [n]index_t
        let vector_nnz = map2 (\start end -> (ind_to_i64 end) - (ind_to_i64 start)) vector_starts vector_ends
        let flags = mkFlagArray vector_nnz 0i32 (replicate n 1i32) :> [nnz]i32
        let row_indices = (map (\x -> I.i32 (x-1))) (scan (+) 0 flags)
        let coo = zip3 vals row_indices indices
            in {n = I.i64 n, d = columns, coo = coo, pointers = pointers}

    let mk_from_dense [n][d] (_: [n][d]val_t): data_rep =
        -- DUMMY IMPLEMENTATION
        let coo = zip3 [R.nan] [I.lowest] [I.lowest]
            in {n = I.i64 n, d = I.i64 d, coo = coo, pointers = [I.lowest]}

    let apply_k_semirings 't [k][d] (X: data_rep)
                                (mixture_vectors: [k][d]t)
                                (pairwise_op: val_t -> t -> val_t) 
                                (add_op: val_t -> val_t -> val_t)
                                (add_ne: val_t) : [][k]val_t =             
                let (values, _, col_indices) = unzip3 X.coo
                in map (\row -> 
                    map (\vector->     
                        let index_start = I.to_i64 X.pointers[row]
                        let nnz =  (I.to_i64 X.pointers[row+1]) - index_start 
                            let row_dist = add_ne
                            let row_dist = 
                                loop row_dist for j < nnz do
                                    let element_value = values[index_start+j]
                                    let column = I.to_i64 col_indices[index_start+j]
                                    let vector_value = vector[column]
                                    let value = pairwise_op element_value vector_value
                                    in add_op row_dist value
                                in row_dist
                            ) mixture_vectors
                    ) (iota (I.to_i64 X.n))

    let take (X: data_rep) (k: i64): [k][]val_t =
        let d = I.to_i64 X.d
        let first_k_pointers = take (k+1) X.pointers
        let k_vector_starts = init first_k_pointers :> [k]index_t
        let k_vector_ends = tail first_k_pointers :> [k]index_t
        let first_k_nnz = map2 (\start end -> I.to_i64 (end I.- start)) k_vector_starts k_vector_ends

        let first_k_total_nz = reduce (+) 0 first_k_nnz
        let (values, row_indices, col_indices) = unzip3 X.coo
        in unflatten k d 
            (scatter (replicate (k*d) (R.f32 0f32)) 
                    (map2 (\row col -> I.to_i64 (row I.* X.d I.+ col)) (take first_k_total_nz row_indices) (take first_k_total_nz col_indices))
                    (take first_k_total_nz values))

    let map_index_reduce_by_key [n] (X: data_rep) 
                                (keys: [n]index_t)
                                (n_distinct_keys: i64) 
                                (map_f: val_t -> index_t -> index_t -> val_t) -- map value row_index col_index
                                (red_op: val_t -> val_t -> val_t)
                                (ne: val_t) : [n_distinct_keys][]val_t = 
            let d = I.to_i64 X.d
            let (values, row_indices, col_indices) = unzip3 X.coo       
            let map_vals = map3 map_f values row_indices col_indices
            let flat_results = reduce_by_index (replicate (n_distinct_keys*d) ne)
                                                red_op ne 
                                                (map2 (\row col -> I.to_i64 (( (I.*) X.d keys[I.to_i64 row]) I.+ col))
                                                row_indices col_indices) map_vals
                in unflatten n_distinct_keys d flat_results

    let map_index_reduce (X: data_rep) 
                            (d: i64) 
                            (map_f: val_t -> index_t -> index_t -> val_t) -- map value row_index col_index
                            (red_op: val_t -> val_t -> val_t)
                            (ne: val_t) : [d]val_t = 
        -- let d = I.to_i64 X.d
        let (values, row_indices, col_indices) = unzip3 X.coo       
        let map_vals = map3 map_f values row_indices col_indices
        in reduce_by_index (replicate d ne)
                            red_op ne 
                            (map (\col -> I.to_i64 col) col_indices)
                            map_vals

    let normalize (X: data_rep): data_rep = 
                let (values, row_indices, col_indices) = unzip3 X.coo
                let row_sums = map (\row ->   
                    let index_start = I.to_i64 X.pointers[row]
                    let nnz =  (I.to_i64 X.pointers[row+1]) - index_start 
                    let row_sum = V.f32 0
                    let row_sum = 
                        loop row_sum for j < nnz do
                            let element_value = values[index_start+j]
                            in row_sum V.+ (element_value V.* element_value)
                        in V.sqrt row_sum
                    ) (iota (I.to_i64 X.n))
                let new_values = map2 (\v r -> v V./ row_sums[I.to_i64 r]) values row_indices
                let new_coo = zip3 new_values row_indices col_indices
                in {n=X.n, d=X.d, coo = new_coo, pointers = X.pointers}


    let sort_by_keys [n] (X: data_rep) (keys: [n]i64) (num_bits: i32) : data_rep = 
        let starts = init X.pointers :> [n]index_t
        let ends = tail X.pointers :> [n]index_t
        let nnz = length X.coo
        let nnz_row = map2 (\start end -> I.to_i64 (end I.- start)) starts ends
        
        -- sort iota by given keys
        let zipped = zip keys (iota n)
        let new_order = map (.1) (radix_sort_int_by_key (.0) num_bits i64.get_bit zipped)

        -- find new number of nnz in each row
        let new_nnz = map (\i -> nnz_row[i]) new_order
        -- make it an exclusive scan but keep the whole sum as last element for csr pointer array
        let new_nnz_scan = (scan (+) 0 new_nnz)
        let new_pointers = map (\i -> if i == 0 then I.i64 0 else I.i64 new_nnz_scan[i-1]) (indices X.pointers)
            
        let flags = mkFlagArray new_nnz 0i32 (map (\x -> i32.i64 (x+1)) new_order)  :> [nnz]i32  
        let segmented_iota = map (\x -> x-1 ) (segscan (+) 0i64 (map bool.i32 flags) (replicate nnz 1))
        let segmented_replicate = map (\x -> x-1) (segscan (+) 0i32 (map bool.i32 flags) flags)

        let (values, row_indices, col_indices) = unzip3 X.coo
        let new_row_indices = map2 (\colindex old_row -> row_indices[(I.to_i64 X.pointers[old_row])+colindex]) segmented_iota segmented_replicate
        let new_col_indices = map2 (\colindex old_row -> col_indices[(I.to_i64 X.pointers[old_row])+colindex]) segmented_iota segmented_replicate
        let new_values = map2 (\colindex old_row -> values[(I.to_i64 X.pointers[old_row])+colindex]) segmented_iota segmented_replicate
        let new_coo = zip3 new_values new_row_indices new_col_indices
         in {n=X.n, d=X.d, coo = new_coo, pointers = new_pointers}
}