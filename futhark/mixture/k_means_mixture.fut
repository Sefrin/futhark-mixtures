import "mixture"
import "../dataset/mixture_dataset"

module k_means_mixture (dataset: mixture_dataset): (mixture with data = dataset.data_rep) = {
    -- module DAT = dataset
    module V = dataset.V
    module I = dataset.I
    module C = i32
    type val_t = V.t
    type index_t = I.t

    type theta_sized [k][d] = [k][d]val_t
    type posteriors_sized [n][k] = [n][k]val_t

    type~ theta = theta_sized [][] -- the cluster centers
    type~ posteriors = posteriors_sized [][] -- memberships instead of probabilities
    type~ data = dataset.data_rep
    type conv_limit_t = C.t

    let i0 = (I.i32 0)
    let f0 = (V.f32 0f32)

    let closest_point (p1: (i64, val_t)) (p2: (i64, val_t)): (i64, val_t) =
        if p1.1 V.< p2.1 then p1 else p2

    let argmin [k] (dist: [k]val_t) = 
                let (i, _) = foldl (\acc (i, dist) -> closest_point acc (i, dist))
                                                        (0, V.inf)
                                                        (zip (iota k) dist)
                    in I.i64 i
                
    let eval [k] (X: data) (params: theta_sized [k][]): posteriors =
        -- sum mu^2
        let cluster_squared_sum = map (\c -> V.sum (map (\x -> x V.* x) c)) params
        
        -- x^2 - 2 * x * mu 
        let mul_op =  \data_val cluster_val -> (data_val V.- ((V.f32 2) V.* cluster_val)) V.* data_val
        let sparse_distances = dataset.apply_k_semirings X params mul_op (V.+) f0
        in map (\dist_row ->
                map2 (V.+) dist_row cluster_squared_sum
            ) sparse_distances

    let maximize [k] (X: data) (_: theta_sized [k][]) (posts: posteriors): theta = 
        
        let membership = map argmin posts
        let cluster_sums = dataset.map_index_reduce_by_key X membership k (\x _ _ -> x) (V.+) f0
        -- count elements
        let ones = map (\_ -> I.i32 1) membership
        let center_counts = reduce_by_index (replicate k i0) (I.+) i0 (map I.to_i64 membership) ones
        -- need to divide by number of elements 
        in map2 (\center count -> map (\x -> x V./ (V.i64 (I.to_i64 (if count I.== i0 then I.i32 1 else count)))) center) cluster_sums center_counts
        

    let init_theta (X: data) (k: i64): theta =
                        dataset.take X k

    let check_converged (old_posts: posteriors) (new_posts: posteriors) (conv_limit: conv_limit_t): bool = 
        let old_membership = map argmin old_posts
        let new_membership = map argmin new_posts
        let delta = i32.sum (map (\b -> if b then 0 else 1)
                                   (map2 (I.==) old_membership new_membership))
        in delta <= conv_limit

    let compute_objective_function (posts: posteriors): f64 =
                        V.to_f64 (V.sum (map V.sum posts))
                        
    let get_means [k] (params: theta_sized [k][]): [][]f64 = map (\row -> map V.to_f64 row) params
                        
    let get_weights [k] (_: theta_sized [k][]): []f64 =
                    (replicate k (1.0f64 / (f64.i64 k)))
}       