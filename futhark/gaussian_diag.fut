import "mixture"
import "mixture_dataset"
import "lib/github.com/diku-dk/cpprandom/random"

module gaussian_diag (dataset: mixture_dataset): (mixture with data = dataset.data_rep) = {
    module V = dataset.V
    module I = dataset.I
    module C = f32
    type val_t = V.t
    type index_t = I.t

    type theta_sized [k][d] = {alphas: [k]val_t, mus: [k][d]val_t, sigmas: [k][d]val_t} -- arguments -- alpha, mean and diagonal of covariance matrix
    type posteriors_sized [n][k] = [n][k]val_t 

    type~ theta = theta_sized [][] -- the cluster centers
    type~ posteriors = posteriors_sized [][] -- memberships instead of probabilities    
    type~ data = dataset.data_rep
    type conv_limit_t = C.t

    module rng_engine = minstd_rand
    module rand_val = uniform_real_distribution V rng_engine
    let seed = 13i32

    let logsumexp [n] (arr: [n]val_t): val_t =
                let max_x = V.maximum arr
                let exps = map (\p -> V.(exp (p-max_x))) arr    
                in V.(max_x + (log (sum exps)))

    -- produces the log probabilities for each component
    let eval [k][d] (X: data) (params: theta_sized [k][d]): posteriors =
        -- -0.5 because log (1/sqrt(x)) = -0.5 log(x)
        -- in this case x =  
        let log_normalization = map (\sigs ->      
                        let log2pi = V.log V.((V.f32 2) * V.pi)
                        let logsigmasum = V.sum (map (\s -> V.log s) sigs)
                            in V.((V.f32 (-0.5)) * ((V.i64 d) * log2pi + logsigmasum))
                    ) params.sigmas
        --mu squared / 2 sigma
        let mu_squared_scaled_sum = map2 (\mus sigs ->
                                            V.sum (
                                                map2 (\m s -> V.(m * m / ((V.f32 2) * s))) mus sigs
                                            )
                                    ) params.mus params.sigmas

        let dot_op = \data_val theta_tup -> 
                let mu = theta_tup.0
                let sigma = theta_tup.1
                -- in ((data_val V.- (V.f32 2) V.* mu) V.* data_val) V./ ((V.f32 2) V.* sigma)
                in V.(((data_val - (V.f32 2) * mu) * data_val) / ((V.f32 2) * sigma))
        
        -- zip mus and sigmas to allow access from semiring function
        let mu_sigmas = map2 (\mus sigmas ->
                            zip mus sigmas
                        ) params.mus params.sigmas

        let unscaled_log_probs = dataset.apply_k_semirings X mu_sigmas dot_op (V.+) (V.f32 0f32)
        -- so far we have sum of positive exponents.... 

        let scaled_log_probs = map (\row -> 
                                let component_probs = map4 (\probs log_norm mu_sq alpha ->
                                                            let logalpha = V.log alpha
                                                        -- add missing mu squared sum term, negate and scale prob with alpha and denominator
                                                            in V.(logalpha + log_norm - (probs + mu_sq))
                                                        ) row log_normalization mu_squared_scaled_sum params.alphas
                                in component_probs
                                -- sum of probs to normalize
                                -- let component_prob_sum = V.sum component_probs
                                -- -- divide by sum
                                -- in map (\p -> p V./ component_prob_sum) component_probs
                            ) unscaled_log_probs

        in scaled_log_probs

    let maximize [n][k][d] (X: data) (_: theta_sized [k][d]) (log_posts: posteriors_sized [n][]): theta = 
        let log_posts = log_posts :> [][k]val_t
        
        -- need to normalize the log probabilities:
        let log_prob_sums = map logsumexp log_posts
        let normalized_log_probs = map2 (\prob logprobsum ->
                                        map (\x -> x V.- logprobsum) prob
                                    ) log_posts log_prob_sums
           
        -- make normal probs from log probs
        let posts = map (\row ->
                            map (V.exp) row
                        ) normalized_log_probs 
 
        let posterior_sum = map (\col -> 
                            (V.sum col) V.+ (V.f32 0.00000001) -- we might end up dividing by 0 otherwise?
                        ) (transpose posts)

        let new_alphas = map (\alpha -> V.(alpha / (V.i64 n))) posterior_sum

        let new_mus = map2 (\i post_sum ->
                            -- dot product expressed as semiring, with scaling by posterior probability
                            let mu_unscaled = dataset.map_index_reduce X d
                                                (\v row _ -> V.(v * posts[(I.to_i64 row), i]))
                                                (V.+) (V.f32 0)
                                in map (\x -> x V./ post_sum) mu_unscaled
                        ) (iota k) posterior_sum
        

        let new_mu_squared_scaled = map2 (\mus post ->
                                        map (\m -> V.(m * m * post)) mus
                                    ) new_mus posterior_sum

        let new_sigmas = map3 (\i new_mu_squareds post_sum ->
                        let data_sum = dataset.map_index_reduce X d
                                        (\v row col -> 
                                            let post_fac = posts[(I.to_i64 row), i]
                                            in V.((v - (V.f32 2) * new_mus[i, I.to_i64 col]) * v * post_fac) -- this is 0 if v is 0
                                        ) (V.+) (V.f32 0)
                        -- add missing mu*mu*post_facz
                        in map2 (\a b -> V.(((a + b) / post_sum) + (V.f32 0.000001))) data_sum new_mu_squareds
                    ) (iota k) new_mu_squared_scaled posterior_sum
        in {alphas = new_alphas, mus = new_mus, sigmas = new_sigmas}


    let empty_theta (k:i64) (d: i64): theta =
                                    let one_by_k = ((V.f32 1f32) V./ (V.i64 k))
                                    -- let one_by_d = ((V.f32 1f32) V./ (V.i64 d))
                                    in  {
                                            alphas = replicate k one_by_k,
                                            mus = map (\_ -> replicate d (V.f32 0.000000001)) (iota k),
                                            sigmas = map (\_ -> replicate d (V.f32 1)) (iota k)       -- or 1/d? 
                                        }
                            
    let init_theta (X: data) (k: i64): theta =
                        let engine = rng_engine.rng_from_seed [seed]
                        let n = dataset.get_n X
                        let d = dataset.get_d X
                        let splits = rng_engine.split_rng (n*k) engine
                        let random_probs = map (\s ->
                                let (_, v) = rand_val.rand (V.f32 0.000000001, V.f32 1) s -- we don't need the new state anymore
                                in v
                          ) splits
                        let random_probs = unflatten n k random_probs
                        -- need to sum up to 1 in each row
                        let random_probs = map (\row ->
                                let row_sum = V.sum row
                                in map (\x -> x V./ row_sum) row 
                                ) random_probs
                        -- need log probs for the algo
                        let random_log_probs = map (\row ->
                                        map V.log row 
                                    ) random_probs
                        let theta = empty_theta k d

                        in maximize X theta random_log_probs

    let compute_log_prob_sum (posts: posteriors): f64 =
                        -- compute logsumexp
                        let log_probs = map (\x -> V.to_f64 (logsumexp x)) posts
                        in f64.sum log_probs

    let compute_objective_function = compute_log_prob_sum

    let check_converged (old_posts: posteriors) (new_posts: posteriors) (conv_limit: conv_limit_t): bool = 
                        let old_log_prob_sum = compute_log_prob_sum old_posts
                        let new_log_prob_sum = compute_log_prob_sum new_posts
                        
                        let diff = f64.abs (old_log_prob_sum - new_log_prob_sum) 
                        in diff <= (f64.f32 conv_limit)

    let get_means (params: theta): [][]f64 =
                        map (\row -> 
                                map V.to_f64 row
                            ) params.mus
    let get_sigma (params: theta): [][]f64 =
                        map (\row -> 
                                map V.to_f64 row
                            ) params.sigmas
    let get_weights (params: theta): []f64 =
                    map V.to_f64 params.alphas
                           
}