import "mixture"
import "mixture_dataset"
import "lib/github.com/diku-dk/cpprandom/random"

module vMF_mixture (dataset: mixture_dataset): (mixture with data = dataset.data_rep) = {
    module V = dataset.V
    module I = dataset.I
    module C = f32
    type val_t = V.t
    type index_t = I.t

    type theta_sized [k][d] = {alphas: [k]val_t, mus: [k][d]val_t, kappas:[k]val_t}

    type~ theta = theta_sized [][] -- arguments -- alpha, mean and diagonal of covariance matrix
    type~ posteriors = [][]val_t -- posterior probabilities
    type~ data = dataset.data_rep
    type conv_limit_t = C.t -- ?    

    module rng_engine = minstd_rand
    module rand_val = uniform_real_distribution V rng_engine
    let seed = 1337i32

    let one = V.f32 1
    let two = V.f32 2

    let bessel_fun (k: val_t) (order: val_t) = k  -- todo impls

    let c (k: val_t) (d: val_t): val_t = (k V.** (d V./ two V.- one)) V./ (((two V.* V.pi) V.** (d V./ two)) V.* (bessel_fun k ((d V./ two) V.- one)))
    -- let c (k: val_t) (d: val_t): val_t = (k V.** (d V./ two V.- one)) V./ (((two V.* V.pi) V.** (d V./ two)) V.* (bessel_fun k ((d V./ two) V.- one)))


    let eval [k][d] (X: data) (params: theta_sized[k][d] ): posteriors =
        -- ugly trick to cheat type checker!
        let kappas = params.kappas :> [k]val_t
        let mus = params.mus :> [k][]val_t
        let d = dataset.get_d X
        let normalizations = map (\kappa -> 
                        c kappa (V.i64 d)
                    ) kappas
        --mu squared / 2 sigma

        let dot_op = \data_val mu -> data_val V.* mu 
        
        let unscaled_log_probs = dataset.apply_k_semirings X mus dot_op (V.+) (V.f32 0f32)
        -- so far we have sum of positive exponents.... 

        let scaled_probs = map (\row -> 
                                let component_probs = map4 (\probs norm_term kappa alpha ->
                                                        -- add missing mu squared sum term, negate, take exp and scale prob with alpha and denominator
                                                            alpha V.* norm_term V.* (V.exp (kappa V.* probs))
                                                        ) row normalizations kappas params.alphas
                                -- sum of probs to normalize
                                let component_prob_sum = V.sum component_probs
                                -- divide by sum
                                in map (\p -> p V./ component_prob_sum) component_probs
                            ) unscaled_log_probs
        in scaled_probs

    let maximize [k] (X: data) (_: theta_sized [k][]) (posts: posteriors): theta = 
        let n = dataset.get_n X
        let d = dataset.get_d X
        let posts = posts :> [][k]val_t
        -- let mus = params.mus :> [k][]val_t

        let posterior_sum = map (\col -> 
                            V.sum col
                        ) (transpose posts)
        let new_alphas = map (\alpha -> alpha V./ (V.i64 n)) posterior_sum



        let new_mus = map (\i ->
                            dataset.map_index_reduce X d
                            (\v row _ -> v V.* posts[(I.to_i64 row), i])
                            (V.+) (V.f32 0)
                        ) (iota k)
        let cluster_magnitudes = map (\c -> 
                            let c_squared = map (\x -> x V.* x) c
                            in V.sqrt (V.sum c_squared))
                            new_mus
        
        let r_bars = map2 (\magnitude alpha ->
                            magnitude V./ ((V.i64 n) V.* alpha)
            ) cluster_magnitudes new_alphas
        
        let new_mus = map2 (\mu magnitude ->
                            map (\x -> x V./ magnitude) mu
        ) new_mus cluster_magnitudes

        let new_kappas = map (\r -> 
                            (r V.* (V.i64 d) V.- (r V.* r V.* r) ) V./ ((V.i32 1) V.- (r V.* r))
                        ) r_bars
        in {alphas = new_alphas, mus = new_mus, kappas = new_kappas}


    let empty_theta (k:i64) (d: i64): theta =
                            {
                                alphas = replicate k (V.f32 0),
                                mus = map (\_ -> replicate d (V.f32 0)) (iota k),
                                kappas = replicate k (V.f32 0)        
                            }                    

    let init_theta (X: data) (k: i64): theta =
                        let engine = rng_engine.rng_from_seed [seed]
                        let n = dataset.get_n X
                        let d = dataset.get_d X
                        let splits = rng_engine.split_rng (n*k) engine
                        let random_probs = map (\s ->
                                let (_, v) = rand_val.rand (V.f32 0, V.f32 1) s -- we don't need the new state anymore
                                in v
                          ) splits
                        let random_probs = unflatten n k random_probs
                        -- need to sum up to 1 in each row
                        let random_probs = map (\row ->
                                let row_sum = V.sum row
                                in map (\x -> x V./ row_sum) row 
                                ) random_probs

                        let theta = empty_theta k d

                        in maximize X theta random_probs

    let check_converged (old_posts: posteriors) (new_posts: posteriors) (conv_limit: conv_limit_t): bool = 
                    let old_log_probs = map (\probs ->
                        V.log (V.sum probs)
                    ) old_posts
                    let old_log_prob_sum = V.sum old_log_probs
                    let new_log_probs = map (\probs ->
                        V.log (V.sum probs)
                    ) new_posts
                    let new_log_prob_sum = V.sum new_log_probs
                    
                    let diff = V.abs (old_log_prob_sum V.- new_log_prob_sum) 
                    in diff V.<= (V.f32 conv_limit)
}