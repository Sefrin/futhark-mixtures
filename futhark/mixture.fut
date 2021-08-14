import "mixture_dataset"

module type mixture = {
    module C : numeric
    type~ theta
    type~ posteriors
    type~ data
    type conv_limit_t = C.t

    val eval: data -> theta -> posteriors
    val maximize: data -> theta -> posteriors -> theta
    val init_theta: data -> (k: i64) -> theta
    val check_converged: posteriors -> posteriors -> conv_limit_t-> bool
    val compute_objective_function: posteriors -> f64
    val get_means: theta -> [][]f64
    val get_weights: theta -> []f64
}