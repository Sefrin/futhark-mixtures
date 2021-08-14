import "mixture/mixture"

module em (distribution: mixture) = {

  let fit (threshold: distribution.conv_limit_t) (k: i32) (max_iterations: i32)
        (X: distribution.data): (distribution.theta, i32, distribution.posteriors, []f64) =
      let k = i64.i32 k

      let objective_history = replicate (i64.i32 max_iterations) (f64.f32 0)
      -- initial guess of parameters based on data and k
      let params = distribution.init_theta X k

      -- Initial assignment of posteriors.
      let posterior = distribution.eval X params
      
      let converged = false
      let i = 0
      let (posts,params,_,i,objective_history) =
        loop (posterior, params, converged, i, objective_history)
        while !converged && i < max_iterations do

          let new_params = distribution.maximize X params posterior
          
          let new_posterior = distribution.eval X new_params

          let converged = distribution.check_converged posterior new_posterior threshold

          let objective_value = distribution.compute_objective_function new_posterior
          let objective_history[i] = objective_value

          in (new_posterior, new_params, converged, i+1, objective_history)
      in (params, i, posts, (take (i64.i32 i) objective_history))
}