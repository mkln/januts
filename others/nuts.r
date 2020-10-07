#' No-U-Turn sampler (https://github.com/kasparmartens/NUTS)
#'
#' @param theta Initial value for the parameters
#' @param n_iter Number of MCMC iterations
#' @param M_adapt Parameter M_adapt in algorithm 6 in the NUTS paper
#' @param delta Target acceptance ratio, defaults to 0.5
#' @param max_treedepth Maximum depth of the binary trees constructed by NUTS
#' @param eps Starting guess for epsilon
#' @return Matrix with the trace of sampled parameters. Each mcmc iteration in rows and parameters in columns.
#' @export
NUTS <- function(theta, n_iter, M_adapt = 50, delta = 0.5, max_treedepth = 10, eps = 1, verbose = TRUE){
  theta_trace <- matrix(0, n_iter, length(theta))
  par_list <- list(M_adapt = M_adapt)
  for(iter in 1:n_iter){
    nuts <- rNUTS_one_step(theta, iter, par_list, delta = delta, max_treedepth = max_treedepth, eps = eps, verbose = verbose)
    theta <- nuts$theta
    par_list <- nuts$pars
    theta_trace[iter, ] <- theta
  }
  theta_trace
}

NUTS2 <- function(theta, n_iter, M_adapt = 50, delta = 0.5, max_treedepth = 4, eps = 1, verbose = TRUE){
  theta_trace <- matrix(0, n_iter, length(theta))
  for(iter in 1:n_iter){
    theta <- NUTS_one_step(theta, iter, M_adapt, delta = delta, max_treedepth = max_treedepth, eps = eps, verbose = verbose)
    theta_trace[iter, ] <- theta
  }
  theta_trace
}


rNUTS_one_step2 <- function(theta, iter, M_adapt, delta = 0.5, max_treedepth = 10, eps = 1, verbose = TRUE){
  #set.seed(1)
  kappa <- 0.75
  t0 <- 10
  gamma <- 0.05
  #M_adapt <- par_list$M_adapt
  
  eps <- 1
  mu <- log(10*eps)
  H <- 0
  eps_bar <- 1
  
  r0 <- rnorm(length(theta), 0, 1)
  u <- runif(1, 0, exp(f(theta) - 0.5 * sum(r0**2)))
  
  if(is.nan(u)){
    warning("NUTS: sampled slice u is NaN")
    u <- runif(1, 0, 1e5)
  }
  theta_minus <- theta
  theta_plus <- theta
  r_minus <- r0
  r_plus <- r0
  j=0
  n=1
  s=1
  if(iter > M_adapt){
    eps <- runif(1, 0.9*eps_bar, 1.1*eps_bar)
  }
  # r0 = -0.6264538
  # 
  while(s == 1){
    # choose direction {-1, 1}
    direction <- sample(c(-1, 1), 1)
    if(direction == -1){
      temp <- build_tree(theta_minus, r_minus, u, direction, j, eps, theta, r0)
      theta_minus <- temp$theta_minus
      r_minus <- temp$r_minus
    } else{
      temp <- build_tree(theta_plus, r_plus, u, direction, j, eps, theta, r0)
      theta_plus <- temp$theta_plus
      r_plus <- temp$r_plus
    }
    if(is.nan(temp$s)) temp$s <- 0
    if(temp$s == 1){
      if(runif(1) < temp$n / n){
        theta <- temp$theta
      }
    }
    n <- n + temp$n
    s <- check_NUTS(temp$s, theta_plus, theta_minus, r_plus, r_minus)
    j <- j + 1
    if(j > max_treedepth){
      warning("NUTS: Reached max tree depth")
      break
    }
  }
  
  return(theta)
}


rNUTS_one_step <- function(theta, iter, par_list, delta = 0.5, max_treedepth = 10, eps = 1, verbose = TRUE){
  #set.seed(1)
  kappa <- 0.75
  t0 <- 10
  gamma <- 0.05
  M_adapt <- par_list$M_adapt
  eps <- 1
  eps_bar <- 1
  H <- 0
  mu <- log(10*eps)
  
  if(F){
    if(iter == 1){
      eps <- 1#find_reasonable_epsilon(theta, eps = eps, verbose = verbose)
      mu <- log(10*eps)
      H <- 0
      eps_bar <- 1
    } else {
      eps <- par_list$eps
      eps_bar <- par_list$eps_bar
      H <- par_list$H
      mu <- par_list$mu
    }
  }

  
  r0 <- rnorm(length(theta), 0, 1)
  u <- runif(1, 0, exp(f(theta) - 0.5 * sum(r0**2)))
  
  if(is.nan(u)){
    warning("NUTS: sampled slice u is NaN")
    u <- runif(1, 0, 1e5)
  }
  theta_minus <- theta
  theta_plus <- theta
  r_minus <- r0
  r_plus <- r0
  j=0
  n=1
  s=1
  if(iter > M_adapt && F){
    eps <- runif(1, 0.9*eps_bar, 1.1*eps_bar)
  }
  while(s == 1){
    # choose direction {-1, 1}
    direction <- sample(c(-1, 1), 1)
    if(direction == -1){
      temp <- rbuild_tree(theta_minus, r_minus, u, direction, j, eps, theta, r0)
      theta_minus <- temp$theta_minus
      r_minus <- temp$r_minus
    } else{
      temp <- rbuild_tree(theta_plus, r_plus, u, direction, j, eps, theta, r0)
      theta_plus <- temp$theta_plus
      r_plus <- temp$r_plus
    }
    if(is.nan(temp$s)) temp$s <- 0
    if(temp$s == 1){
      if(runif(1) < temp$n / n){
        theta <- temp$theta
      }
    }
    n <- n + temp$n
    s <- rcheck_NUTS(temp$s, theta_plus, theta_minus, r_plus, r_minus)
    j <- j + 1
    if(j > max_treedepth){
      warning("NUTS: Reached max tree depth")
      break
    }
  }
  if(iter <= M_adapt && F){
    H <- (1 - 1/(iter + t0))*H + 1/(iter + t0) * (delta - temp$alpha / temp$n_alpha)
    log_eps <- mu - sqrt(iter)/gamma * H
    eps_bar <- exp(iter**(-kappa) * log_eps + (1 - iter**(-kappa)) * log(eps_bar))
    eps <- exp(log_eps)
  } else{
    eps <- eps_bar
  }
  
  return(list(theta = theta,
              pars = list(eps = eps, eps_bar = eps_bar, H = H, mu = mu, M_adapt = M_adapt)))
}
