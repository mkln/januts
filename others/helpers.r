rleapfrog_step = function(theta, r, eps, grad_f, M_diag){
  r_tilde <- r + 0.5 * eps * grad_f(theta)
  theta_tilde <- theta + eps * r_tilde / M_diag
  r_tilde <- r_tilde + 0.5 * eps * grad_f(theta_tilde)
  list(theta = theta_tilde, r = r_tilde)
}

joint_log_density = function(theta, r){
  f(theta) - 0.5*sum(r**2)
}

find_reasonable_epsilon = function(theta, eps = 1, verbose = TRUE){
  r <- rnorm(length(theta), 0, 1)
  proposed <- leapfrog_step(theta, r, eps)#, grad_f, M_diag)
  log_ratio <- joint_log_density(proposed$theta, proposed$r) - joint_log_density(theta, r)
  alpha <- ifelse(exp(log_ratio) > 0.5, 1, -1)
  if(is.nan(alpha)) alpha <- -1
  count <- 1
  while(is.nan(log_ratio) || alpha * log_ratio > (-alpha)*log(2)){
    eps <- 2**alpha * eps
    proposed <- leapfrog_step(theta, r, eps)#, grad_f, M_diag)
    log_ratio <- joint_log_density(proposed$theta, proposed$r) - joint_log_density(theta, r)
    count <- count + 1
    if(count > 100) {
      stop("Could not find reasonable epsilon in 100 iterations!")
    }
  }
  if(verbose) message("Reasonable epsilon = ", eps, " found after ", count, " steps")
  eps
}

rcheck_NUTS = function(s, theta_plus, theta_minus, r_plus, r_minus){
  if(is.na(s)) return(0)
  condition1 <- crossprod(theta_plus - theta_minus, r_minus) >= 0
  condition2 <- crossprod(theta_plus - theta_minus, r_plus) >= 0
  s && condition1 && condition2
}



rbuild_tree = function(theta, r, u, v, j, eps, theta0, r0, Delta_max = 1000){
  if(j == 0){
    proposed <- rleapfrog_step(theta, r, v*eps, grad_f, 1)#, grad_f, M_diag)
    theta <- proposed$theta
    r <- proposed$r
    log_prob <- joint_log_density(theta, r)
    log_prob0 <- joint_log_density(theta0, r0)
    n <- (log(u) <= log_prob)
    s <- (log(u) < Delta_max + log_prob)
    alpha <- min(1, exp(log_prob - log_prob0))
    if(is.nan(alpha)) stop()
    if(is.na(s) || is.nan(s)){
      s <- 0
    }
    if(is.na(n) || is.nan(n)){
      n <- 0
    }
    return(list(theta_minus=theta, theta_plus=theta, theta=theta, r_minus=r,
                r_plus=r, s=s, n=n, alpha=alpha, n_alpha=1))
  } else{
    obj0 <- rbuild_tree(theta, r, u, v, j-1, eps, theta0, r0)
    theta_minus <- obj0$theta_minus
    r_minus <- obj0$r_minus
    theta_plus <- obj0$theta_plus
    r_plus <- obj0$r_plus
    theta <- obj0$theta
    if(obj0$s == 1){
      if(v == -1){
        obj1 <- rbuild_tree(obj0$theta_minus, obj0$r_minus, u, v, j-1, eps, theta0, r0)
        theta_minus <- obj1$theta_minus
        r_minus <- obj1$r_minus
      } else{
        obj1 <- rbuild_tree(obj0$theta_plus, obj0$r_plus, u, v, j-1, eps, theta0, r0)
        theta_plus <- obj1$theta_plus
        r_plus <- obj1$r_plus
      }
      n <- obj0$n + obj1$n
      if(n != 0){
        prob <- obj1$n / n
        if(runif(1) < prob){
          theta <- obj1$theta
        }
      }
      s <- rcheck_NUTS(obj1$s, theta_plus, theta_minus, r_plus, r_minus)
      alpha <- obj0$alpha + obj1$alpha
      n_alpha <- obj0$n_alpha + obj1$n_alpha
      
    } else{
      n <- obj0$n
      s <- obj0$s
      alpha <- obj0$alpha
      n_alpha <- obj0$n_alpha
    }
    if(is.na(s) || is.nan(s)){
      s <- 0
    }
    if(is.na(n) || is.nan(n)){
      n <- 0
    }
    return(list(theta_minus=theta_minus, theta_plus=theta_plus, theta=theta,
                r_minus=r_minus, r_plus=r_plus, s=s, n=n, alpha=alpha, n_alpha=n_alpha))
  }
}

