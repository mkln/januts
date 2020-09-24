#include <RcppArmadillo.h>
using namespace std;

double f(const arma::vec& x){
  return -.5*arma::conv_to<double>::from(x.t()*x);
}


arma::vec grad_f(const arma::vec& x){
  return -x;
}

Rcpp::List leapfrog_step_rlist(const arma::vec& theta, 
                         const arma::vec& r, 
                         double eps){
  arma::vec r_tilde = r + 0.5 * eps * grad_f(theta);
  arma::vec theta_tilde = theta + eps * r_tilde;
  r_tilde += 0.5 * eps * grad_f(theta_tilde);
  return Rcpp::List::create(
    Rcpp::Named("theta") = theta_tilde,
    Rcpp::Named("r") = r_tilde
  );
}

void leapfrog_step(arma::vec& theta, 
                         arma::vec& r, 
                         const double& eps){
  r += 0.5 * eps * grad_f(theta);
  theta += eps * r;
  r += 0.5 * eps * grad_f(theta);
}

bool check_NUTS(int s, const arma::vec& theta_plus, const arma::vec& theta_minus, const arma::vec& r_plus, const arma::vec& r_minus){
  arma::vec dtheta = theta_plus - theta_minus;
  bool condition1 = arma::conv_to<double>::from(dtheta.t() * r_minus) >= 0;
  bool condition2 =  arma::conv_to<double>::from(dtheta.t() * r_plus) >= 0;
  return s && condition1 && condition2;
}

Rcpp::List build_tree_rlist(arma::vec theta, arma::vec r, double u, int v, int j, double eps, arma::vec theta0, arma::vec r0, int Delta_max = 1000){
    if(j == 0){
      Rcpp::List proposed = leapfrog_step_rlist(theta, r, v*eps);
      theta = Rcpp::as<arma::vec>(proposed["theta"]);
      r = Rcpp::as<arma::vec>(proposed["r"]);
      
      double log_prob = f(theta) - .5* arma::conv_to<double>::from(r.t() * r);
      double log_prob0 = f(theta0) - .5* arma::conv_to<double>::from(r0.t() * r0);
      int n = (log(u) <= log_prob);
      int s = (log(u) < Delta_max + log_prob);
      double alpha = std::min(1.0, exp(log_prob - log_prob0));
      
      return Rcpp::List::create(
        Rcpp::Named("theta_minus") = theta, 
        Rcpp::Named("theta_plus") = theta, 
        Rcpp::Named("theta") = theta, 
        Rcpp::Named("r_minus") = r,
        Rcpp::Named("r_plus") = r,
        Rcpp::Named("s") = s,
        Rcpp::Named("n") = n,
        Rcpp::Named("alpha") = alpha,
        Rcpp::Named("n_alpha") = 1);
      
    } else {
      int s=0;
      int n=0;
      double alpha=0; 
      double n_alpha=0;
      
      Rcpp::List obj0 = build_tree_rlist(theta, r, u, v, j-1, eps, theta0, r0);
      arma::vec theta_minus =  Rcpp::as<arma::vec>(obj0["theta_minus"]);
      arma::vec r_minus = Rcpp::as<arma::vec>(obj0["r_minus"]);
      arma::vec theta_plus = Rcpp::as<arma::vec>(obj0["theta_plus"]);
      arma::vec r_plus = Rcpp::as<arma::vec>(obj0["r_plus"]);
      arma::vec theta = Rcpp::as<arma::vec>(obj0["theta"]);
      int obj0_s = obj0["s"];
      if(obj0_s == 1){
        Rcpp::List obj1;
        if(v == -1){
          obj1 = build_tree_rlist(theta_minus, r_minus, u, v, j-1, eps, theta0, r0);
          theta_minus =  Rcpp::as<arma::vec>(obj1["theta_minus"]);
          r_minus = Rcpp::as<arma::vec>(obj1["r_minus"]);
        } else{
          obj1 = build_tree_rlist(theta_plus, r_plus, u, v, j-1, eps, theta0, r0);
          theta_plus = Rcpp::as<arma::vec>(obj1["theta_plus"]);
          r_plus = Rcpp::as<arma::vec>(obj1["r_plus"]);
        }
        int obj1_s = obj1["s"];
        int n0 = obj0["n"]; 
        int n1 = obj1["n"];
        n = n0 + n1;
        if(n != 0){
          double prob = (obj1["n"] + .0) / (n+.0);
          if(R::runif(0, 1) < prob){
            theta = Rcpp::as<arma::vec>(obj1["theta"]);
          }
        }
        s = check_NUTS(obj1_s, theta_plus, theta_minus, r_plus, r_minus);
        double alpha0 = obj0["alpha"];
        double alpha1 = obj1["alpha"];
        alpha = alpha0 + alpha1;
        double n_alpha0 = obj0["n_alpha"];
        double n_alpha1 = obj1["n_alpha"];
        n_alpha = n_alpha0 + n_alpha1;
      } else {
        n = obj0["n"];
        s = obj0["s"];
        alpha = obj0["alpha"];
        n_alpha = obj0["n_alpha"];
      }
      return Rcpp::List::create(
        Rcpp::Named("theta_minus") = theta_minus, 
        Rcpp::Named("theta_plus") = theta_plus, 
        Rcpp::Named("theta") = theta, 
        Rcpp::Named("r_minus") = r_minus,
        Rcpp::Named("r_plus") = r_plus,
        Rcpp::Named("s") = s,
        Rcpp::Named("n") = n,
        Rcpp::Named("alpha") = alpha,
        Rcpp::Named("n_alpha") = n_alpha);
    }
  }


arma::vec NUTS_one_step_rlist(arma::vec theta, int iter, int M_adapt, double delta = 0.5, 
                         int max_treedepth = 10, double eps = 1, bool verbose = true){
  
  double kappa = 0.75;
  int t0 = 10;
  double gamma = 0.05;
  double H = 0;
  double eps_bar = 1;
  double mu = log(10*eps);
  
  arma::vec r0 = arma::randn(theta.n_elem);
  
  double l0 = f(theta) - 0.5 * arma::conv_to<double>::from(r0.t() * r0);
  
  double u = R::runif(0, exp(l0));
  
  arma::vec theta_minus = theta;
  arma::vec theta_plus = theta;
  arma::vec r_minus = r0;
  arma::vec r_plus = r0;
  
  int j=0;
  int n=1;
  int s=1;

  while((s == 1) & j <= max_treedepth){
    int direction = 2 * (R::runif(0, 1) < 0.5) - 1;
    Rcpp::List temp;
      
    if(direction == -1){
      temp = build_tree_rlist(theta_minus, r_minus, u, direction, j, eps, theta, r0);
      theta_minus = Rcpp::as<arma::vec>(temp["theta_minus"]);
      r_minus = Rcpp::as<arma::vec>(temp["r_minus"]);
    } else{
      temp = build_tree_rlist(theta_plus, r_plus, u, direction, j, eps, theta, r0);
      theta_plus = Rcpp::as<arma::vec>(temp["theta_plus"]);
      r_plus = Rcpp::as<arma::vec>(temp["r_plus"]);
    }
    
    int temp_s = temp["s"];
    if(temp_s == 1){
      double prob = (temp["n"] + .0) / (n+.0);
      if(R::runif(0, 1) < prob){
        theta = Rcpp::as<arma::vec>(temp["theta"]);
      }
    }
    int temp_n = temp["n"];
    n = n + temp_n;
    s = check_NUTS(temp_s, theta_plus, theta_minus, r_plus, r_minus);
    j += 1;
  }
  return theta;
}



void build_tree(const arma::vec& theta, const arma::vec& r, 
                const double& u, const int& v, const int& j, const double& eps, 
                const double& log_prob0,
                
                arma::vec& theta_minus, arma::vec& theta_plus, 
                arma::vec& r_minus, arma::vec& r_plus,
                arma::vec& theta_out,
                int& s, int& n, double& alpha, double& n_alpha){
  
  
  //const arma::vec& theta0, const arma::vec& r0,
  int Delta_max = 1000;
  //arma::vec thetacp = theta;
  
  if(j == 0){
    theta_out = theta;
    arma::vec rcp = r;
    
    leapfrog_step(theta_out, rcp, v*eps);
    
    double log_prob = f(theta_out) - .5* arma::conv_to<double>::from(rcp.t() * rcp);
    
    n = (log(u) <= log_prob);
    s = (log(u) < Delta_max + log_prob);
    alpha = std::min(1.0, exp(log_prob - log_prob0));
    n_alpha = 1;
    
    theta_minus = theta_out;
    theta_plus = theta_out;
    r_minus = rcp;
    r_plus = rcp;
    //theta_out = thetacp;
    
  } else {
    int s=0;
    int n=0;
    double alpha=0; 
    double n_alpha=0;
    
    //arma::vec theta_out = arma::zeros(theta.n_elem);
    
    build_tree(theta, r, u, v, j-1, eps, log_prob0,
                                 theta_minus, theta_plus, r_minus, r_plus,
                                 theta_out, s, n, alpha, n_alpha);
    
    arma::vec theta_dummy = arma::zeros(theta.n_elem);
    arma::vec r_dummy = arma::zeros(theta.n_elem);

    if(s == 1){
      int s1=0;
      int n1=0;
      double alpha1=0; 
      double n_alpha1=0;
      // autoaccept
      if(v == -1){
        build_tree(theta_minus, r_minus, u, v, j-1, eps, log_prob0,
                          theta_minus, theta_dummy, r_minus, r_dummy, theta_out,
                          s1, n1, alpha1, n_alpha1);
      } else{
        build_tree(theta_plus, r_plus, u, v, j-1, eps, log_prob0,
                          theta_dummy, theta_plus, r_dummy, r_plus, theta_out,
                          s1, n1, alpha1, n_alpha1);
      }
      
      n += n1;
      if(n != 0){
        double prob = (n1 + .0) / (n+.0);
        if(R::runif(0, 1) > prob){
          // revert change 
          theta_out = theta;//Rcpp::as<arma::vec>(obj1["theta"]);
        }
      }
      s = check_NUTS(s1, theta_plus, theta_minus, r_plus, r_minus);
      alpha += alpha1;
      n_alpha += n_alpha1;
    } 
  }
}

arma::vec NUTS_one_step(arma::vec& theta, int iter, int M_adapt, double delta = 0.5, 
                              int max_treedepth = 10, double eps = 1, bool verbose = true){
  
  double kappa = 0.75;
  int t0 = 10;
  double gamma = 0.05;
  double H = 0;
  double eps_bar = 1;
  double mu = log(10*eps);
  
  int K = theta.n_elem;
  
  arma::vec r0 = arma::randn(K);
  double l0 = f(theta) - 0.5 * arma::conv_to<double>::from(r0.t() * r0);
  double u = R::runif(0, exp(l0));
  
  arma::vec theta_minus = theta;
  arma::vec theta_plus = theta;
  arma::vec r_minus = r0;
  arma::vec r_plus = r0;
  
  
  arma::vec theta_dummy = theta;
  arma::vec r_dummy = r0;
  
  arma::vec theta_out = theta;
  
  int j=0;
  int n=1;
  int ntree=0;
  int s=1;
  
  double alpha=0;
  double n_alpha=0;
  
  while((s == 1) & j <= max_treedepth){
    int direction = 2 * (R::runif(0, 1) < 0.5) - 1;
    
    //double log_prob0 = f(theta) - .5* arma::conv_to<double>::from(r0.t() * r0);
    
    if(direction == -1){
      build_tree(theta_minus, r_minus, u, direction, j, eps, l0,
                        theta_minus, theta_dummy, r_minus, r_dummy, theta_out,
                        ntree, s, alpha, n_alpha);
    } else{
      build_tree(theta_plus, r_plus, u, direction, j, eps, l0,
                        theta_dummy, theta_plus, r_dummy, r_plus, theta_out,
                        ntree, s, alpha, n_alpha);
    }
    
    if(s == 1){
      double prob = (ntree + .0) / (n+.0);
      if(R::runif(0, 1) < prob){
        theta = theta_out;//Rcpp::as<arma::vec>(temp["theta"]);
      }
    }
    
    n += ntree;
    s = check_NUTS(s, theta_plus, theta_minus, r_plus, r_minus);
    j += 1;
  }
  /*
  if(j > 1){
    Rcpp::Rcout << j << endl;
  }*/
  return theta;
}
