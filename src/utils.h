#ifndef HMCUTILS
#define HMCUTILS

#include "RcppArmadillo.h"


class DistParams {
public:
  int n;
  arma::vec m;
  arma::mat Si;
  
  arma::mat M;
  arma::mat Michol;
  
  DistParams();
  DistParams(const arma::vec& m_in, const arma::mat& S_in);
};

inline DistParams::DistParams(){
  n=-1;
}

inline DistParams::DistParams(const arma::vec& m_in, const arma::mat& Si_in){
  m = m_in;
  Si = Si_in;
  n = m.n_elem;
  
  M = arma::eye(n, n);
  Michol = M;
}

// log posterior 
inline double loglike_cpp(const arma::vec& x, const DistParams& postparams){
  return -.5 * arma::conv_to<double>::from((x - postparams.m).t() * postparams.Si * (x - postparams.m));
}

// Gradient of the log posterior
inline arma::vec grad_loglike_cpp(const arma::vec& x, const DistParams& postparams){
  return -postparams.Si * (x-postparams.m);
}

class AdaptE {
public:
  int i;
  
  bool active;
  
  double mu;
  double eps;
  double eps_bar;
  double H_bar;
  double gamma;
  double t0;
  double kappa;
  int M_adapt;
  double delta;
  
  double alpha;
  double n_alpha;
  
  AdaptE();
  AdaptE(double, int);
  void adapt_step();
};

inline AdaptE::AdaptE(){
  
}

inline AdaptE::AdaptE(double eps0, int M_adapt_in=0){
  i = 0;
  mu = log(10 * eps0);
  eps = eps0;
  eps_bar = M_adapt_in == 0? eps0 : 1;
  H_bar = 0;
  gamma = .05;
  t0 = 10;
  kappa = .75;
  delta = 0.75;
  M_adapt = M_adapt_in;
  
  alpha = 0;
  n_alpha = 0;
  active = true;
}

inline void AdaptE::adapt_step(){
  if(active){
    int m = i+1;
    if(m < M_adapt){
      H_bar = (1 - 1.0/(m + t0)) * H_bar + 1.0/(m + t0) * (delta - alpha/n_alpha);
      eps = exp(mu - sqrt(m)/gamma * H_bar);
      eps_bar = exp(pow(m, -kappa) * log(eps) + (1-pow(m, -kappa)) * log(eps_bar));
    } else {
      eps = eps_bar;
    }
  }
}


// Position q and momentum p
struct pq_point {
  arma::vec q;
  arma::vec p;
  
  explicit pq_point(int n): q(n), p(n) {}
  pq_point(const pq_point& z): q(z.q.size()), p(z.p.size()) {
    q = z.q;
    p = z.p;
  }
  
  pq_point& operator= (const pq_point& z) {
    if (this == &z)
      return *this;
    
    q = z.q;
    p = z.p;
    
    return *this;
  }
};

struct nuts_util {
  // Constants through each recursion
  double log_u; // uniform sample
  double H0; 	// Hamiltonian of starting point?
  int sign; 	// direction of the tree in a given iteration/recursion
  
  // Aggregators through each recursion
  int n_tree;
  double sum_prob; 
  bool criterion;
  
  // just to guarantee bool initializes to valid value
  nuts_util() : criterion(false) { }
};

inline bool compute_criterion(arma::vec& p_sharp_minus, 
                              arma::vec& p_sharp_plus,
                              arma::vec& rho) {
  double crit1 = arma::conv_to<double>::from(p_sharp_plus.t() * rho);
  double crit2 = arma::conv_to<double>::from(p_sharp_minus.t() * rho);
  return crit1 > 0 && crit2 > 0;
}


template <class T>
inline void leapfrog(pq_point &z, float epsilon, T& postparams){
  z.p += epsilon * 0.5 * grad_loglike_cpp(z.q, postparams);
  z.q += epsilon * postparams.M * z.p;
  z.p += epsilon * 0.5 * grad_loglike_cpp(z.q, postparams);
}

template <class T>
inline double find_reasonable_stepsize(const arma::vec& current_q, T& postparams){
  int K = current_q.n_elem;
  pq_point z(K);
  arma::vec p0 = postparams.Michol * arma::randn(K);
  
  double epsilon = 1;
  z.q = current_q;
  z.p = p0;
  
  double p_orig = loglike_cpp(z.q, postparams) - 0.5* arma::conv_to<double>::from(z.p.t() * postparams.M * z.p);//sum(z.p % z.p); 
  
  leapfrog(z, epsilon, postparams);
  double p_prop = loglike_cpp(z.q, postparams) - 0.5* arma::conv_to<double>::from(z.p.t() * postparams.M * z.p);//sum(z.p % z.p); 
  
  double p_ratio = exp(p_prop - p_orig);
  double a = 2 * (p_ratio > .5) - 1;
  int it=0;
  bool condition = (pow(p_ratio, a) > pow(2.0, -a)) || std::isnan(p_ratio);
  
  while( condition & (it < 50) ){
    it ++;
    epsilon = pow(2.0, a) * epsilon;
    
    leapfrog(z, epsilon, postparams);
    p_prop = loglike_cpp(z.q, postparams) - 0.5* arma::conv_to<double>::from(z.p.t() * postparams.M * z.p);//sum(z.p % z.p); 
    p_ratio = exp(p_prop - p_orig);
    
    condition = (pow(p_ratio, a) > pow(2.0, -a)) || std::isnan(p_ratio);
    //Rcpp::Rcout << "epsilon : " << epsilon << " p_ratio " << p_ratio << " " << p_prop << "," << p_orig << " .. " << pow(p_ratio, a) << "\n";
    // reset
    z.q = current_q;
    z.p = p0;
  }
  if(it == 50){
    epsilon = .01;
    Rcpp::Rcout << "Set epsilon to " << epsilon << " after no reasonable stepsize could be found. (?)\n";
  }
  return epsilon/2.0;
} 


#endif