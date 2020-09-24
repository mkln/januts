#include "utils.h"

#ifndef HMCH
#define HMCH

using namespace std;

template <class T>
inline arma::vec sample_one_hmc_cpp(arma::vec current_q, 
                                    T& postparams,
                                    AdaptE& adaptparams){
  
  int K = current_q.n_elem;
  
  arma::vec p = arma::randn(K);  
  arma::vec q = current_q;
  
  double joint0 = loglike_cpp(q, postparams); - 0.5* arma::conv_to<double>::from(p.t() * p);
  double epsilondirection = adaptparams.eps * 2 * (R::runif(0, 1) < 0.5) - 1;
  p += epsilondirection * 0.5 * grad_loglike_cpp(q, postparams);
  q += epsilondirection * p;
  p += epsilondirection * 0.5 * grad_loglike_cpp(q, postparams);
  double joint = loglike_cpp(q, postparams) - 0.5 * arma::conv_to<double>::from(p.t() * p);
  
  adaptparams.alpha = std::min(1.0, exp(joint - joint0));
  adaptparams.n_alpha = 1.0;
  
  if(R::runif(0, 1) < adaptparams.alpha){ 
    current_q = q; // Accept proposal
  }
  adaptparams.adapt_step();
  
  return current_q;
}

#endif