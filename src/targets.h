#include "RcppArmadillo.h"

// log posterior 
inline double loglike_cpp(const arma::vec& x, const DistParams& postparams){
  return -.5 * arma::conv_to<double>::from((x - postparams.m).t() * postparams.Si * (x - postparams.m));
}

// Gradient of the log posterior
inline arma::vec grad_loglike_cpp(const arma::vec& x, const DistParams& postparams){
  return -postparams.Si * (x-postparams.m);
}
