#include "RcppArmadillo.h"
#include "nuts.h"
#include "hmc.h"

//[[Rcpp::export]]
arma::mat rmvn_nuts(const arma::vec& mean, const arma::mat& Sigma, int mcmc=100, 
                           double epsin=1, bool adapting=false, int max_depth=7){
  
  arma::mat Si = arma::inv_sympd(Sigma);
  arma::vec x = mean;
  
  DistParams xdist(mean, Si);
  AdaptE xadapt(epsin, 1000);
  xadapt.active = adapting;
  
  arma::mat xout_mcmc = arma::zeros(x.n_elem, mcmc);
  int maxdepth = max_depth;
  for(int m=0; m<mcmc; m++){
    x = sample_one_nuts_cpp(x, xdist, xadapt);
    xout_mcmc.col(m) = x;
  }
  
  Rcpp::Rcout << "Eps: " << xadapt.eps << "\n";
  
  return xout_mcmc;
}

//[[Rcpp::export]]
arma::mat rmvn_hmc(const arma::vec& mean, const arma::mat& Sigma, int mcmc=100, 
                                double epsin=1, bool adapting=false){
  
  arma::mat Si = arma::inv_sympd(Sigma);
  
  arma::vec x = mean;
  
  DistParams xdist(mean, Si);
  AdaptE xadapt(epsin, 1000);
  xadapt.active = adapting;
  
  arma::mat xout_mcmc = arma::zeros(x.n_elem, mcmc);
  for(int m=0; m<mcmc; m++){
    x = sample_one_hmc_cpp(x, xdist, xadapt);
    xout_mcmc.col(m) = x;
  }
  
  Rcpp::Rcout << "Eps: " << xadapt.eps << "\n";
  
  return xout_mcmc;
}
