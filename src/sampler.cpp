#include "RcppArmadillo.h"
#include "nuts_fixmem.h"

//[[Rcpp::export]]
arma::mat mvn_nuts_sampler(const arma::vec& mean, const arma::mat& Sigma, int mcmc=100, 
                              double epsin=1, bool adapting=false, int max_depth=7){
  
  arma::mat Si = arma::inv_sympd(Sigma);

  arma::vec x = mean;
  
  DistParams xdist(mean, Si);
  AdaptE xadapt(epsin, 1000);
  xadapt.active = adapting;
  
  arma::mat xout_mcmc = arma::zeros(x.n_elem, mcmc);
  
  int maxdepth = max_depth;
  nuts_util util(maxdepth, x.n_rows);
  
  for(int m=0; m<mcmc; m++){
    util.reinit();
    x = sample_one_nuts_cpp(x, xdist, xadapt, util);
    xout_mcmc.col(m) = x;
  }
  
  Rcpp::Rcout << "Eps: " << xadapt.eps << "\n";
  
  return xout_mcmc;
}