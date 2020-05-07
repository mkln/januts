#include <RcppArmadillo.h>
#include "R.h"
#include <numeric>

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

