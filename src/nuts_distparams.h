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
