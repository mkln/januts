#include "utils.h"

#ifndef NUTSH
#define NUTSH

using namespace std;


template <class T>
inline int BuildTree(pq_point& z, pq_point& z_propose, 
                     arma::vec& p_sharp_left, 
                     arma::vec& p_sharp_right, 
                     arma::vec& rho, 
                     nuts_util& util, 
                     int depth, float epsilon,
                     T& postparams,
                     double& alpha,
                     double& n_alpha,
                     double joint_zero){
  
  //Rcpp::Rcout << "\n Tree direction:" << util.sign << " Depth:" << depth << std::endl;
  
  int K = z.q.n_rows;
  
  //std::default_random_engine generator;
  //std::uniform_real_distribution<double> unif01(0.0,1.0);
  //int F = postparams.W.n_rows;  
  float delta_max = 1000; // Recommended in the NUTS paper: 1000
  
  // Base case - take a single leapfrog step in the direction v
  if(depth == 0){
    leapfrog(z, util.sign * epsilon, postparams);
    
    float joint = loglike_cpp(z.q, postparams) - 0.5 * arma::conv_to<double>::from(z.p.t() * z.p);//sum(z.p % z.p); 
    
    int valid_subtree = (util.log_u <= joint);    // Is the new point in the slice?
    util.criterion = util.log_u - joint < delta_max; // Is the simulation wildly inaccurate? // TODO: review
    util.n_tree += 1;
    
    //Rcpp::Rcout << "joint: " << joint << " joint_zero: " << joint_zero << "\n";
    alpha = std::min(1.0, exp( joint - joint_zero ));
    n_alpha = 1;
    
    z_propose = z;
    rho += z.p;
    p_sharp_left = z.p;  // p_sharp = inv(M)*p (Betancourt 58)
    p_sharp_right = p_sharp_left;
    
    return valid_subtree;
  } 
  
  // General recursion
  arma::vec p_sharp_dummy(K);
  
  // Build the left subtree
  arma::vec rho_left = arma::zeros(K);
  double alpha_prime1=0;
  double n_alpha_prime1=0;
  int n1 = BuildTree(z, z_propose, p_sharp_left, p_sharp_dummy, 
                     rho_left, util, depth-1, epsilon, postparams, 
                     alpha_prime1, n_alpha_prime1, joint_zero);
  
  if (!util.criterion) return 0; // early stopping
  
  // Build the right subtree
  pq_point z_propose_right(z);
  arma::vec rho_right(K); rho_left.zeros();
  double alpha_prime2=0;
  double n_alpha_prime2=0;
  int n2 = BuildTree(z, z_propose_right, p_sharp_dummy, p_sharp_right, 
                     rho_right, util, depth-1, epsilon, postparams, 
                     alpha_prime2, n_alpha_prime2, joint_zero);
  
  // Choose which subtree to propagate a sample up from.
  double accept_prob = static_cast<double>(n2) / std::max((n1 + n2), 1); // avoids 0/0;
  //Rcpp::RNGScope scope;
  float rand01 = R::runif(0, 1);//unif01(generator);
  if(util.criterion && (rand01 < accept_prob)){
    z_propose = z_propose_right;
  }
  
  // Break when NUTS criterion is no longer satisfied
  arma::vec rho_subtree = rho_left + rho_right;
  rho += rho_subtree;
  util.criterion = compute_criterion(p_sharp_left, p_sharp_right, rho);
  
  int n_valid_subtree = n1 + n2;
  
  alpha = alpha_prime1 + alpha_prime2;
  n_alpha = n_alpha_prime1 + n_alpha_prime2;
  
  
  return(n_valid_subtree);
}

template <class T>
inline arma::vec sample_one_nuts_cpp(arma::vec current_q, 
                                     T& postparams,
                                     AdaptE& adaptparams){
  
  
  
  int K = current_q.n_elem;
  //int F = W.n_rows;
  int MAXDEPTH = 7;
  //int iter = 3;
  
  //arma::mat h_n_samples(K, iter);   // traces of p
  arma::vec p0 = arma::randn(K);                  // initial momentum
  //current_q = log(current_q); 		// Transform to unrestricted space
  //h_n_samples.col(1) = current_q;
  
  pq_point z(K);
  
  nuts_util util;
  
  // Initialize the path. Proposed sample,
  // and leftmost/rightmost position and momentum
  ////////////////////////
  z.q = current_q;
  z.p = p0;
  pq_point z_plus(z);
  pq_point z_minus(z);
  pq_point z_propose(z);
  
  // Utils o compute NUTS stop criterion
  arma::vec p_sharp_plus = z.p;
  arma::vec p_sharp_dummy = p_sharp_plus;
  arma::vec p_sharp_minus = p_sharp_plus;
  arma::vec rho(z.p);
  
  // Hamiltonian
  // Joint logprobability of position q and momentum p
  //Rcpp::Rcout << "sample_one_nuts_cpp: \n";
  double current_logpost = loglike_cpp(current_q, postparams);
  ///Rcpp::Rcout << "starting from: " << current_logpost << "\n";
  double joint = current_logpost - 0.5* arma::conv_to<double>::from(z.p.t() * z.p);//sum(z.p % z.p); 
  
  // Slice variable
  ///////////////////////
  // Sample the slice variable: u ~ uniform([0, exp(joint)]). 
  // Equivalent to: (log(u) - joint) ~ exponential(1).
  // logu = joint - exprnd(1);
  //Rcpp::RNGScope scope;
  float random = R::rexp(1); 
  util.log_u = joint - random;
  
  int n_valid = 1;
  util.criterion = true;
  
  // Build a trajectory until the NUTS criterion is no longer satisfied
  int depth_ = 0;
  int divergent_ = 0;
  util.n_tree = 0;
  util.sum_prob = 0;
  
  
  // Build a balanced binary tree until the NUTS criterion fails
  while(util.criterion && (depth_ < MAXDEPTH)){
    // Build a new subtree in the chosen direction
    // (Modifies z_propose, z_minus, z_plus)
    arma::vec rho_subtree = arma::zeros(K);
    
    // Build a new subtree in a random direction
    //Rcpp::RNGScope scope;
    util.sign = 2 * (R::runif(0, 1) < 0.5) - 1;
    int n_valid_subtree=0;
    if(util.sign == 1){    
      z.pq_point::operator=(z_minus);
      n_valid_subtree = BuildTree(z, z_propose, p_sharp_dummy, p_sharp_plus, rho_subtree, 
                                  util, depth_, adaptparams.eps, postparams, 
                                  adaptparams.alpha, adaptparams.n_alpha, joint);
      z_plus.pq_point::operator=(z);
    } else {  
      z.pq_point::operator=(z_plus);
      n_valid_subtree = BuildTree(z, z_propose, p_sharp_dummy, p_sharp_minus, rho_subtree, 
                                  util, depth_, adaptparams.eps, postparams, 
                                  adaptparams.alpha, adaptparams.n_alpha, joint);
      z_minus.pq_point::operator=(z);
    }
    ++depth_;  // Increment depth.
    
    if(util.criterion){ 
      // Use Metropolis-Hastings to decide whether or not to move to a
      // point from the half-tree we just generated.
      double subtree_prob = std::min(1.0, static_cast<double>(n_valid_subtree)/n_valid);
      //Rcpp::RNGScope scope;
      if(R::runif(0, 1) < subtree_prob){ 
        current_q = z_propose.q; // Accept proposal
      }
    }
    
    // Update number of valid points we've seen.
    n_valid += n_valid_subtree;
    
    // Break when NUTS criterion is no longer satisfied
    rho += rho_subtree;
    util.criterion = util.criterion && compute_criterion(p_sharp_minus, p_sharp_plus, rho);
  } // end while
  
  //Rcpp::Rcout << "eps: " << adaptparams.eps << "\n";
  //Rcpp::Rcout << "DEPTH: " << depth_ << "\n";
  
  // adapting
  adaptparams.adapt_step();
  
  //current_logpost = loglike_cpp(current_q, postparams);
  //Rcpp::Rcout << "ending with: " << current_logpost << "\n";
  
  return current_q;
}

#endif