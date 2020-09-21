#include "RcppArmadillo.h"

#include "nuts_distparams.h"
#include "nuts_adapt.h"
#include "targets.h"


// Position q and momentum p  :  column indices in z
const int pix = 0;
const int qix = 1;

void tree_indexing(
    int& index, int& child_left, int& child_right,
    const int& parent, const int& depth, const int& max_depth, const int& leftright){
  /*
   * a dyadic tree has been built storing all the nuts parameters at some stage
   * this function maps each node in the tree to a depth and position
   */
  int height = max_depth-depth;
  if(height == 0){
    index = 0;
    child_left = 1; child_right = 2;
  } else {
    int parent_baseline = pow(2, height-1) - 1;
    //int nodes_at_parent_baseline = pow(2, depth-1);
    int parent_position_at_level = parent - parent_baseline;
    //int lr01 = leftright == -1? 0 : 1;
    int node_position_at_level = parent_position_at_level*2 + (leftright==1);
    int child_position_at_level = node_position_at_level*2;
    int baseline = (parent_baseline+1)*2 - 1;
    int child_baseline = (baseline+1)*2 - 1;
    
    index = baseline + node_position_at_level;
    child_left = child_baseline + child_position_at_level;
    child_right = child_baseline + child_position_at_level + 1;
  }
}

struct nuts_util {
  // Constants through each recursion
  double log_u; // uniform sample
  //double H0; 	// Hamiltonian of starting point?
  int sign; 	// direction of the tree in a given iteration/recursion
  
  // Aggregators through each recursion
  //int n_tree;
  //double sum_prob; 
  bool criterion;
  double delta_max;
  int maxdepth;
  int K;
  int size;
  arma::uvec touched;
  
  // just to guarantee bool initializes to valid value
  arma::field<arma::vec> p_sharp_left;
  arma::field<arma::vec> p_sharp_right;
  arma::field<arma::vec> rho;
  nuts_util(){};
  nuts_util(int, int);
  
  void reinit();
};

inline nuts_util::nuts_util(int max_depth, int K_in){
  criterion = false;
  delta_max = 1000; // Recommended in the NUTS paper: 1000
  maxdepth = max_depth;
  K = K_in;
  size = pow(2, max_depth+1)-1;
  touched = arma::zeros<arma::uvec>(size);
  p_sharp_left = arma::field<arma::vec>(size);
  p_sharp_right = arma::field<arma::vec>(size);
  rho = arma::field<arma::vec>(size);
  for(int i=0; i<size; i++){
    p_sharp_left(i) = arma::zeros(K);
    p_sharp_right(i) = arma::zeros(K);
    rho(i) = arma::zeros(K);
  }
}

inline void nuts_util::reinit(){
  criterion = false;
  for(int i=0; i<size; i++){
    if(touched(i) == 1){
      p_sharp_left(i) = arma::zeros(K);
      p_sharp_right(i) = arma::zeros(K);
      rho(i) = arma::zeros(K);
    }
  }
  touched = arma::zeros<arma::uvec>(size);
}


inline bool compute_criterion(const arma::vec& p_sharp_minus, 
                              const arma::vec& p_sharp_plus,
                              const arma::vec& rho) {
  double crit1 = arma::conv_to<double>::from(p_sharp_plus.t() * rho);
  double crit2 = arma::conv_to<double>::from(p_sharp_minus.t() * rho);
  return crit1 > 0 && crit2 > 0;
}


template <class T>
inline void leapfrog(arma::mat &z, const float& epsilon, T& postparams){
  // z: column 0 = p, column 1 = q
  z.col(pix) += epsilon * 0.5 * grad_loglike_cpp(z.col(qix), postparams);
  z.col(qix) += epsilon * z.col(pix);
  z.col(pix) += epsilon * 0.5 * grad_loglike_cpp(z.col(qix), postparams);
}

template <class T>
inline double find_reasonable_stepsize(const arma::vec& current_q, T& postparams){
  int K = current_q.n_elem;
  //pq_point z(K);
  arma::mat z = arma::zeros(K, 2);
  arma::vec p0 = arma::randn(K);
  
  double epsilon = 1;
  z.col(qix) = current_q;
  z.col(pix) = p0;
  
  double p_orig = loglike_cpp(z.col(qix), postparams) - 0.5* arma::conv_to<double>::from(z.col(pix).t() * z.col(pix));//sum(z.col(pix) % z.col(pix)); 

  leapfrog(z, epsilon, postparams);
  double p_prop = loglike_cpp(z.col(qix), postparams) - 0.5* arma::conv_to<double>::from(z.col(pix).t() * z.col(pix));//sum(z.col(pix) % z.col(pix)); 

  double p_ratio = exp(p_prop - p_orig);
  double a = 2 * (p_ratio > .5) - 1;
  int it=0;
  bool condition = (pow(p_ratio, a) > pow(2.0, -a)) || std::isnan(p_ratio);
  
  while( condition & (it < 50) ){
    it ++;
    epsilon = pow(2.0, a) * epsilon;
    
    leapfrog(z, epsilon, postparams);
    p_prop = loglike_cpp(z.col(qix), postparams) - 0.5* arma::conv_to<double>::from(z.col(pix).t() * z.col(pix));//sum(z.col(pix) % z.col(pix)); 
    p_ratio = exp(p_prop - p_orig);
    
    condition = (pow(p_ratio, a) > pow(2.0, -a)) || std::isnan(p_ratio);
    //Rcpp::Rcout << "epsilon : " << epsilon << " p_ratio " << p_ratio << " " << p_prop << "," << p_orig << " .. " << pow(p_ratio, a) << "\n";
    // reset
    z.col(qix) = current_q;
    z.col(pix) = p0;
  }
  if(it == 50){
    epsilon = .01;
    Rcpp::Rcout << "Set epsilon to " << epsilon << " after no reasonable stepsize could be found. (?)\n";
  }
  return epsilon/2.0;
} 

template <class T>
inline int BuildTree(
    arma::mat& z, 
    arma::mat& z_propose, 
    nuts_util& util, 
    int parent, int depth, int leftright,
    float epsilon,
    T& postparams,
    double& alpha,
    double& n_alpha,
    const double& joint_zero){
  
  int K = z.n_rows;
  int this_index = 0;
  int child_index_left, child_index_right;
  tree_indexing(this_index, child_index_left, child_index_right,
                parent, depth, util.maxdepth, leftright);

  // Base case - take a single leapfrog step in the direction v
  if(depth == 0){
    leapfrog(z, util.sign * epsilon, postparams);
    
    float joint = loglike_cpp(z.col(qix), postparams) - 0.5 * arma::conv_to<double>::from(z.col(pix).t() * z.col(pix));//sum(z.col(pix) % z.col(pix)); 
    
    int valid_subtree = (util.log_u <= joint);    // Is the new point in the slice?
    util.criterion = util.log_u - joint < util.delta_max; // Is the simulation wildly inaccurate? // TODO: review
    //util.n_tree += 1;
    
    alpha = std::min(1.0, exp( joint - joint_zero ));
    n_alpha = 1;
    
    z_propose = z;
    util.rho(this_index) += z.col(pix);//rho += z.col(pix);
    
    util.p_sharp_left(this_index) = z.col(pix);
    util.p_sharp_right(this_index) = util.p_sharp_left(this_index);
    
    return valid_subtree;
  } 
  // Build the left subtree
  
  double alpha_prime1=0;
  double n_alpha_prime1=0;
  int n1 = BuildTree(z, z_propose, 
                     
                     util, 
                     this_index, depth-1, -1,
                     epsilon, postparams, 
                     alpha_prime1, n_alpha_prime1, joint_zero);
  
  if (!util.criterion) return 0; // early stopping
  
  // Build the right subtree
  arma::mat z_propose_right = z;
  double alpha_prime2=0;
  double n_alpha_prime2=0;
  int n2 = BuildTree(z, z_propose_right, 
                     
                     util, 
                     this_index, depth-1, 1,
                     epsilon, postparams, 
                     alpha_prime2, n_alpha_prime2, joint_zero);
  
  // Choose which subtree to propagate a sample up from.
  double accept_prob = static_cast<double>(n2) / std::max((n1 + n2), 1); // avoids 0/0;

  float rand01 = R::runif(0, 1);
  if(util.criterion && (rand01 < accept_prob)){
    z_propose = z_propose_right;
  }
  
  // Break when NUTS criterion is no longer satisfied
  util.rho(this_index) += util.rho(child_index_left) + util.rho(child_index_right);
  util.criterion = compute_criterion(util.p_sharp_left(this_index), 
                                     util.p_sharp_right(this_index), 
                                     util.rho(this_index));
  
  int n_valid_subtree = n1 + n2;
  
  alpha = alpha_prime1 + alpha_prime2;
  n_alpha = n_alpha_prime1 + n_alpha_prime2;
  
  
  return(n_valid_subtree);
}

template <class T>
inline arma::vec sample_one_nuts_cpp(arma::vec current_q, 
                              T& postparams,
                              AdaptE& adaptparams,
                              nuts_util& util){
  
  int K = current_q.n_elem;
  int MAXDEPTH = util.maxdepth;
  
  int this_index = 0;
  int child_index_left = 1;
  int child_index_right = 2;
  
  arma::mat z = arma::zeros(K, 2);
  
  // Initialize the path. Proposed sample,
  // and leftmost/rightmost position and momentum
  ////////////////////////
  z.col(qix) = current_q;
  z.col(pix) = arma::randn(K);                  // initial momentum
  
  arma::mat z_propose = arma::zeros(K, 2);
  
  double current_logpost = loglike_cpp(current_q, postparams);
  ///Rcpp::Rcout << "starting from: " << current_logpost << "\n";
  double joint = current_logpost - 0.5* arma::conv_to<double>::from(z.col(pix).t() * z.col(pix));//sum(z.col(pix) % z.col(pix)); 
  
  float random = R::rexp(1); 
  util.log_u = joint - random;
  
  int n_valid = 1;
  util.criterion = true;
  
  // Build a trajectory until the NUTS criterion is no longer satisfied
  int depth_ = 0;
  
  // Build a balanced binary tree until the NUTS criterion fails
  while(util.criterion && (depth_ < MAXDEPTH)){
    
    // Build a new subtree in a random direction
    util.sign = 2 * (R::runif(0, 1) < 0.5) - 1;
    int n_valid_subtree=0;
    if(util.sign == 1){    
      n_valid_subtree = BuildTree(z, z_propose, 
                                  
                                  util, 
                                  this_index, depth_, 1,
                                  adaptparams.eps, postparams, 
                                  adaptparams.alpha, adaptparams.n_alpha, joint);
    } else {  
      n_valid_subtree = BuildTree(z, z_propose, 
                                  
                                  util, 
                                  this_index, depth_, -1,
                                  adaptparams.eps, postparams, 
                                  adaptparams.alpha, adaptparams.n_alpha, joint);
    }
    ++depth_;  // Increment depth.
    
    if(util.criterion){ 
      // Use Metropolis-Hastings to decide whether or not to move to a
      // point from the half-tree we just generated.
      double subtree_prob = std::min(1.0, static_cast<double>(n_valid_subtree)/n_valid);
      //Rcpp::RNGScope scope;
      if(R::runif(0, 1) < subtree_prob){ 
        current_q = z_propose.col(qix);  // Accept proposal
      }
    }
    
    // Update number of valid points we've seen.
    n_valid += n_valid_subtree;
    
    // Break when NUTS criterion is no longer satisfied
    //rho += rho_subtree;
    util.rho(this_index) += util.rho(child_index_left) + util.rho(child_index_right);
    util.criterion = util.criterion && compute_criterion(util.p_sharp_left(this_index), 
                                                         util.p_sharp_right(this_index), 
                                                         util.rho(this_index));
  } // end while
  
  
  // adapting
  adaptparams.adapt_step();
  
  return current_q;
}
