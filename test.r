library(januts)

xmean <- c(0, 1.5)
xSig <- diag(c(1, 1.2))
xSig[1,2] <- xSig[2,1] <- -.5
samples <- 1e5


hmc_sampled <- rmvn_hmc(xmean, xSig, samples, .8, T) 
nuts_sampled <- rmvn_nuts(xmean, xSig, samples, .8, T)
mvn_sampled <- t(matrix(rnorm(samples*2), ncol=2) %*% chol(xSig)) + (matrix(1, ncol=samples) %x% xmean)
  
par(mfrow=c(3,2), mar=rep(2,4))
hist(hmc_sampled[1,], breaks=200, main="HMC, 1")
hist(hmc_sampled[2,], breaks=200, main="HMC, 2")
hist(nuts_sampled[1,], breaks=200, main="NUTS, 1")
hist(nuts_sampled[2,], breaks=200, main="NUTS, 2")
hist(mvn_sampled[1,], breaks=200, main="MVN, 1")
hist(mvn_sampled[2,], breaks=200, main="MVN, 2")

