# Just another No-U-Turn sampler

This package implements NUTS (Hoffman & Gelman, 2014) with dual averaging to adapt the leapfrog stepsize. The target is fixed as a multivariate Gaussian.  

```
devtools::install_github("mkln/januts")
library(januts)

xmean <- c(0)
xSig <- diag(1)

results <- mvn_nuts_sampler(xmean, xSig, 100000, 1, F)[1,]

hist(results, breaks=100)
```


*This package is an extension of source code at https://github.com/alumbreras/NUTS-Cpp*