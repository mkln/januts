# Just another No-U-Turn sampler

This package implements NUTS (Hoffman & Gelman, 2014) with dual averaging to adapt the leapfrog stepsize. The target is fixed as a multivariate Gaussian.  

2020-09-21: Some recursive steps have been removed and some algorithm data are now stored in memory in a pre-made tree (using the maximum depth available) with indicators tracking changes. New code in `nuts_fixmem.h`, old code is in `nuts.h`

```
devtools::install_github("mkln/januts")
library(januts)

xmean <- c(0)
xSig <- diag(1)

results <- mvn_nuts_sampler(xmean, xSig, 100000, 1, F)[1,]

hist(results, breaks=100)
```


*This package is an extension of source code at https://github.com/alumbreras/NUTS-Cpp*