library(mirt)
data = read.csv('~/Documents/GitHub/MIRT-VAE-QMAtrix/data/missing/data.csv', row.names=1)

pars = mirt(data, 3, pars = 'values')
pars[pars$name == 'a1' & pars$item %in% paste0('Item_', 1:10), ]$value = 0
pars[pars$name == 'a1' & pars$item %in% paste0('Item_', 1:10), ]$est = FALSE
pars[pars$name == 'a2' & pars$item %in% paste0('Item_', 11:20), ]$value = 0
pars[pars$name == 'a2' & pars$item %in% paste0('Item_', 11:20), ]$est = FALSE
pars[pars$name == 'a3' & pars$item %in% paste0('Item_', 21:30), ]$value = 0
pars[pars$name == 'a3' & pars$item %in% paste0('Item_', 21:30), ]$est = FALSE

fit = mirt(data, 3, pars=pars, lambda=c(0,0))
itempars <- coef(fit, simplify = TRUE)$items
a <- itempars[,1:(ncol(itempars)-3)]
d <- itempars[, ncol(itempars)-2]
theta <- fscores(fit)

flip = sign(colMeans(a))
a = t(apply(a, 1, function(x) x*flip))
theta = t(apply(theta, 1, function(x) x*flip))

a_true = read.csv('~/Documents/GitHub/MIRT-VAE-QMAtrix/data/missing/a.csv', row.names=1)
d_true = read.csv('~/Documents/GitHub/MIRT-VAE-QMAtrix/data/missing/d.csv', row.names=1)
theta_true = read.csv('~/Documents/GitHub/MIRT-VAE-QMAtrix/data/missing/theta.csv', row.names=1)




for (i in 1:3){
  plot(a[,i], a_true[,i], main=paste('a:', i))
  abline(c(0,1))
  plot(theta[,i], theta_true[,i], main=paste('theta:', i))
  abline(c(0,1))
}

