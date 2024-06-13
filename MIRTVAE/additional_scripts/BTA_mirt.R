args = commandArgs(trailingOnly=TRUE)
iteration = as.numeric(args[1]) # number of the current iteration
missing = as.numeric(args[2])
mirt_dim = as.numeric(args[3])

# load adapted version of mirt library (rgmirt)
library(mirt)

# Load necessary libraries
library(MASS)  # for mvrnorm
library(data.table)  # for fread

# Configuration
cfg <- list(covariance = 0, 
            N = 10000,
            simulate= T)

simulate=FALSE
if (simulate){
  # Create the covariance matrix
  covMat <- matrix(cfg$covariance, nrow = mirt_dim, ncol = mirt_dim)
  diag(covMat) <- 1
  set.seed(iteration)
  # Draw values for the dimensions
  theta <- mvrnorm(n = cfg$N, mu = rep(0, mirt_dim), Sigma = covMat)
  
  # Read QMatrix
  Q <- fread(paste0('~/vae/MIRT-VAE-Qmatrix/MIRTVAE/QMatrices/QMatrix', mirt_dim, 'D.csv'), header = FALSE)
  
  Q <- as.matrix(Q)
  
  # Draw discrimination parameters from uniform distribution and apply Q matrix
  a <- matrix(runif(n = nrow(Q) * mirt_dim, min = 0.5, max = 2), nrow = nrow(Q), ncol = mirt_dim)
  a <- a * Q
  
  # Equally spaced values between -2 and 2 for the difficulty
  b <- seq(-2, 2, length.out = nrow(Q))
  
  data = simdata(a, b, Theta=theta, itemtype = '2PL')
  colnames(data) = as.character(1:ncol(data))
}else{
  covMat <- matrix(cfg$covariance, nrow = mirt_dim, ncol = mirt_dim)
  data = as.matrix(read.csv(paste0('./MIRT-VAE-Qmatrix/MIRTVAE/data/simulated/data_', mirt_dim, '_',iteration, '_', missing, '.csv'), header = F))
  a = as.matrix(read.csv(paste0('./MIRT-VAE-Qmatrix/MIRTVAE/parameters/simulated/a_', mirt_dim, '_',iteration, '_', missing,  '.csv'), header = F))
  b = as.matrix(read.csv(paste0('./MIRT-VAE-Qmatrix/MIRTVAE/parameters/simulated/b_', mirt_dim, '_',iteration, '_', missing,  '.csv'), header = F))
  theta = as.matrix(read.csv(paste0('./MIRT-VAE-Qmatrix/MIRTVAE/parameters/simulated/theta_', mirt_dim, '_',iteration, '_', missing,  '.csv'), header = F))
  Q = as.matrix(read.csv(paste0('./MIRT-VAE-Qmatrix/MIRTVAE/QMatrices/QMatrix',mirt_dim,'D.csv'), header = F))
  colnames(data) = as.character(1:ncol(data))
  
  data[sample(1:length(data), ceiling(missing*length(data)))] = NA
}


# initialize paramters
print('initializing pars...')
pars <- mirt(data,mirt_dim, pars = 'values', technical = list())

# set some loadings to zero
for (i in 1:mirt_dim){
  parameter = paste(c('a', i), collapse = '')
  pars[pars$name == parameter & pars$item %in%  which(Q[,i]==0), ]$value = 0
  pars[pars$name == parameter & pars$item %in% which(Q[,i]==0), ]$est = FALSE
}


# Initialize an empty list to store the strings
factor_loadings <- list()

# Loop through each factor (column) in Q
for (factor in 1:ncol(Q)) {
  # Find the indices of items that load on this factor
  items <- which(Q[, factor] == 1)
  # Create the string for this factor
  factor_string <- paste0("F", factor, " = ", paste(items, collapse = ","))
  # Append the string to the list
  factor_loadings[[factor]] <- factor_string
}

# Combine all factor strings into a single string
formula <- paste(factor_loadings, collapse = "\n")
formula <- paste0(formula, '\n COV=F1*F2*F3')

if (mirt_dim == 3){
  model='
    f1=1,3,7,10,11,12,13,14,16,20,21,25,27,28
    f2=1,2,8,17,23,24
    f3=3,4,5,6,7,9,11,12,15,16,17,18,19,20,21,22,26,28'
  method = 'MHRM'
}else if (mirt_dim == 10){
  model = '
    f1 = 1-20
    f2 = 11-30
    f3 = 21-40
    f4 = 31 -50
    f5 = 41-60
    f6 = 51-70
    f7 = 61-80
    f8 = 71-90
    f9 = 81-100
    f10 = 91-110'
  method = 'QMCEM'
}



# fit model
print('fitting model...')
start = Sys.time()
fit <- mirt(data, 
            model=model,
            #pars=pars, 
            itemtype = '2PL',
            method=method)
time = Sys.time()-start
print(time)



itempars = coef(fit, simplify=T)$items
a_est = itempars[,1:mirt_dim]
d_est = itempars[,mirt_dim+1]

QMC = mirt_dim>5
theta_est = fscores(fit,QMC=QMC)
print(any(is.na(theta_est)))
print(any(is.na(theta)))
est_cor = cov2cor(summary(fit, verbose = FALSE)$fcor)

MSE <- function(est, true){
  mean((est-true)^2)
}

bias <- function(est, true){
  mean(est-true)
}


for (i in 1:mirt_dim){
  if (cor(a_est[,i], a[,i]) < 0){
    a_est[,i] = a_est[,i] *-1
    theta_est[,i] = theta_est[,i] *-1
  }
}
est_cor = abs(est_cor)




mse_a = MSE(a_est, a)
bias_a = bias(a_est,a)
var_a = var(as.vector(a_est))
mse_d = MSE(d_est, b)
bias_d = bias(d_est,b)
var_d = var(d_est)
mse_theta = MSE(theta_est, theta)
bias_theta = bias(theta_est,theta)
var_theta = var(as.vector(theta_est))

true_cors = covMat[upper.tri(covMat)]
est_cors = est_cor[upper.tri(est_cor)]

mse_cor = MSE(est_cors, true_cors)

runtime = as.numeric(Sys.time()- start)

lll = logLik(fit)



# save results
fileConn<-file(paste0('~/vae/MIRT-VAE-Qmatrix/results/mirt_',iteration, '_', missing, '_', mirt_dim, '.txt'))
writeLines(c(as.character(mse_a), as.character(mse_d), as.character(mse_theta), as.character(mse_cor),
             as.character(lll), as.character(runtime), 
             as.character(bias_a), as.character(bias_d), as.character(bias_theta),
             as.character(var_a), as.character(var_d), as.character(var_theta)), fileConn)
close(fileConn)

