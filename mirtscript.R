library(mirt)
args = commandArgs(trailingOnly=TRUE)
N = as.numeric(args[1])
ndim=as.numeric(args[4])
Nit=28
sparsity = as.numeric(args[2])
theta1 <- as.matrix(read.csv(paste0('./MIRT-VAE-Qmatrix/parameters/simulated/theta_', iteration, '.csv'), header=F))
d1 = as.matrix(read.csv(paste0('./MIRT-VAE-Qmatrix/parameters/simulated/b_', iteration, '.csv'), header=F))
a1 = as.matrix(read.csv(paste0('./MIRT-VAE-Qmatrix/parameters/simulated/a_', iteration, '.csv'), header=F))[,1:ndim]
data1 = as.matrix(read.csv(paste0('./MIRT-VAE-Qmatrix/data/simulated/data_', iteration, '.csv'), header=F))

if (ndim>1){
	Q = read.csv('./MIRT-VAE-Qmatrix/parameters/QMatrix.csv', header=F)[,1:ndim]
}
#a1 = a1*as.matrix(Q)


#print('simulating data...')
#data1 = simdata(a1, d1, itemtype = '2PL', Theta = theta1)
if (sparsity>0){
	data1[sample(length(data1), length(data1)*sparsity, replace = FALSE)] <- NA
}

# initialize paramters
#print('initializing pars...')
#pars <- mirt(data1,ndim, pars = 'values', technical = list())
#pars[pars$name=='a3' & pars$item == 'Item_28', ]$value = 1
#pars[pars$name=='a3' & pars$item == 'Item_28', ]$est = TRUE
#if (ndim > 1){
#	for (dim in 1:3){
# 		for (item in 1:28){
#    			if (Q[item, dim] == 0){
#     				pars[pars$name == paste0('a', dim) & pars$item == paste0('Item_', item), ]$value = 0
#      				pars[pars$name == paste0('a', dim) & pars$item == paste0('Item_', item), ]$est = FALSE
#      }
#    }
#  }
#}
if (ndim == 1){
  model = ndim
}else {
  model='
    f1=1,3,7,10,11,12,13,14,16,20,21,25,27,28
    f2=1,2,8,17,23,24
    f3=3,4,5,6,7,9,11,12,15,16,17,18,19,20,21,22,26,28'
}

start = Sys.time()
fit =mirt(data1, 
          model, 
          method = 'EM', 
          randompars = T,
	  technical=list(NCYCLES=1000))
runtime = as.numeric(difftime(Sys.time(), start, units='secs'))
itempars <- coef(fit, simplify = TRUE)$items
a <- itempars[,1:(ncol(itempars)-3)]
d <- itempars[, ncol(itempars)-2]
theta <- fscores(fit)


if (ndim>1){
  for (i in 1:ndim){
    if (cor(a[,i], a1[,i])<0){
      a[,i] = a[,i] * -1 
      theta[,i] = theta[,i] * -1 
    }
  }
}

mse <- function(a,b){
  mean((a-b)^2)
}
theta[is.na(theta)]=0
#for (i in 1:ndim){
#  plot(a1[,i], a[,i], main= paste('Dimension ', i, 'MSE: ', round(mse(a1[,i], a[,i]),4)))
#  plot(theta1[,i], theta[,i], main= paste('Dimension ', i, 'MSE: ', round(mse(theta1[,i], theta[,i]),4)))
#}

#plot(d1[,1], d, main= paste('Dimension ', i, 'MSE: ', round(mse(d, d1[,1]),4)))

msea=mse(a1, a)
msed=mse(d1, d)
mset=mse(theta1, theta)
lll = logLik(fit)

print(msea)

fileConn<-file(paste0('./results/', paste(args, collapse='_')))
writeLines(c(as.character(msea), as.character(msed), as.character(mset), as.character(lll), as.character(runtime)), fileConn)
close(fileConn)



