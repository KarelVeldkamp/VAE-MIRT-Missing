library(mirt)
args = commandArgs(trailingOnly=TRUE)
N = as.numeric(args[1])
sparsity = as.numeric(args[2])
iteration=as.numeric(args[3])
model = args[4]
ndim= as.numeric(args[5])
it =1
theta1 <- as.matrix(read.csv(paste0('./MIRT-VAE-Qmatrix/parameters/simulated/theta_',ndim, '_',  it, '.csv'), header=F))
d1 = as.matrix(read.csv(paste0('./MIRT-VAE-Qmatrix/parameters/simulated/b_', ndim, '_', it, '.csv'), header=F))
a1 = as.matrix(read.csv(paste0('./MIRT-VAE-Qmatrix/parameters/simulated/a_', ndim, '_', it, '.csv'), header=F))[,1:ndim]
#data1 = as.matrix(read.csv(paste0('./MIRT-VAE-Qmatrix/data/simulated/data_', ndim, '_', iteration, '.csv'), header=F))


mse <- function(a,b){
  mean((a-b)^2)
}



if (ndim>1){
	Q = read.csv(paste0('./MIRT-VAE-Qmatrix/parameters/QMatrix', ndim, 'D.csv'), header=F)[,1:ndim]
}
#a1 = a1*as.matrix(Q)
Nit = nrow(Q)

#print('simulating data...')
data1 = simdata(a1, d1, itemtype = '2PL', Theta = theta1)
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
  method = 'EM'
}else if (ndim==3){
  model='
    f1=1,3,7,10,11,12,13,14,16,20,21,25,27,28
    f2=1,2,8,17,23,24
    f3=3,4,5,6,7,9,11,12,15,16,17,18,19,20,21,22,26,28'
  method = 'MHRM'
}else if (ndim==10){
  model='f1 = 1-20,
         f2 = 11-30
         f3 = 21-40
         f4 = 31-50,
         f5 = 41-60
         f6 = 51-70
         f7 = 61-80,
         f8 = 71-90
         f9 = 81-100
         f10 = 91-110'
  method='QMCEM'
} else {
  stop(paste0('q matrix not implementented for', 0, 'dimensions'))
}

start = Sys.time()
fit =mirt(data1, 
          model, 
          method = method, 
          randompars = T,
	  technical=list())
runtime = as.numeric(difftime(Sys.time(), start, units='secs'))
itempars <- coef(fit, simplify = TRUE)$items
a <- itempars[,1:(ncol(itempars)-3)]
d <- itempars[, ncol(itempars)-2]

# estimate theta
QMC = ifelse(ndim>3, T, F) # use quasi monte carlo for high dimensional models
theta <- fscores(fit, QMC = QMC)
total_runtime = as.numeric(difftime(Sys.time(), start, units='secs'))
as.numeric(difftime(Sys.time(), start, units='secs'))

if (ndim>1){
  for (i in 1:ndim){
    if (cor(a[,i], a1[,i])<0){
      a[,i] = a[,i] * -1 
      theta[,i] = theta[,i] * -1 
    }
  }
}




theta[is.na(theta)]=0
#for (i in 1:ndim){
#  plot(a1[,i], a[,i], main= paste('Dimension ', i, 'MSE: ', round(mse(a1[,i], a[,i]),4)))
#  plot(theta1[,i], theta[,i], main= paste('Dimension ', i, 'MSE: ', round(mse(theta1[,i], theta[,i]),4)))
#}

#plot(d1[,1], d, main= paste('Dimension ', i, 'MSE: ', round(mse(d, d1[,1]),4)))

print(mse(a1, a))
# biasa=mean(a1-a)
# vara=var(a1)
# msed=mse(d1, d)
# biasd=mean(d1-d)
# vard=var(d1)
# mset=mse(theta1, theta)
# biast=mean(theta1-theta)
# vart=var(theta1)
# lll = logLik(fit)
# 
# print(msea)
# 
# fileConn<-file(paste0('./results/', paste(args, collapse='_')))
# writeLines(c(as.character(msea), as.character(msed), as.character(mset), 
#              as.character(lll), as.character(runtime), 
#              as.character(biasa), as.character(biasd), as.character(biast), 
#              as.character(vara), as.character(vard), as.character(vart)),
#            fileConn)
# close(fileConn)

par = c()
value = c()
par_i = c()
par_j = c()
estimates = list(theta, a, as.matrix(d))
par_names = c('theta', 'a', 'd')
for (i in 1:3){
  est = estimates[[i]]
  print(i)
  for (r in 1:nrow(est)){
    for (c in 1:ncol(est)){
      par = c(par, par_names[i])
      par_i = c(par_i, r)
      par_j = c(par_j, c)
      value = c(value, est[r, c])
    }
  }
}

results = data.frame('n' = N,
                     'missing' = sparsity,
                     'iteration'=iteration,
                     'model' = 'mirt',
                     'mirt_dim'= ndim,
                     'parameter'=par,
                     'i'=par_i,
                     'j'=par_j,
                     'value'=value)

#write.csv(results, file = paste0("./results/", paste(args, collapse='_'), ".csv"),row.names=F)
write.table(total_runtime, paste0("./results/", paste(args, collapse='_'), ".csv"), col.names = F, row.names = F, quote = F, sep = "\n")
