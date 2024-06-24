
library(mirt)

# Load necessary libraries
library(MASS)  # for mvrnorm
library(data.table)  # for fread

data_full = read.csv('../../data/DataFiles/Algebra.csv', header=F)
data_30 = read.csv('../../data/DataFiles/Algebra30.csv', header=F)
Q = read.csv('../QMatrices/QMatrixAlgebra.csv', row.names = 1)

lines = c()
for (f in 1:ncol(Q)){
  lines = c(lines, paste0('f', f, ' = ', paste(which(Q[,f]==1), collapse=',')))
}
covline = 'COV = f1*f2*f3*f4*f5*f6*f7*f8*f9'

model = paste(c(lines, covline), collapse = ' \n')
cat(model)

# full data
fit_full =mirt(data_full, 
          model, 
          method = 'MHRM', 
          randompars = T,
          technical=list())


itempars_full <- coef(fit_full, simplify = TRUE)$items
a_est_full <- itempars_full[,1:(ncol(itempars_full)-3)]
d_est_full <- itempars_full[, ncol(itempars_full)-2]

theta_est_full <- fscores(fit_full, QMC = TRUE)
theta_est_full[is.na(theta_est_full)]=0
est_cor_full = cov2cor(summary(fit_full, verbose = FALSE)$fcor)
cor_est_full= est_cor_full[upper.tri(est_cor_full)]
write.csv(a_est_full, '~/Documents/GitHub/VAE-MIRT-Missing/data/algebra/a_est_full')
write.csv(theta_est_full, '~/Documents/GitHub/VAE-MIRT-Missing/data/algebra/theta_est_full')

# 30 percent missing
fit_30 =mirt(data_30, 
               model, 
               method = 'MHRM', 
               randompars = T,
               technical=list())


itempars_30 <- coef(fit_30, simplify = TRUE)$items
a_est_30 <- itempars_30[,1:(ncol(itempars_30)-3)]
d_est_30 <- itempars_30[, ncol(itempars_30)-2]

theta_est_30 <- fscores(fit_30, QMC = T)
theta_est_30[is.na(theta_est_30)]=0
est_cor_30 = cov2cor(summary(fit_30, verbose = FALSE)$fcor)
cor_est_30= est_cor_30[upper.tri(est_cor_30)]


a_est_cvae <- read.csv('~/Documents/GitHub/VAE-MIRT-Missing/data/algebra/a_est', header = F)
d_est_cvae <- read.csv('~/Documents/GitHub/VAE-MIRT-Missing/data/algebra/d_est', header = F)$V1
d_est_cvae <- d_est_cvae - (mean(d_est_cvae)-mean(d_est_full))
theta_est_cvae <- read.csv('~/Documents/GitHub/VAE-MIRT-Missing/data/algebra/theta_est', header = F)
est_cor_cvae <- cov2cor(as.matrix(read.csv('~/Documents/GitHub/VAE-MIRT-Missing/data/algebra/cor_est', header = F)))
cor_est_cvae = est_cor_cvae[upper.tri(est_cor_cvae)]
cov2cor(as.matrix(est_cor_cvae))

par_cors = matrix(NA, nrow=6, ncol=9)
for (i in 1:9){
  if (cor(theta_est_full[,i], theta_est_30[,i])<0){
    a_est_30[,i] = a_est_30[,i] * -1
    theta_est_30[,i] = theta_est_30[,i] * -1
  }
  if (cor(theta_est_full[,i], theta_est_cvae[,i])<0){
    a_est_cvae[,i] = a_est_cvae[,i] * -1
    theta_est_cvae[,i] = theta_est_cvae[,i] * -1
  }
  
  par_cors[1,i] <- cor(a_est_full[,i], a_est_cvae[,i])
  par_cors[2,i] <- cor(a_est_full[,i], a_est_30[,i])
  par_cors[3,i] <- cor(theta_est_full[,i], theta_est_cvae[,i])
  par_cors[4,i] <- cor(theta_est_full[,i], theta_est_30[,i])
  par_cors[5,i] <- cor(est_cor_full[,i], est_cor_cvae[,i])
  par_cors[6,i] <- cor(est_cor_full[,i], est_cor_30[,i])
}

round(par_cors,4)


MSE(d_est_full, d_est_30)
MSE(d_est_full, d_est_cvae)

MSE(cor_est_full, cor_est_30)
MSE(cor_est_full, cor_est_cvae)


plot(d_est_full, d_est_30)
plot(as.vector(d_est_full), as.vector(d_est_cvae))
abline(0,1)

plot(cor_est_full, cor_est_30)
points(cor_est_full, cor_est_cvae, col='red')

MSE(cor_est_full, cor_est_cvae)
MSE(cor_est_full, cor_est_30)
cor(cor_est_30, cor_est_cvae)
cor(cor_est_full, cor_est_cvae)
cor(cor_est_full, cor_est_30)


