# Script that simulates data from multidimensional IRT model
# theta and d paramters are sampled from a standard normal.
# a parameters are sampled from a uniform between .5 and 2

library(mirt)

NDIM=3              # number of latent dimensions
NITEM=28            # number of items
NUSER=110000        # number of users

# sample parameter values
theta1 <- matrix(rnorm(NUSER*NDIM), ncol=NDIM)
d1 <- matrix(rnorm(NITEM, 0, 1))
a1 <- matrix(runif(NITEM*NDIM,.5,2), ncol=NDIM)

# set a parameters equal to zero according to Q-Matrix
a1[c(2, 4, 5, 6, 8, 9, 15, 17, 18, 19, 22, 23, 24, 26, 28), 1] = 0
a1[c(3:7, 9:16, 18:22, 25:28), 2] = 0
a1[c(1:2, 8, 10, 13:14, 23:25, 27), 3] = 0

# simulate data
X = simdata(a1, d1, Theta=theta1, itemtype ='2PL')

# save data and parameters to file
write.csv(X, file='./data/data.csv')
write.table(a1, file="./data/a.txt", row.names=FALSE, col.names=FALSE)
write.table(theta1, file="./data/theta.txt", row.names=FALSE, col.names=FALSE)
write(d1, file='./data/d.txt')


