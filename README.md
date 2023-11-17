# VAE MIRT for missing data
this repository contains the code and data for the paper 'Handling Missing Data In Variational Item Response Theory'. It compares the effect of different VAE architectures on the estimation of MIRT parameters when a sizeable proportion of observations is missing. The models compared are a regular VAE with the missing inputs removed from the loss, A Conditional VAE with the missing pattern as extra input and a partial VAE. We also implement a novel architecture we dub the Imputation VAE, in which missing inputs are imputed with the reconstruction of the previous epoch. 

In the study, we compare these variational approaches to marginal maximum likelihood in a simualtion study and in an application to the bridge to algebra dataset. 

the [MIRTVAE](MIRTVAE/) directory contains all code for our variational models. The main.py file fits model to existing or simulated data. and the [additional scripts](MIRTVAE/additional_scripts) subdirectory contains the code used to fit MML models to simulated data, as well as the code to fit models to the BTA dataset using both mirt and CVAE. 

The [data](data/) directory contains the BTA datafiles as well as all QMatrices used in the paper. We provide both preprocessed subset of the BTA dataset, as well as the dataset with 30% of observations removed. 

the [parameter](parameters)directory contains the paramter estimates of mirt and cvae on the missing dataset, as well as the mirt estimates on the complete dataset. 
