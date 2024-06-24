

################
#Csvs simualtion
###############

library(readr)
library(dplyr)


pars = c('theta', 'a', 'b')
ln = 3*(28+10000)+ 28 + 10*(110+10000) + 110
parameters <- vector("character", length = ln)
dims <- integer(length = ln)
miss <- numeric(length = ln)
is_vec <- integer(length = ln)
js_vec <- integer(length = ln)
values <- numeric(length = ln)

# Initialize index counter
index <- 1

for (par in pars){
  for (ndim in ndims){
    for (missing in missings){
      print(par)
      print(ndim)
      print(missing)
      file_path <- paste0('~/Downloads/true_pars/simulated/', par, '_', ndim, '_1_', missing, '.csv')
      value <- read.csv(file_path, header = FALSE)
      
      # Get dimensions of the loaded CSV
      num_rows <- nrow(value)
      num_cols <- ncol(value)
      
      for (row in 1:num_rows){
        for (col in 1:num_cols){
          parameters[index] <- par
          dims[index] <- ndim
          miss[index] <- missing
          is_vec[index] <- row
          js_vec[index] <- col
          values[index] <- value[row, col]
          index <- index + 1
        }
      }
    }
  }
}

# Trim vectors to actual filled length (if needed)
parameters <- parameters[1:index - 1]
dims <- dims[1:index - 1]
miss <- miss[1:index - 1]
is_vec <- is_vec[1:index - 1]
js_vec <- js_vec[1:index - 1]
values <- values[1:index - 1]

true = data.frame('ndim'=dims, 'missing'=miss, 'parameter'=parameters, 'i'=is_vec, 'j'=js_vec, 'true'=values) %>%
  mutate(i=i-1,
         j=j-1,
         parameter=ifelse(parameter=='b', 'd', parameter),
         missing=round(missing, 8))


write.csv(true, '~/Documents/GitHub/VAE-MIRT-Missing/parameters/true.csv')


path = 'Downloads/csv_results/results/'


models = c('cvae', 'pvae', 'ivae', 'idvae')
missing = seq(0, .75, length.out=10)
ndim = c(3,10)
IWs = c(1,5,25)

results = list()
i=0
for (model in models){
  print(model)
  for (missing in missings){
    for (ndim in ndims){
      for (IW in IWs){
        pattern = paste(c('*_', IW, '_', model, '_', missing, '_', ndim, '.csv'), collapse = '')
        
        list_of_files <- list.files(path = path,
                                    full.names = TRUE,
                                    pattern = pattern)
        
        sum_results = list_of_files %>%
          lapply(read_csv, show_col_types=F) %>%
          bind_rows() %>%
          mutate(missing=round(missing,8)) %>%
          group_by(missing, model, mirt_dim, parameter, i, j) %>%
          summarise(mean=mean(value),
                    var=var(value),) %>%
          left_join(true, by=join_by(missing==missing,mirt_dim==ndim, parameter==parameter, i==i,j==j)) %>%
          mutate(bias_sq=(mean-true)^2) %>%
          group_by(model, mirt_dim, parameter, missing) %>%
          summarise(bias_sq=mean(bias_sq),
                    var=mean(var)) %>%
          mutate(mse=bias_sq+var)
        
        sum_results$iw=IW
        
        i = i+1
        results[[i]] = sum_results
      }
    }
  }
}
length(results)
vae_results = results %>%
  bind_rows()


missings = seq(0, .75, length.out=10)
ndims = c(3,10)
results = list()
i=0
for (missing in missings){
  for (ndim in ndims){
    pattern = paste(c('^mirt_.*_', missing, '_', ndim, '.csv$'), collapse = '')

    list_of_files <- list.files(path = path,
                                full.names = TRUE,
                                pattern = pattern)
    
    sum_results = list_of_files %>%
      lapply(read_csv, show_col_types=F) %>%
      bind_rows() %>%
      mutate(i=i-1,
             j=j-1,
             missing=round(missing,8)) %>%
      group_by(missing, model, mirt_dim, parameter, i, j) %>%
      summarise(mean=mean(value),
                var=var(value),) %>%
      left_join(true, by=join_by(missing==missing,mirt_dim==ndim, parameter==parameter, i==i,j==j)) %>%
      mutate(bias_sq=(mean-true)^2) %>%
      group_by(model, mirt_dim, parameter, missing) %>%
      summarise(bias_sq=mean(bias_sq),
                var=mean(var)) %>%
      mutate(mse=bias_sq+var)
    
    i = i+1
    results[[i]] = sum_results
    
  }
}
length(results)
mirt_results = results %>%
  bind_rows() 

mirt_results$iw =1


results = list(mirt_results, vae_results) %>%
  bind_rows()


write.csv(results, '~/Documents/GitHub/VAE-MIRT-Missing/simulation_results.csv')

# Create the formatted table
library(tidyr)
library(kableExtra)
table <- results %>%
  pivot_wider(names_from = iw, values_from = c(var, bias_sq, mse)) %>%
  filter(mirt_dim==10, missing%in%c(0,.25, .75)) %>%
  ungroup() %>%
  select(parameter, missing, model, var_1, bias_sq_1, mse_1, 
         var_5, bias_sq_5, mse_5, 
         var_25, bias_sq_25, mse_25) %>%
  arrange(parameter, missing, factor(model, levels=c('cvae', 'idvae', 'ivae', 'pvae', 'mirt')))

colnames(table) <- sub("_.*", "", colnames(table)) # remove unnecessary labels

# Display the table
latex_table = kable(table, format = "latex", booktabs = TRUE, longtable = TRUE, digits = 4) %>%
  add_header_above(c(" " = 3, "IW=1" = 3, "IW=5" = 3, "IW=25" = 3)) %>%
  collapse_rows(c(1,2),latex_hline='custom',custom_latex_hline = c(1,2)) 

gsub('NA', '-', latex_table)
################
#Correlation
###############



library(dplyr)
path = '~/Downloads/cor_res_mhrm/'
files = list.files(path)

vars_a = vars_d = vars_theta = biases_a = biases_d = biases_theta = mses_a =
  mses_theta=iws = its = mses_cor = models = missings = ndims = mses_d = c()
for (file in files){
  results = scan(paste(c(path, file), collapse = ''))
  if (strsplit(file, '_')[[1]][1] == 'mirt'){
    model = 'mirt'
    iw = 1
    it = strsplit(file, '_')[[1]][2]
    missing = strsplit(file, '_')[[1]][3]
    ndim = as.numeric(substr(strsplit(file, '_')[[1]][4], 1, nchar(strsplit(file, '_')[[1]][4])-4))
  }
  else{
    iw = strsplit(file, '_')[[1]][2]
    it = strsplit(file, '_')[[1]][1]
    model = strsplit(file, '_')[[1]][3]
    missing = strsplit(file, '_')[[1]][4]
    ndim = round(as.numeric(substr(strsplit(file, '_')[[1]][5], 1, nchar(strsplit(file, '_')[[1]][5])-4)),2) 
  }
  
  mses_a = c(mses_a, results[1])
  mses_d = c(mses_d, results[2])
  mses_theta = c(mses_theta, results[3])
  mses_cor = c(mses_cor, results[4])
  vars_a = c(vars_a, results[10])
  vars_d = c(vars_d, results[11])
  vars_theta = c(vars_theta, results[12])
  biases_a = c(biases_a, results[7])
  biases_d = c(biases_d, results[8])
  biases_theta = c(biases_theta, results[9])
  iws = c(iws, iw)
  its = c(its, it)
  models = c(models, model)
  missings = c(missings, as.numeric(missing))
  ndims=c(ndims, ndim)
}


results = data.frame('iteration'=its, 'iw'=iws, 'model'=models, 'missing'=missings, 
                     'ndim'=ndims, 'mse_a'=mses_a,'mse_d'=mses_d, 'mse_theta'=mses_theta, 
                     'bias_a'=biases_a,'bias_d'=biases_d, 'bias_theta'=biases_theta,
                     'var_a'=vars_a,'var_d'=vars_d, 'var_theta'=vars_theta, 'mse_cor'=mses_cor)



results %>%
  group_by(iw, missing, model, ndim) %>%
  summarise('mse a' = mean(mse_a), 
            'se a'=sd(mse_a)/n(),
            'mse cor'= mean(mse_cor), 
            'se cor'= sd(mse_cor)/n(),
            'mse theta'=mean(mse_theta),
            'se theta'=sd(mse_theta)/n(), 
            'mse d'=mean(mse_d),
            'se d'=sd(mse_d)/n())

results %>%
  filter(ndim==10, missing==.5)

results %>%
  group_by(iw, missing, model, ndim) %>%
  filter(iw%in%c(5,25)|model=='mirt') %>%
  summarise('mse_a'=mean(mse_a),
            'mse_d'=mean(mse_d),
            'mse_theta'=mean(mse_theta),
            'mse_cor'=mean(mse_cor)) %>%
  pivot_longer(cols = starts_with("mse_"), 
               names_to = "mse_type", 
               values_to = "mse_value") %>%
  ggplot(aes(x = missing, y = mse_value, color = model, linetype = factor(iw))) +
  geom_line() +
  facet_grid(mse_type ~ ndim, scales='free_y') +
  labs(x = "Missing", y = "MSE Value", color = "Model", linetype = "IW") +
  theme_minimal() +
  theme(legend.position = "bottom")

