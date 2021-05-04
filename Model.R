
########################################
##                                    ##
##       Andrea Nigri                 ##
##                                    ##
##         andrea.nigri88@gmail.com   ##
##                                    ##
########################################

### --------------------------------------------------------------------------
### R Code: DNN mx from e0
### --------------------------------------------------------------------------

## For each code, please check technical info about packages for DNN implementation:

## R version 4.0.1 (2020-06-06)
## Platform: x86_64-w64-mingw32/x64 (64-bit)
## Running under: Windows 10 x64 (build 18363)
## Intel(R) Core(TM) i7 - 8550U CPU @ 1.80GHz 1.99GHz - RAM 16 GB

### --- Technical info about packages for DNN implementation
### 
### Keras version (R interface): 2.3.0.0
### Tensorflow version (R interface): 2.2.0
### Tensorflow version (Python module for backend tensor operations): 1.10.1 (Python module)
### Python version : 3.6

### Data source : Mortality Rates from Human Mortality Database
### Period : 1947-last
### Country : ITA
### Gender : Males
### Forecasting period : ys-last

# Library, seed and error function -----------------------------------------------------------------

# DATA
# 

library(keras)
# Function

mac_mse <- function(y_test,test_predictions,ages){
  
  ages  <- (ages)
  l <- (dim(y_test)[1])
  mac_obs <-( rep(NA,l))
  mac_nn <- (rep(NA,l))
  
  for (i in 1:l) {
    
    mac_obs[i] <- sum(sum( as.numeric((ages*(y_test[i,])*100)))/sum(as.numeric((y_test[i,])*100)))
    mac_nn[i]<- sum(sum( as.numeric((ages*(test_predictions[i,])*100)))/sum(as.numeric((test_predictions[i,])*100)))
  }
  
  K   <- backend()
  # calculate the metric
  loss <- K$sum((K$pow(mac_obs - mac_nn, 2))) 
  return(loss)
}


wlse_wrapper <- custom_metric("mac_mse", function(y_test,test_predictions) {
  mac_mse(y_test,test_predictions)})



# Error functions 
rmse = function (truth, prediction)  {
  sqrt(mean((prediction - truth)^2))
}
mae = function(truth, prediction){
  mean(abs(prediction-truth))
}

options("scipen"=100, "digits"=4)
theme_set(theme_bw(base_family = "mono",base_size = 15))
set.seed(123)


# DATA
library(keras)
library(data.table)
library(reshape2)
library(ggplot2)
library(tidyverse)
library(splines)


load("ex.RData")
load("mx.RData")
load("ex_val.RData")
load("mx_val.RData")

# years
ys <- 1965
yf <- 1995
years <- ys:yf
last <- 2005

# Ages
ages <- 15:50


##--------------------------------------------------------------------
##
##  Deep Neural Network
##  
##--------------------------------------------------------------------
set.seed(123)
# Row sampling for train-test split
smp=sample(1:nrow(ex),nrow(ex)/1.2)

#TRAIN
x_train <- as.matrix(ex[smp,c(1,2)]) #x_train
y_train <- as.matrix(mx[smp,]) #y_train
year_X_train <- sort(x_train[,1])
year_Y_train <- sort(y_train[,1])

##TEST
x_test <- as.matrix(ex[-smp,c(1,2)]) #x_test
y_test <- as.matrix(mx[-smp,]) #y_test

year_X_test <- sort(x_test[,1])
year_Y_test <- sort(y_test[,1])


# DELETE FIRST COLUMN: YEAR
x_train <- x_train[,-c(1)] 
y_train <- y_train[,-c(1)]

x_test <- x_test[,-c(1)]
y_test <- y_test[,-c(1)]

# Normalize training data
x_train <- scale(x_train) 

# Use means and standard deviations from training set to normalize test set
col_means_train <- attr(x_train, "scaled:center") 
col_stddevs_train <- attr(x_train, "scaled:scale")
x_test <- scale(x_test, center = col_means_train, scale = col_stddevs_train)

##################################
##                              ##
##  DNN MODEL                   ##
##                              ##
##################################


ep <- 800
# Grid search to get best number of units
tuning = expand.grid(epochs = ep, unit=seq(325,325,1)) 
tuning = cbind(tuning, performance = rep(0, nrow(tuning)))

for (g in 1:(nrow(tuning))){
  
  use_session_with_seed(42)
  build_model <- function() {
    
    model <- keras_model_sequential() %>%
      layer_dense(units =tuning$unit[g], activation = 'relu',
                  input_shape = dim(x_train)[2]) %>%
      layer_dense(units = tuning$unit[g], activation = "relu") %>%
      layer_dense(units = tuning$unit[g], activation = "relu") %>%
      layer_dense(units = tuning$unit[g], activation = "relu") %>%
      layer_dense(units = tuning$unit[g], activation = "relu") %>%
      layer_dense(units = tuning$unit[g], activation = "relu") %>%
      layer_dense(units = tuning$unit[g], activation = "relu") %>%
      layer_dense(units = tuning$unit[g], activation = "relu") %>%
      layer_dense(units = tuning$unit[g], activation = "relu") %>%
      layer_dense(units = tuning$unit[g], activation = "relu") %>%
      layer_dense(units = tuning$unit[g], activation = "relu") %>%
      layer_dense(units = 36,activation = "sigmoid")
    
    model %>% compile(
      loss = wlse_wrapper,
      optimizer = optimizer_rmsprop(), 
      metrics = list("mean_absolute_error"))
    
    model}
  
  model <- build_model()
  model %>% summary()
  
    print_dot_callback <- callback_lambda(
    on_epoch_end = function(epoch, logs) {
      if (epoch %% 80 == 0) cat("\n")
      cat(".")
    })    
  
  CBs <- callback_early_stopping(monitor = "val_loss", min_delta = 0.0005,
                                 patience = 100, verbose = 1, mode = 'min',
                                 baseline = NULL, restore_best_weights = TRUE)
  
  epochs <- ep

  history <- model %>% fit(
    x_train,
    y_train,
    epochs = epochs,
    batch_size = 1,
    validation_data = list(x_test, y_test),
    verbose = 2,
    callbacks = CBs)
  
  plot(history, metrics = "mean_absolute_error", smooth = T) +
    coord_cartesian(ylim = c(0, 5))
  
  history
  test_predictions <- model %>% predict(x_test)
  test_predictions
  
  #########################################
  ##                                     ##
  ## Validation - Backtesting: 2006-last ##
  ##                                     ##
  #########################################
  
  # e0 validation
  x_val <- as.matrix(ex_val[,c(2)])
  # e0 validation scaled
  x_val_sc <- scale(x_val, center = col_means_train, scale = col_stddevs_train)
  
  # NN prediction
  val_predictions <- model %>% predict(x_val_sc)
  val_predictions
  
  
  tuning$performance[g] = rmse(mx_val,val_predictions)
  View(tuning)
}
