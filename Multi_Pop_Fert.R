library(tidyverse)
library(ggplot2)
library(reticulate)
library(data.table)

# FERTILITY DATA ----------------------------------------------------------

# Fertility rates
D <- read.table("asfrRR.txt",header = T)
D <- D %>% filter(Age<51&Age>14,Year<2016,Year>1969) 

# MAB Index
DD <- read.table("mabRR.txt",header = T)
DD <- DD  %>% select(Code,Year,MAB)

DD <- merge(D,DD)
head(DD)

DD <- DD %>%  rename(Country=Code,mx=ASFR,ex=MAB) 

str(DD)
DD$Age <- as.integer(DD$Age)
DD$Country <- as.factor(DD$Country)

# DATA STRUCTURE FOR MODEL ----------------------------------------------------------

DD <- DD %>% data.table
DD[,logmx:=(mx)] # now we are working with rates...we can also try to work using log(mx)
DD[,Country_fact:=as.integer(as.factor(Country))-1]

#DD$logmx[DD$logmx=="-Inf"] <- 0


scale_min_max = function(dat,dat_test)  {
  min_dat = min(dat)
  max_dat = max(dat)
  dat_scaled=(dat-min_dat)/(max_dat-min_dat)
  dat_scaled_test = (dat_test-min_dat)/(max_dat-min_dat)
  return(list(train = dat_scaled, test = dat_scaled_test, min = min_dat, max=max_dat))
}

scale_z = function(dat,dat_test)  {
  mean_dat = mean(dat)
  sd_dat = sd(dat)
  dat_scaled=(dat-mean_dat)/(sd_dat)
  dat_scaled_test = (dat_test-mean_dat)/(sd_dat)
  return(list(train = dat_scaled, test = dat_scaled_test, mean_dat = mean_dat, sd_dat=sd_dat))
}

train = DD[Year < 2001]
test = DD[Year >= 2001]

#scale mx
scaled = scale_min_max(train$logmx, test$logmx)
train$mx_scale = scaled$train
test$mx_scale = scaled$test

#scale e0
scaled2 = scale_min_max(train$ex, test$ex)
train$e0_scale = scaled2$train
test$e0_scale = scaled2$test

# Regression

train_reg = train[,c(2,3,7,8,9),with=F]
test_reg = test[,c(2,3,7,8,9),with=F]

year_scale = scale_min_max(train_reg$Year,test_reg$Year)

train_reg$Year = year_scale[[1]]
test_reg$Year = year_scale[[2]]

#train
x = list(Year      = train_reg$Year,
         Age = train_reg$Age, Country = train_reg$Country_fact, e0=train_reg$e0_scale)

y = (main_output= train_reg$mx_scale)

#test
x_test = list(Year      = test_reg$Year,
              Age = test_reg$Age, Country = test_reg$Country_fact,  e0=test_reg$e0_scale)

y_test = (main_output= test_reg$mx_scale)


# MODEL ----------------------------------------------------------
require(keras)
# Build embedding layers
Year <- layer_input(shape = c(1), dtype = 'float32', name = 'Year')
Age <- layer_input(shape = c(1), dtype = 'int32', name = 'Age')
Country <- layer_input(shape = c(1), dtype = 'int32', name = 'Country')

Age_embed = Age %>% 
  layer_embedding(input_dim = 100, output_dim = 10,input_length = 1, name = 'Age_embed') %>%
  keras::layer_flatten()

Country_embed = Country %>% 
  layer_embedding(input_dim = 49, output_dim = 10,input_length = 1, name = 'Country_embed') %>%
  keras::layer_flatten()

main_output <- layer_concatenate(list(Year,Age_embed,Country_embed
)) %>% 
  layer_dense(units = 128, activation = 'tanh') %>% 
  layer_dropout(0.10) %>% 
  layer_dense(units = 128, activation ='tanh') %>% 
  layer_dropout(0.10) %>% 
  layer_dense(units = 1, activation = 'sigmoid', name = 'main_output')

model <- keras_model(
  inputs = c(Year,Age,Country), 
  outputs = c(main_output))

adam = optimizer_adam(lr=0.0005)
lr_callback = callback_reduce_lr_on_plateau(factor=.80, patience = 8, verbose=1, cooldown = 5, min_lr = 0.00005)
model_callback = callback_model_checkpoint(filepath = "best.mod", verbose = 1,save_best_only = TRUE)

model %>% compile(
  optimizer = adam,
  loss = "mse")

fit = model %>% fit(
  x = x,
  y = y, 
  epochs = 100,
  batch_size = 500,verbose = 1, shuffle = T, validation_split = 0.2, callbacks = list(lr_callback,model_callback))

model = load_model_hdf5("best.mod")

test$mx_deep_reg_full = model %>% predict(x_test)
test[,mx_deep_reg_full:=exp(mx_deep_reg_full*(scaled$max-scaled$min)+scaled$min)]
test[,.(        Deep_reg = sum((mx-mx_deep_reg_full)^2)), keyby = .(Country)
     ] %>% fwrite("all_country.csv")

test %>% filter(Country=="AUT",Year==2010) %>% ggplot(aes(Age,mx))+geom_point()
test %>% filter(Country=="AUT",Year==2010) %>% ggplot(aes(Age,mx_deep_reg_full))+geom_point()


