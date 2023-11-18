# Rscript scale_factor.R
library(keras)
library(tensorflow)
packageVersion("tensorflow")
library(tfdatasets)
library(abind)
library(EBImage)

image_size = 64L
batch_size = 8L
data_dir = "/mnt/e/512AnimeFaces/512AnimeFaces"
VAE_dir = "/home/neko/LDM/AutoEncoder/VAE"

# Set work dir
#setwd("/root/autodl-tmp")

VAE <- load_model_tf(VAE_dir)
Encoder <- VAE$get_layer("encoder")
Decoder <- VAE$get_layer("decoder")
Encoder$trainable <- FALSE
Decoder$trainable <- FALSE
rm(VAE);gc()

source("dataset.R")

train_dataset <- prepare_dataset(data_dir,
                                 validation_split = 0.01,
                                 subset = "training",
                                 seed = 1919810L)
test_dataset <- prepare_dataset(data_dir,
                                validation_split = 0.01,
                                subset = "validation",
                                seed = 1919810L)

compute_scale_factor <- new_model_class(
  "compute_scale_factor",
  initialize = function(encoder){
    super$initialize()
    self$encoder <- encoder
  },
  
  compile = function(){
    super$compile()
    self$var_tracker <- keras$metrics$Mean(name="var")
  },
  
  metrics = mark_active(function() {
    list(self$var_tracker)
  }),
  
  train_step = function(images){
    latent_images <- self$encoder(images, training = FALSE)
    mean <- tf$reduce_mean(latent_images)
    var <- tf$reduce_mean((latent_images - mean) ^ 2)
    
    self$var_tracker$update_state(var)
    
    results <- list()
    for (m in self$metrics)
      results[[m$name]] <- m$result()
    results
  }
)

model <- compute_scale_factor(encoder = Encoder)
model %>% compile()
model %>% fit(train_dataset, epoch = 1)


std <- model$var_tracker$result() %>% 
  tf$sqrt()
scale_factor = 1.0 / std
print(scale_factor)