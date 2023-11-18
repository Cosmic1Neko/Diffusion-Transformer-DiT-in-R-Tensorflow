# Rscript train_vae.R
#################### Hyperparameterers ####################  
# data
image_size = 256L
data_dir = "/mnt/e/512AnimeFaces/test"
work_dir = ""

# architecture
z_channels = 4
widths = c(80, 160, 320)

# optimization
epochs = 20L
batch_size = 16L
learning_rate = 1e-4
lambda = 1e-5 # weight of KL loss
dropout_rate = 0.0
weight_decay = 0.01
clipnorm = NULL
ema = 0.999
warmup_steps = 6000
mixed_precision = TRUE









####################  Don't modify the following code ####################  
library(keras)
library(tensorflow)
packageVersion("tensorflow")
library(tfdatasets)
library(abind)
library(EBImage)

source("dataset.R")
source("architecture.R")
source("vae_callback.R")
source("KLVAE_model.R")
source("loss.R")



train_dataset <- prepare_dataset(data_dir,
                                 validation_split = 0.01,
                                 subset = "training",
                                 seed = 1919810L)
test_dataset <- prepare_dataset(data_dir,
                                validation_split = 0.01,
                                subset = "validation",
                                seed = 1919810L)

train_dataset %>% 
  reticulate::as_iterator() %>% 
  reticulate::iter_next() %>% 
  as.array() %>% 
  {(. + 1.0) * 127.5} %>% 
  {.[1,,,]} %>% 
  as.raster(max = 255) %>% 
  plot()

test_dataset %>% 
  reticulate::as_iterator() %>% 
  reticulate::iter_next() %>% 
  as.array() %>% 
  {(. + 1.0) * 127.5} %>% 
  {.[1,,,]} %>% 
  as.raster(max = 255) %>% 
  plot()



# mixed precision
# Note: In this script, all the model will output float32, but model still compute with float16
if (mixed_precision){
  tf$keras$mixed_precision$set_global_policy("mixed_float16")
}

# VAE optimizer
lr_schedule <- keras$optimizers$schedules$CosineDecay(initial_learning_rate = learning_rate, 
                                                      warmup_steps = warmup_steps,
                                                      decay_steps = length(train_dataset) * epochs - warmup_steps)
optimizer <- keras$optimizers$AdamW(learning_rate = lr_schedule, weight_decay = weight_decay, clipnorm = clipnorm)
# Exclude Norm layer and bias terms from weight decay.
optimizer$exclude_from_weight_decay(var_names = list("bias", "gamma", "beta"))
if (mixed_precision){
  optimizer <- tf$keras$mixed_precision$LossScaleOptimizer(optimizer)
}

model <- KLVAE(VAE = get_KLVAE(),
               EMA_VAE = get_KLVAE())

model %>% compile(
  optimizer = optimizer, 
  reco_loss_fn = L1_loss, 
  recp_loss_fn = Receptive_loss(image_size)
) 

# Set work dir
setwd(work_dir)

if (!fs::dir_exists("tf-logs")) fs::dir_create("tf-logs")
tensorboard("tf-logs", port = 6007)

model %>% fit(train_dataset,
             epoch = epochs,
             validation_data = test_dataset,
             callbacks = list(gan_monitor(save_epochs = 1,plot = T),
                              callback_tensorboard(log_dir = "tf-logs",histogram_freq = 1,update_freq=1000L))
             )
