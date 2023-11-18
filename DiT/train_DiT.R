# Rscript train_DiT.R
#################### Hyperparameterers ####################  
# data
setwd("/home/neko/DL/DiT/DiT/")
image_size = 256L
latent_size = image_size %/% 8L
scale_factor = 0.23516
data_dir = "/mnt/e/512AnimeFaces/test"
VAE_dir = "/home/neko/DL/DiT/AutoEncoder/VAE-f8"
token_vocabulary_dir = "token_vocabulary.Rdata"
work_dir = "/root/autodl-tmp"

# diffusion algorithmic
diffusion_schedule = "cosine" # "linear" or "cosine"
beta_start = 1e-4
beta_end = 0.02
alphas_cumprod_start = 0.999
alphas_cumprod_end = 0.001 # should be set relatively large to avoid extreme values during sampling.
timesteps = 1000L
noise_offset = 0.0
use_ema = T

# architecture
patch_size = 2L
embedding_dims = 256
width = 1024
num_layers = 16
num_heads = 16
mlp_ratio = 2.0
out_channels = 4L
dropout_rate = 0.1
text_dropout = 0.2
max_text_len = 16L
   
# optimization
epochs = 100L
batch_size = 32L
learning_rate = 5e-5
weight_decay = 1e-4
clipnorm = NULL
ema = 0.999
warmup_steps = 10000
mixed_precision = TRUE







####################  Don't modify the following code ####################  
library(keras)
library(tensorflow)
packageVersion("tensorflow")
library(tfdatasets)
library(abind)
library(EBImage)
library(progress)

source("dataset.R")
source("architecture.R")
source("DiT_callback.R")
source("diffusion_model.R")

# load VAE 
VAE <- load_model_tf(VAE_dir)
Encoder <- VAE$get_layer("encoder")
Decoder <- VAE$get_layer("decoder")
Encoder$trainable <- FALSE
Decoder$trainable <- FALSE
rm(VAE);gc()

# load token vocabulary
load(token_vocabulary_dir)
num_tokens = length(token_vocabulary) + 2L # including "[null]" and "[UNK]"
 
# Data pipeline
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
  {.[[1]]} %>% 
  as.array() %>% 
  {(. + 1.0) * 127.5} %>% 
  {.[1,,,]} %>% 
  as.raster(max = 255) %>% 
  plot()

test_dataset %>% 
  reticulate::as_iterator() %>% 
  reticulate::iter_next() %>% 
  {.[[1]]} %>% 
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

lr_schedule <- keras$optimizers$schedules$CosineDecay(initial_learning_rate = learning_rate, 
                                                      warmup_steps = warmup_steps,
                                                      decay_steps = length(train_dataset) * epochs - warmup_steps)
optimizer <- keras$optimizers$AdamW(learning_rate = lr_schedule, weight_decay = weight_decay, clipnorm = clipnorm)
# Exclude Norm layer and bias terms from weight decay.
optimizer$exclude_from_weight_decay(var_names = list("bias", "gamma", "beta"))
if (mixed_precision){
  optimizer <- tf$keras$mixed_precision$LossScaleOptimizer(optimizer)
}

model <- DiffusionModel(DiT = get_DiT(),
                        EMA_DiT = get_DiT(),
                        Encoder = Encoder,
                        Decoder = Decoder)
model %>% compile(
  optimizer = optimizer, 
  loss = keras$losses$mean_squared_error
)


# Train
# Set work dir
setwd(work_dir)

if (!fs::dir_exists("tf-logs")) fs::dir_create("tf-logs")
tensorboard("tf-logs", port = 6007)

checkpoint_filepath = "checkpoint/model_weights"
model_checkpoint_callback <- callback_model_checkpoint(
  filepath = checkpoint_filepath,
  save_best_only = TRUE,
  save_weights_only = TRUE,
  monitor = "n_loss",
  mode = "min",
  save_freq = "epoch",
  verbose = 1)

model %>% fit(train_dataset,
              epoch = epochs,
              validation_data = test_dataset,
              callbacks = list(gan_monitor(save_epochs = 5,plot_diffusion_steps = 20,plot = T),
                               callback_tensorboard(log_dir = "tf-logs",histogram_freq = 1,update_freq=1000L),
                               model_checkpoint_callback)
              )
