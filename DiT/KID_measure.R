library(keras)
library(tensorflow)
packageVersion("tensorflow")
library(tfdatasets)
library(abind)
library(EBImage)
library(progress)

# data
VAE_dir = "../AutoEncoder/VAE-f4"
checkpoint_dir = NULL
DiT_dir = "../UNET/ema_unet(Epoch 50)"
image_size = 256L
latent_size = image_size %/% 4L
scale_factor = 0.14581

# diffusion algorithmic
diffusion_schedule = "cosine" # "linear" or "cosine"
# linear schedule
beta_start = 0.00085
beta_end = 0.0120
# cosine schedule
alphas_cumprod_start = 0.999
alphas_cumprod_end = 0.001 # should be set relatively large to avoid extreme values during sampling.
timesteps = 1000L
noise_offset = 0.0
use_ema = T

num_images = 10000
batch_size = 32L
mixed_precision = FALSE


source("dataset.R")
source("diffusion_model.R")

KID <- new_metric_class(
  "KID",
  initialize = function(kid_image_size = 75L, name = name, ...){
    super$initialize(...)
    self$kid_image_size <- as.integer(kid_image_size)
    # KID is estimated per batch and is averaged across batches
    self$kid_tracker = keras$metrics$Mean(name="kid_tracker")
    
    # a pretrained InceptionV3 is used without its classification layer
    # input the pixel values from [0,255] range, then use the same
    # preprocessing as during pretraining
    
    self$encoder <- keras_model_sequential(
      list(layer_input(shape = c(image_size, image_size, 3L)),
           layer_resizing(height = self$kid_image_size, width = self$kid_image_size),
           layer_lambda(f = keras$applications$inception_v3$preprocess_input),
           keras$applications$InceptionV3(
             include_top = FALSE,
             input_shape = c(self$kid_image_size, self$kid_image_size, 3L),
             weights="imagenet",
           ),
           layer_global_average_pooling_2d()),
      name="inception_encoder"
      )
  },
  
  polynomial_kernel = function(features_1, features_2){
    feature_dimensions <- tf$cast(tf$shape(features_1)[2], dtype = "float32")
    return (tf$matmul(features_1, features_2, transpose_b = T) / feature_dimensions + 1.0) ^ 3.0
  },
  
  update_state = function(real_images, generated_images, sample_weight = NULL){
    real_features <- self$encoder(real_images, training=FALSE)
    generated_features <- self$encoder(generated_images, training=FALSE)
    
    # compute polynomial kernels using the two sets of features
    kernel_real = self$polynomial_kernel(real_features, real_features)
    kernel_generated = self$polynomial_kernel(generated_features, generated_features)
    kernel_cross = self$polynomial_kernel(real_features, generated_features)
    
    # estimate the squared maximum mean discrepancy using the average kernel values
    batch_size = tf$shape(real_features)[1]
    batch_size_f = tf$cast(batch_size, dtype = "float32")
    mean_kernel_real = tf$reduce_sum(kernel_real * (1.0 - tf$eye(batch_size))) / (
      batch_size_f * (batch_size_f - 1.0)
    )
    mean_kernel_generated = tf$reduce_sum(
      kernel_generated * (1.0 - tf$eye(batch_size))
    ) / (batch_size_f * (batch_size_f - 1.0))
    mean_kernel_cross = tf$reduce_mean(kernel_cross)
    kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross
    
    # update the average KID estimate
    self$kid_tracker$update_state(kid)
  },
  
  result = function(){
    return(self$kid_tracker$result())
  },
  
  reset_state = function(){
    self$kid_tracker$reset_state()
  }
)
KID_messure <- KID(name = "KID")

VAE <- load_model_tf(VAE_dir)
Encoder <- VAE$get_layer("encoder")
Decoder <- VAE$get_layer("decoder")
Encoder$trainable <- FALSE
Decoder$trainable <- FALSE
rm(VAE);gc()

model <- DiffusionModel(DiT = NULL,
                        EMA_DiT = load_model_tf(DiT_dir),
                        Encoder = Encoder,
                        Decoder = Decoder)

pb <- progress_bar$new(
  format = "[:bar] :current/:total ETA::eta",
  total = ceiling(num_images/batch_size),
  width = 60
)
for (i in 1:ceiling(num_images/batch_size)){
  real_images <- train_dataset %>% 
    reticulate::as_iterator() %>% 
    reticulate::iter_next() %>% 
    {.[[1]]}
    {(. + 1.0) * 127.5}
  
  generated_images <- model$generate(num_images = batch_size,
                                     diffusion_steps = 20,
                                     eta = 0.0) %>% {(. + 1.0) * 127.5}
  
  KID_measure$update_state(real_images, generated_images)
  pb$tick()
}
KID_measure$result()
