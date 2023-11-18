L1_loss <- function(real_samples, reconstructed_samples){
  L1_loss <- tf$abs(real_samples - reconstructed_samples)
  return(tf$reduce_mean(L1_loss))
}

L2_loss <- function(real_samples, reconstructed_samples){
  L2_loss <- tf$square(real_samples - reconstructed_samples)
  return(tf$reduce_mean(L2_loss))
}

Receptive_loss <- new_loss_class(
  "Receptive_loss",
  # compute receptive loss in dtype float32 to avoid "Inf"
  initialize = function(image_size, ...){
    super$initialize(...)
    self$encoder_layers <- list(
      "block1_conv1",
      "block2_conv1",
      "block3_conv1",
      "block4_conv1",
      "block5_conv1"
    )
    self$weights <- c(1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0)
    # create float32 VGG19
    tf$keras$mixed_precision$set_global_policy("float32")
    vgg <- application_vgg19(include_top = FALSE, weights = "imagenet", input_shape = c(image_size, image_size, 3))
    if (mixed_precision){
      tf$keras$mixed_precision$set_global_policy("mixed_float16")
    }
    layer_output <- list()
    for(x in self$encoder_layers){
      layer_output <- c(layer_output, vgg$get_layer(x)$output)
    }
    self$vgg_model <- keras_model(vgg$input, layer_output, name = "VGG19")
    self$mae <- keras$losses$MeanAbsoluteError()
  },
  
  call = function(y_true, y_pred){
    y_true <- keras$applications$vgg19$preprocess_input(127.5 * (y_true + 1.0))
    y_pred <- keras$applications$vgg19$preprocess_input(127.5 * (y_pred + 1.0))
    real_features <- self$vgg_model(y_true)
    fake_features <- self$vgg_model(y_pred)
    loss <- 0.0
    for(i in 1:length(real_features)){
      loss <- loss + self$weights[i] * self$mae(real_features[i], fake_features[i])
    }
    return(loss)
  }
)