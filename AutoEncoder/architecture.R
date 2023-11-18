layer_group_normalization <- function(object, groups = 32, axis = -1, epsilon = 0.001, center = TRUE, 
                                      scale = TRUE, beta_initializer = "zeros", gamma_initializer = "ones", 
                                      beta_regularizer = NULL, gamma_regularizer = NULL, beta_constraint = NULL, 
                                      gamma_constraint = NULL, trainable = TRUE, name = NULL){
  create_layer(keras$layers$GroupNormalization, object, 
               list(groups = as.integer(groups), axis = as.integer(axis), epsilon = epsilon, 
                    center = center, scale = scale, 
                    beta_initializer = beta_initializer, gamma_initializer = gamma_initializer, 
                    beta_regularizer = beta_regularizer, gamma_regularizer = gamma_regularizer, 
                    beta_constraint = beta_constraint, gamma_constraint = gamma_constraint, 
                    trainable = trainable, name = name))
}

ResidualBlock <- new_layer_class(
  "ResidualBlock",
  initialize = function(width, dropout = 0.0, ...){
    super$initialize(...)
    self$width <- as.integer(width)
    self$dropout <- dropout
  },
  
  build = function(input_shape){
    self$norm1 <- layer_group_normalization(groups = 16, epsilon = 1e-5, name = "norm1")
    self$conv1 <- layer_conv_2d(filters = self$width, kernel_size = 3, padding = "same", name = "conv1")
    
    self$norm2 <- layer_group_normalization(groups = 16, epsilon = 1e-5, name = "norm2")
    self$dropout <- layer_dropout(rate = self$dropout)
    self$conv2 <- layer_conv_2d(filters = self$width, kernel_size = 3, padding = "same", name = "conv2")
    
    if(input_shape[[4]] == self$width){
      self$residual <- tf$identity
    } else {
      self$residual <- layer_conv_2d(filters = self$width, kernel_size = 1, name = "residual")
    }
  },
  
  call = function(input, training = NULL){
    residual <- self$residual(input)
    
    x <- input %>% 
      self$norm1() %>% 
      layer_activation(activation = "swish") %>% 
      self$conv1()
    
    x <- x %>% 
      self$norm2() %>% 
      layer_activation(activation = "swish") %>% 
      self$dropout(training = training) %>% 
      self$conv2()
    
    x <- layer_add(x, residual)
    
    return(x)
  },
  
  get_config = function(){
    config <- super$get_config()
    config$width <- self$width
    config$dropout <- self$dropout
    config
  }
)

MultiHeadSelfAttention <- new_layer_class(
  "MultiHeadSelfAttention",
  initialize = function(num_heads = 8, dropout = 0.0, ...){
    super$initialize(...)
    self$num_heads <- as.integer(num_heads)
    self$dropout <- dropout
  },
  
  build = function(input_shape){
    self$c <- input_shape[[4]]
    self$norm <- layer_group_normalization(groups = 16, epsilon = 1e-5, name = "norm")
    self$multi_attn <- layer_multi_head_attention(num_heads = self$num_heads, 
                                                  key_dim = self$c %/% self$num_heads, 
                                                  attention_axes = c(1L,2L), dropout = self$dropout,
                                                  name = "multi_attn")
  },
  
  call = function(input, training = NULL){
    residual <- input
    
    x <- input %>% 
      self$norm() %>% 
      self$multi_attn(., .)
    
    x <- layer_add(x, residual)
    
    return(x)
  },
  
  get_config = function(){
    config <- super$get_config()
    config$num_heads <- self$num_heads
    config$dropout <- self$dropout
    config
  }
)

DiagonalGaussianDistribution <- new_layer_class(
  "DiagonalGaussianDistribution",
  initialize = function(...){
    super$initialize(...)
  },
  
  call = function(inputs, training = NULL){
    c(mean, log_var) %<-% tf$split(inputs, num_or_size_splits = 2L, axis = -1L)
    
    # compute kl loss
    self$add_loss((0.5 * tf$square(mean) + tf$exp(log_var) - 1.0 - log_var) %>% 
                    tf$reduce_mean())
    
    # sampling
    std <- tf$exp(0.5 * log_var)
    z <- mean + std * tf$random$normal(tf$shape(std), dtype = "float32")
    
    return(z)
  },
  
  get_config = function(){
    config <- super$get_config()
    config
  }
)

get_encoder <- function(){
  input <- layer_input(shape = c(image_size, image_size, 3L))
  
  output <- input %>% 
    layer_conv_2d(filters = widths[1], kernel_size = 3, padding = "same") %>% 
    
    ResidualBlock(width = widths[1], dropout = dropout_rate) %>% 
    ResidualBlock(width = widths[1], dropout = dropout_rate) %>% 
    
    # downsample
    layer_conv_2d(filters = widths[1], kernel_size = 3, strides = 2, padding = "same", name = "downsample_1") %>% 
    
    ResidualBlock(width = widths[2], dropout = dropout_rate) %>% 
    ResidualBlock(width = widths[2], dropout = dropout_rate) %>%
    
    # downsample
    layer_conv_2d(filters = widths[2], kernel_size = 3, strides = 2, padding = "same", name = "downsample_2") %>% 
    
    ResidualBlock(width = widths[3], dropout = dropout_rate) %>% 
    ResidualBlock(width = widths[3], dropout = dropout_rate) %>%
    
    # downsample
    layer_conv_2d(filters = widths[3], kernel_size = 3, strides = 2, padding = "same", name = "downsample_3") %>% 
    
    ResidualBlock(width = widths[3], dropout = dropout_rate) %>% 
    ResidualBlock(width = widths[3], dropout = dropout_rate) %>%
    
    ResidualBlock(width = widths[3], dropout = dropout_rate) %>% 
    MultiHeadSelfAttention(num_heads = 8, dropout = dropout_rate) %>% 
    ResidualBlock(width = widths[3], dropout = dropout_rate) %>% 
    
    layer_group_normalization(groups = 16, epsilon = 1e-5) %>% 
    layer_activation(activation = "swish") %>% 
    layer_conv_2d(filters = 2 * z_channels, kernel_size = 3, padding = "same")
  
  quant_conv <- output %>% 
    layer_conv_2d(filters = 2 * z_channels, kernel_size = 1, name = "quant_conv")
  
  sample_z <- quant_conv %>% 
    DiagonalGaussianDistribution(dtype = "float32", name = "DiagonalGaussianDistribution")
  
  encoder <- keras_model(input, sample_z, name = "encoder")
  return(encoder)
}

get_decoder <- function(){
  input <- layer_input(shape = c(image_size / 8, image_size / 8, z_channels))
  
  post_quant_conv <- input %>% 
    layer_conv_2d(filters = z_channels, kernel_size = 1, name = "post_quant_conv")
  
  output <- post_quant_conv %>%
    layer_conv_2d(filters = widths[3], kernel_size = 3, padding = "same") %>% 
    
    ResidualBlock(width = widths[3], dropout = dropout_rate) %>% 
    MultiHeadSelfAttention(num_heads = 8, dropout = dropout_rate) %>% 
    ResidualBlock(width = widths[3], dropout = dropout_rate) %>% 
    
    ResidualBlock(width = widths[3], dropout = dropout_rate) %>% 
    ResidualBlock(width = widths[3], dropout = dropout_rate) %>%
    
    # upsample
    ResidualBlock(width = widths[3], dropout = dropout_rate) %>%
    layer_upsampling_2d(size = c(2L,2L), interpolation = "nearest") %>% 
    layer_conv_2d(filters = widths[3], kernel_size = 3, strides = 1, padding = "same", name = "upsample_1") %>%
    
    ResidualBlock(width = widths[3], dropout = dropout_rate) %>% 
    ResidualBlock(width = widths[3], dropout = dropout_rate) %>%
    
    # upsample
    ResidualBlock(width = widths[3], dropout = dropout_rate) %>%
    layer_upsampling_2d(size = c(2L,2L), interpolation = "nearest") %>% 
    layer_conv_2d(filters = widths[3], kernel_size = 3, strides = 1, padding = "same", name = "upsample_2") %>% 
    
    ResidualBlock(width = widths[2], dropout = dropout_rate) %>% 
    ResidualBlock(width = widths[2], dropout = dropout_rate) %>%
    
    # upsample
    ResidualBlock(width = widths[2], dropout = dropout_rate) %>%
    layer_upsampling_2d(size = c(2L,2L), interpolation = "nearest") %>% 
    layer_conv_2d(filters = widths[2], kernel_size = 3, strides = 1, padding = "same", name = "upsample_3") %>% 
    
    ResidualBlock(width = widths[1], dropout = dropout_rate) %>% 
    ResidualBlock(width = widths[1], dropout = dropout_rate) %>%
    
    layer_group_normalization(groups = 16, epsilon = 1e-5) %>% 
    layer_activation(activation = "swish") %>% 
    layer_conv_2d(filters = 3, kernel_size = 3, padding = "same", activation = "tanh", dtype = "float32", name = "last_layer")
  
  decoder <- keras_model(input, output, name = "decoder")
  return(decoder)
}

get_KLVAE <- function(){
  encoder <- get_encoder()
  decoder <- get_decoder()
  
  input <- layer_input(shape = c(image_size, image_size, 3L))
  
  encoder_output <-  encoder(input)
  
  output <- decoder(encoder_output)
  
  KLVAE <- keras_model(input, output, name = "KLVAE")
  return(KLVAE)
}