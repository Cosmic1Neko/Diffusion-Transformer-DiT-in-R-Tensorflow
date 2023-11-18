patch_position_embedding <- new_layer_class(
  "Patch_and_Position_Embedding",
  initialize = function(patch_size = 2, width = 768, ...) {
    super$initialize(...)
    self$patch_size <- as.integer(patch_size)
    self$width <- as.integer(width)
  },
  
  build = function(input_shape){
    img_size <- input_shape[[2]]
    self$num_patches <- as.integer((img_size / patch_size) ^ 2)
    self$embed_conv <- layer_conv_2d(filters = self$width,
                                     kernel_size = self$patch_size,
                                     strides = self$patch_size,
                                     name = "patch_embedding")
    self$position_embedding <- layer_embedding(input_dim = self$num_patches,
                                               output_dim = self$width,
                                               name = "position_embedding")
    self$positions <- tf$range(start = 0, limit = self$num_patches, delta = 1)
  },
  
  call = function(input){
    patches <- self$embed_conv(input)
    embed_patches <- tf$reshape(patches, shape = c(-1L, self$num_patches, self$width)) # bhwc -> bnc
    embed_positions <- self$position_embedding(self$positions)
    emb <- embed_patches + embed_positions
    return(emb)
  },
  
  get_config = function(){
    config <- super$get_config()
    config$patch_size <- self$patch_size
    config$width <- self$width
    config
  }
)

timestep_embedding <- new_layer_class(
  "Timestep_Embedding",
  # A sinusoidal embedding layer for diffusion timestep embedding
  # used for discrete times diffusion model
  initialize = function(embedding_dims, width, ...) {
    super$initialize(...)
    self$embedding_dims <- as.integer(embedding_dims)
    self$width <- as.integer(width)
    self$half_dim <- self$embedding_dims %/% 2
    self$emb <- tf$math$log(10000) / (self$half_dim - 1.0)
    self$emb <- tf$exp(tf$range(self$half_dim, dtype="float32") * (-self$emb))
    
    self$ffn1 <- layer_dense(units = self$width, name = "ffn1")
    self$ffn2 <- layer_dense(units = self$width, name = "ffn2")
  },
  
  call = function(x) {
    x <- tf$cast(x, "float32")
    embeddings <- tf$concat(list(tf$sin(x * self$emb), tf$cos(x * self$emb)), axis=-1L)
    
    x <- embeddings %>% 
      self$ffn1() %>% 
      layer_activation(activation = "swish") %>% 
      self$ffn2() %>% 
      tf$reshape(shape = c(-1L, self$width))
    
    return(x)
  },
  
  get_config = function(){
    config <- super$get_config()
    config$embedding_dims <- self$embedding_dims
    config$width <- self$width
    config
  }
)

BasicTransformerBlock <- new_layer_class(
  "BasicTransformerBlock",
  # It is a standard transformer encoder block.
  # Input shape: (b, t, c)
  # Output shape: (b, t, c)
  initialize = function(width, num_heads, dropout = 0.0, ...){
    super$initialize(...)
    self$width <- as.integer(width)
    self$num_heads <- as.integer(num_heads)
    self$dropout <- dropout
  },
  
  build = function(input_shape){
    self$norm_attn <- layer_layer_normalization(epsilon = 1e-5, name = "norm_attn")
    self$multi_attn <- layer_multi_head_attention(num_heads = self$num_heads, 
                                                  key_dim = self$width %/% self$num_heads,
                                                  dropout = self$dropout,
                                                  name = "multi_attn")
    
    self$norm_ffn <- layer_layer_normalization(epsilon = 1e-5, name = "norm_ffn")
    self$ffn1 <- layer_dense(units = self$width * 4, name = "ffn1")
    self$dropout <-layer_dropout(rate = self$dropout, name = "dropout")
    self$ffn2 <- layer_dense(units = self$width, name = "ffn2")
  },
  
  call = function(input, mask = NULL, training = NULL){
    attn_out <- input %>% 
      self$norm_attn() %>% 
      self$multi_attn(., .) %>% 
      layer_add(., input)
    
    ffn_out <- attn_out %>% 
      self$norm_ffn() %>%
      self$ffn1() %>% 
      layer_activation(activation = "gelu") %>% 
      self$dropout(training = training) %>% 
      self$ffn2() %>% 
      layer_add(., attn_out)
    
    return(ffn_out)
  },
  
  get_config = function(){
    config <- super$get_config()
    config$width <- self$width
    config$num_heads <- self$num_heads
    config$dropout <- self$dropout
    config
  }
)

text_embedding <- new_layer_class(
  "Text_Embedding",
  initialize = function(num_tokens, max_text_len, width, num_heads, dropout = 0.0, ...) {
    super$initialize(...)
    self$num_tokens <- as.integer(num_tokens)
    self$max_text_len <- as.integer(max_text_len)
    self$width <- as.integer(width)
    self$num_heads <- as.integer(num_heads)
    self$dropout <- dropout
  },
  
  build = function(input_shape){
    self$token_embedding <- layer_embedding(input_dim = self$num_tokens, output_dim = self$width, 
                                            mask_zero = T, name = "token_embedding")
    self$encoder_1 <- BasicTransformerBlock(width = self$width, num_heads = self$num_heads, 
                                            dropout = self$dropout, name = "TextEncoder_1")
    self$encoder_2 <- BasicTransformerBlock(width = self$width, num_heads = self$num_heads, 
                                            dropout = self$dropout, name = "TextEncoder_2")
  },
  
  call = function(input, training = NULL){
    x <- input %>% 
      self$token_embedding() %>% 
      self$encoder_1() %>% 
      self$encoder_2() 
    cls_token <- tf$gather(x, indices = 0L, axis = 1L) # get the first [cls] token
    return(cls_token)
  },
  
  get_config = function(){
    config <- super$get_config()
    config$num_tokens <- self$num_tokens
    config$max_text_len <- self$max_text_len
    config$width <- self$width
    config$num_heads <- self$num_heads
    config$dropout <- self$dropout
    config
  }
)

DiTBlock <- new_layer_class(
  "DiT_Block",
  # A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
  initialize = function(width, num_heads, mlp_ratio = 4.0, dropout = 0.0, ...){
    super$initialize(...)
    self$width <- as.integer(width)
    self$num_heads <- as.integer(num_heads)
    self$mlp_ratio <- mlp_ratio
    self$dropout <- dropout
  },
  
  build = function(input_shape){
    self$norm_attn <- layer_layer_normalization(epsilon = 1e-5, center = FALSE, scale = FALSE, name = "norm_attn_DiT")
    self$attn <- layer_multi_head_attention(num_heads = self$num_heads, 
                                            key_dim = self$width %/% self$num_heads,
                                            dropout = self$dropout,
                                            name = "multi_attn")
    
    self$norm_ffn <- layer_layer_normalization(epsilon = 1e-5, center = FALSE, scale = FALSE, name = "norm_ffn_DiT")
    self$ffn1 <- layer_dense(units = self$width * self$mlp_ratio, name = "ffn1")
    self$dropout <- layer_dropout(rate = self$dropout, name = "dropout")
    self$ffn2 <- layer_dense(units = self$width, name = "ffn2")
    
    self$adaLN_modulation <- layer_dense(units =  6 * self$width, name = "adaLN_modulation", kernel_initializer = "zeros")
  },
  
  call = function(inputs, training = NULL){
    c(x, c) %<-% inputs
    c(shift_msa, scale_msa, 
      gate_msa, shift_mlp, 
      scale_mlp, gate_mlp) %<-% tf$split(c %>% 
                                           layer_activation(activation = "swish") %>% 
                                           self$adaLN_modulation(), 
                                         num_or_size_splits = 6L, axis = -1L)
    
    # Attention layer
    x <- x %>% 
      self$norm_attn() %>% 
      {. * (1.0 + tf$expand_dims(scale_msa, 1L)) + tf$expand_dims(shift_msa, 1L)} %>% # modulate
      self$attn(., .) %>% 
      {. * tf$expand_dims(gate_msa, 1L)} %>% 
      layer_add(., x)
    
    # Feed forward layer
    x <- x %>% 
      self$norm_ffn() %>% 
      {. * (1.0 + tf$expand_dims(scale_mlp, 1L)) + tf$expand_dims(shift_mlp, 1L)} %>% # modulate
      self$ffn1() %>% 
      layer_activation(activation = "gelu") %>% 
      self$dropout(training = training) %>% 
      self$ffn2() %>% 
      {. * tf$expand_dims(gate_msa, 1L)} %>% 
      layer_add(., x)
    
    return(x)
  },
  
  get_config = function(){
    config <- super$get_config()
    config$width <- self$width
    config$num_heads <- self$num_heads
    config$mlp_ratio <- self$mlp_ratio
    config$dropout <- self$dropout
    config
  }
)

FinalLayer <- new_layer_class(
  "Final_Layer",
  # The final layer of DiT.
  initialize = function(width, patch_size, out_channels, ...) {
    super$initialize(...)
    self$width <- as.integer(width)
    self$patch_size <- as.integer(patch_size)
    self$out_channels <- as.integer(out_channels)
  },
  
  build = function(input_shape){
    self$norm <- layer_layer_normalization(epsilon = 1e-5, center = FALSE, scale = FALSE, name = "norm")
    self$linear <- layer_dense(units = self$patch_size * self$patch_size * self$out_channels, kernel_initializer = "zeros")
    self$adaLN_modulation <- layer_dense(units =  2 * self$width, name = "adaLN_modulation", kernel_initializer = "zeros")
  },
  
  call = function(inputs) {
    c(x, c) %<-% inputs
    c(shift, scale) %<-% tf$split(c %>% 
                                    layer_activation(activation = "swish") %>% 
                                    self$adaLN_modulation(),
                                  num_or_size_splits = 2L, axis = -1L)
    x <- x %>% 
      self$norm() %>% 
      {. * (1.0 + tf$expand_dims(scale, 1L)) + tf$expand_dims(shift, 1L)} %>%  # modulate
      self$linear()
    
    return(x)
  },
  
  get_config = function(){
    config <- super$get_config()
    config$width <- self$width
    config$patch_size <- self$patch_size
    config$out_channels <- self$out_channels
    config
  }
)

Unpatchify <- new_layer_class(
  "Unpatchify_Layer",
  # input: (batch_size, num_patches, patch_size * patch_size * out_channels)
  initialize = function(patch_size, ...) {
    super$initialize(...)
    self$patch_size <- as.integer(patch_size)
  },
  
  call = function(input) {
    num_patches <- input$shape[[2]]
    out_channels <- as.integer(input$shape[[3]] / self$patch_size ^ 2)
    width <- as.integer(num_patches ^ 0.5)
    height <- as.integer(num_patches ^ 0.5)
    
    x <- input %>% 
      tf$reshape(shape = c(-1L, num_patches, patch_size, patch_size, out_channels)) %>%
      tf$reshape(shape = c(-1L, height, width, patch_size, patch_size, out_channels)) %>% 
      # (batch_size, height, patch_size, width, patch_size, out_channels)
      tf$einsum("bhwpqc->bhpwqc", .) %>% 
      # (batch_size, img_height, img_width, out_channels)
      tf$reshape(shape = c(-1L, height * patch_size, width * patch_size, out_channels))
    
    return(x)
  },
  
  get_config = function(){
    config <- super$get_config()
    config$patch_size <- self$patch_size
    config
  }
)

get_DiT <- function(){
  noisy_images <- layer_input(shape = c(latent_size, latent_size, 4L), name = "noisy_images")
  timestep_input <- layer_input(shape = c(1L, 1L, 1L), name = "timestep_input")
  condition_input <- layer_input(shape = c(max_text_len + 1),name = "condition_input") # [cls] + max_text_len
  
  x <- noisy_images %>% 
    patch_position_embedding(patch_size = patch_size, width = width, name = "emb_x")
  
  emb_t <- timestep_input %>% 
    timestep_embedding(embedding_dims = embedding_dims, width = width, name = "emb_t")
  
  emb_c <- condition_input %>% 
    text_embedding(num_tokens = num_tokens, max_text_len = max_text_len, 
                   width = width, num_heads = num_heads, 
                   dropout = dropout_rate, name = "emb_c")
  
  emb_y <- layer_add(list(emb_t, emb_c)) 
  
  for (i in 1:num_layers){
    x <- DiTBlock(list(x, emb_y), width = width, num_heads = num_heads, 
                  mlp_ratio = mlp_ratio, dropout = dropout_rate,
                  name = paste0("DiTBlock_", i))
  }
  
  output <- list(x, emb_y) %>% 
    FinalLayer(width = width, patch_size = patch_size, out_channels = out_channels, name = "FinalLayer") %>% 
    Unpatchify(patch_size = patch_size, dtype = "float32", name = "Unpatchify")
  
  DiT <- keras_model(inputs = list(noisy_images, timestep_input, condition_input), outputs = output, name = "DiT")
  return(DiT)
}
