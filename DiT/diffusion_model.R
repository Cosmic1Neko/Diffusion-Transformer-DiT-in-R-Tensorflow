# schedule only output alpha_cumprod (alpha_hat), because DDIM don't need beta or alpha to sample
linear_schedule <- function(beta_start = 0.00085, beta_end = 0.0120, timesteps = 1000L){
  # linear beta schedule in "DDPM", sqrt(alpha_cumprod) ~ [0.9995749, 0.03973617]
  scale <- 1000 / timesteps
  betas <- tf$linspace(scale * beta_start, scale * beta_end, timesteps)
  
  alphas <- 1.0 - betas
  alphas_cumprod <- tf$math$cumprod(alphas, axis = 0L)
  return(list(alphas_cumprod, betas))
}

cosine_schedule <- function(alphas_cumprod_start = 0.999, alphas_cumprod_end = 0.001, timesteps = 1000L){
  # cosine alphas_cumprod(alphas_hat) schedule in "improved DDPM"
  # I set the minimum alphas_cumprod to avoid the extreme value when sampling (1 / sqrt(alphas_cumprod_999) = 1 / 0.03162278)
  # sqrt(alpha_cumprod) ~ [0.99945104, 0.03162471]
  start_angle <- tf$acos(alphas_cumprod_start ^ 0.5)
  end_angle <- tf$acos(alphas_cumprod_end ^ 0.5)
  x <- tf$linspace(1, timesteps, timesteps)
  diffusion_angles <- start_angle + x / timesteps * (end_angle - start_angle)
  alphas_cumprod <- tf$cos(diffusion_angles) ^ 2
  
  tmp_alphas_cumprod <- tf$concat(list(tf$constant(1.0, shape = c(1L)), alphas_cumprod), axis = 0L)
  betas <- 1.0 - (tmp_alphas_cumprod[2:length(tmp_alphas_cumprod)] / tmp_alphas_cumprod[1:(length(tmp_alphas_cumprod) - 1)])
  return(list(alphas_cumprod, betas))
}

DiffusionModel <- new_model_class(
  "DiffusionModel",
  initialize = function(DiT, EMA_DiT, Encoder, Decoder){
    super$initialize()
    self$DiT <- DiT
    self$EMA_DiT <- EMA_DiT
    if(!is.null(self$DiT)){self$EMA_DiT$set_weights(self$DiT$get_weights())}
    self$Encoder <- Encoder
    self$Decoder <- Decoder
    
    self$text_vectorization <- layer_text_vectorization(standardize = NULL,
                                                        output_sequence_length = max_text_len + 1L, # [cls]
                                                        split = function(x){tf$strings$split(x, sep = ",")},
                                                        output_mode = "int",
                                                        vocabulary = token_vocabulary,
                                                        name = "text_vectorization")
    
    ###########################################################################
    ######                      diffusion parameter                      ######
    ###########################################################################
    # define alphas_cumprod (alphas_hat)
    if (diffusion_schedule == "linear"){
      c(self$alphas_cumprod, self$betas) %<-% linear_schedule(beta_start = beta_start, 
                                                              beta_end = beta_end, 
                                                              timesteps = timesteps)
    } else if (diffusion_schedule == "cosine"){
      c(self$alphas_cumprod, self$betas) %<-% cosine_schedule(alphas_cumprod_start = alphas_cumprod_start, 
                                                              alphas_cumprod_end = alphas_cumprod_end, 
                                                              timesteps = timesteps)
    }
    self$alphas_cumprod_prev <- tf$concat(list(tf$constant(1.0, shape = c(1L)),
                                               self$alphas_cumprod[1:(length(self$alphas_cumprod) - 1)]), axis = 0L)
    self$alphas_cumprod_next <- tf$concat(list(self$alphas_cumprod[2:length(self$alphas_cumprod)],
                                               tf$constant(0.0, shape = c(1L))), axis = 0L)
    
    # calculations for diffusion q(x_t | x_{t-1}) and others
    self$sqrt_alphas_cumprod <- tf$sqrt(self$alphas_cumprod)
    self$sqrt_one_minus_alphas_cumprod <- tf$sqrt(1.0 - self$alphas_cumprod)
    self$log_one_minus_alphas_cumprod <- tf$math$log(1.0 - self$alphas_cumprod)
    self$sqrt_recip_alphas_cumprod <- tf$sqrt(1.0 / self$alphas_cumprod)
    self$sqrt_recipm1_alphas_cumprod <- tf$sqrt(1.0 / self$alphas_cumprod - 1.0)
    
    # calculations for posterior q(x_{t-1} | x_t, x_0)
    self$posterior_variance <- (self$betas * (1.0 - self$alphas_cumprod_prev) / (1.0 - self$alphas_cumprod))
    # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
    self$posterior_log_variance_clipped <- tf$math$log(
      tf$concat(list(
        tf$constant(self$posterior_variance[2], shape = c(1L)),
        self$posterior_variance[2:length(self$posterior_variance)]
        ), axis = 0L),
      )
    
    self$alphas <- 1.0 - self$betas
    self$posterior_mean_coef1 <- (self$betas * tf$sqrt(self$alphas_cumprod_prev) / (1.0 - self$alphas_cumprod))
    self$posterior_mean_coef2 <- ((1.0 - self$alphas_cumprod_prev) * tf$sqrt(self$alphas) / (1.0 - self$alphas_cumprod))
  },
  
  compile = function(...){
    super$compile(...)
    self$noise_loss_tracker <- keras$metrics$Mean(name="n_loss")
  },
  
  metrics = mark_active(function() {
    list(self$noise_loss_tracker)
  }),
  
  to_latents = function(images){
    latents <- self$Encoder(images, training = FALSE) * scale_factor
    return(latents)
  },
  
  to_images = function(latents){
    latents <- latents * 1.0 / scale_factor
    images <- self$Decoder(latents, training = FALSE)
    return(images)
  },
  
  extract = function(x, diffusion_times){
    # get the value according to t from the previously defined α_t list.
    return(tf$gather(x, tf$cast(diffusion_times, "int32")))
  },
  
  forward_diffusion = function(latents){
    batch_size <- tf$shape(latents)[1]
    # generate noises and offset noises (https://www.crosslabs.org/blog/diffusion-with-offset-noise)
    noises <- tf$random$normal(shape = c(batch_size, latent_size, latent_size, 4L))
    if (noise_offset != 0){
      noises <- noises + noise_offset * tf$random$normal(shape = c(batch_size, 1L, 1L, 4L))
    }
    
    # generate random diffusion times
    diffusion_times <- tf$random$uniform(shape=c(batch_size, 1L, 1L, 1L), minval=0L, maxval=timesteps, dtype = "int32")
    
    # signal_rates is sqrt(α_t) and noise_rates is sqrt(1 - α_t) in DDIM
    signal_rates <- self$extract(self$sqrt_alphas_cumprod, diffusion_times)
    noise_rates <- self$extract(self$sqrt_one_minus_alphas_cumprod, diffusion_times)
    
    # mix the images with noises accordingly
    noisy_latents <- signal_rates * latents + noise_rates * noises
    
    return(list(noisy_latents, 
                diffusion_times,
                noises))
  },
  
  reverse_process = function(start_x,
                             start_t = timesteps, # used for img2img
                             diffusion_steps,
                             eta = 0.0,
                             prompt,
                             negative_prompt = NULL,
                             CFG_scale = 4.0){
    # reverse diffusion = sampling 
    num_images <- start_x$shape[[1]]
    diffusion_steps <- as.integer(diffusion_steps)
    
    # unconditional
    if (is.null(negative_prompt)){
      null_condition <- tf$concat(list(
        tf$constant(2L, shape = c(num_images, 1L), dtype = "int32"),
        tf$constant(0L, shape = c(num_images, max_text_len), dtype = "int32")
      ), axis = 1L)
    } else{
      null_condition <- tf$constant(negative_prompt, shape = c(num_images), dtype = "string") %>% self$text2index(training = FALSE)
    }
    # conditional
    if (is.null(prompt)){
      condition <- tf$concat(list(
        tf$constant(2L, shape = c(num_images, 1L), dtype = "int32"),
        tf$constant(0L, shape = c(num_images, max_text_len), dtype = "int32")
      ), axis = 1L)
    } else{
      condition <- tf$constant(prompt, shape = c(num_images), dtype = "string") %>% self$text2index(training = FALSE)
    }
    
    t_seq <- tf$linspace(start_t - 1L, 0L, diffusion_steps) %>% 
      as.integer()
    
    message(paste0("Sampling with steps ", diffusion_steps, ", eta ", eta, " and start_t ", start_t))
    pb <- progress_bar$new(
      format = "[:bar] :current/:total ETA::eta",
      total = diffusion_steps,
      width = 60
    )
    # at the first sampling step, the "x" is pure noise, diffusion_times is 999 and not 1000
    # in img2img, the "x" is noisy images with the "start_t" steps
    x <- start_x 
    for(i in 1:length(t_seq)){
      diffusion_times <- t_seq[i] # get the t
      diffusion_times_prev <- t_seq[i + 1] # get the next t
      if(is.na(diffusion_times_prev)) {last_step = TRUE} else {last_step = FALSE}
      
      # get t and t_prev
      diffusion_times <- tf$constant(diffusion_times, shape = c(num_images, 1L, 1L, 1L), dtype = "int32")
      diffusion_times_prev <- tf$constant(diffusion_times_prev, shape = c(num_images, 1L, 1L, 1L), dtype = "int32")
      
      # predict one component of the noisy images with the network
      if (use_ema){
        pred_noises_uncond <- self$EMA_DiT(list(x, diffusion_times, null_condition), training = FALSE)
        pred_noises_cond <- self$EMA_DiT(list(x, diffusion_times, condition), training = FALSE)
        pred_noises = pred_noises_uncond + CFG_scale * (pred_noises_cond - pred_noises_uncond)
      } else{
        pred_noises_uncond <- self$DiT(list(x, diffusion_times, null_condition), training = FALSE)
        pred_noises_cond <- self$DiT(list(x, diffusion_times, condition), training = FALSE)
        pred_noises = pred_noises_uncond + CFG_scale * (pred_noises_cond - pred_noises_uncond)
      }
      
      # get alpha_cumprod and alpha_cumprod_prev
      alpha_cumprod <- self$extract(self$alphas_cumprod, diffusion_times)
      # when the last step diffusion_times = 0, diffusion_times_prev = NA, so we set alpha_cumprod_prev = 1.0
      if (!last_step){
        alpha_cumprod_prev <- self$extract(self$alphas_cumprod, diffusion_times_prev)
      } else {
        alpha_cumprod_prev <- tf$constant(1.0, dtype = "float32")
      }
      
      if (all(as.vector(diffusion_times) > 0)){
        noises <- tf$random$normal(shape = x$shape)
      } else {
        noises <- tf$zeros_like(x)
      }
      
      # eta = 0 is DDIM, eta = 1 is fixedsmall variances DDPM
      # 1.0 - alpha_cumprod / alpha_cumprod_prev = 1 - alpha_t = beta_t
      sigma <- eta * tf$sqrt((1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)) * tf$sqrt(1.0 - alpha_cumprod / alpha_cumprod_prev)
      predict_x0 <- (x - tf$sqrt(1.0 - alpha_cumprod) * pred_noises) / tf$sqrt(alpha_cumprod)
      direction_point <- tf$sqrt(1.0 - alpha_cumprod_prev - tf$square(sigma)) * pred_noises
      random_noise <- sigma * noises
      x <- tf$sqrt(alpha_cumprod_prev) * predict_x0 + direction_point + random_noise # x_{t-1}
      
      pb$tick()
    }
    return(x)
  },
  
  generate = function(num_images,
                      diffusion_steps = 20,
                      eta = 0.0,
                      prompt,
                      negative_prompt = NULL,
                      CFG_scale = 4.0){
    # noise -> latent images -> pixel images
    num_images <- as.integer(num_images)
    initial_noise <- tf$random$normal(shape=c(num_images, latent_size, latent_size, 4L))
    latents <- self$reverse_process(start_x = initial_noise,
                                    start_t = timesteps, 
                                    diffusion_steps = diffusion_steps,
                                    eta = eta,
                                    prompt = prompt,
                                    negative_prompt = negative_prompt,
                                    CFG_scale = CFG_scale)
    generated_images <- self$to_images(latents)
    return(generated_images)
  },
  
  img2img = function(images,
                     diffusion_steps = 20,
                     denoising_strength = 0.5,
                     eta = 0.0,
                     prompt,
                     negative_prompt = NULL,
                     CFG_scale = 4.0){
    num_images <- tf$shape(images)[1]
    latents <- self$to_latents(images)
    start_t <- as.integer(denoising_strength * timesteps - 1) %>% # [1,1000] -> [0,999]
      tf$constant(shape = c(num_images, 1L, 1L, 1L))
    
    # forward diffusion
    noises <- tf$random$normal(shape = c(num_images, latent_size, latent_size, 4L))
    signal_rates <- self$extract(self$sqrt_alphas_cumprod, start_t)
    noise_rates <- self$extract(self$sqrt_one_minus_alphas_cumprod, start_t)
    start_x <- signal_rates * latents + noise_rates * noises
    
    # denoising
    latents <- self$reverse_process(start_x = start_x,
                                    start_t = start_t + 1, # [0,999] -> [1,1000]
                                    diffusion_steps = diffusion_steps,
                                    eta = eta,
                                    prompt = prompt,
                                    negative_prompt = negative_prompt,
                                    CFG_scale = CFG_scale)
    
    generated_images <- self$to_images(latents)
    return(generated_images)
  },
  
  text2index = function(text, training){
    batch_size = tf$shape(text)[1]
    # Pre-processing
    text <- text %>% 
      tf$strings$lower() %>%
      tf$strings$regex_replace(", +", ",") # "blush, smile, long hair" -> "blush,smile,long hair"
    
    if (training){
      # 1. replace with null text randomly (for CFG)
      null_ids <- tf$less(tf$random$uniform(shape = c(batch_size, 1L)) %>% 
                            tf$reshape(c(batch_size)), 
                          text_dropout)
      null_text <- tf$where(null_ids, "", text) # replace the dropout text to the "null" text
      
      shuffle_and_dropout <- tf_function(f = function(null_text){
        # 2. shuffle labels ("blush,smile,long hair" -> "long hair,blush,smile")
        shuffle_text <- null_text %>% 
          tf$strings$split(sep = ",") %>% 
          tf$random$shuffle()
        
        # 3. dropout labels ("long hair,blush,smile" -> "long hair,smile")
        drop_ids <- tf$less(tf$random$uniform(shape = c(tf$shape(shuffle_text)[1], 1L)) %>% 
                              tf$reshape(c(tf$shape(shuffle_text)[1])), 
                            0.1) # I tend to set it to a lower value
        shuffle_dropout_text <- tf$where(drop_ids, "", shuffle_text) %>% 
          tf$strings$reduce_join(separator = ",") %>% 
          tf$strings$regex_replace(",+", ",") # "long hair,,smile" -> "long hair,smile"
        return(shuffle_dropout_text)
      })
      output_text <- tf$map_fn(shuffle_and_dropout, null_text)
    } else{
      output_text <- text
    }
    
    condition <- tf$strings$join(c("[cls],", output_text)) %>% 
      self$text_vectorization() %>% # 0 is [null], 1 is [UNK], 2 is [cls]
      tf$cast(dtype = "int32")
    return(condition)
  },
  
  train_step = function(data){
    c(images, text) %<-% data
    batch_size <- tf$shape(images)[1]
    # images augmentation
    images <- tf$image$random_flip_left_right(images)
    # to latent space
    latents <- self$to_latents(images)
    
    # set training = TRUE to :
    # 1. replace with null text randomly (for CFG)
    # 2. shuffle labels ("blush,smile,long hair" -> "long hair,blush,smile")
    # 3. dropout labels ("long hair,blush,smile" -> "long hair,smile")
    # 2. and 3. may be useful for regularization.
    condition <- self$text2index(text, training = TRUE)
    
    # forward diffusion, add noises to the images
    c(noisy_latents, diffusion_times, noises) %<-% self$forward_diffusion(latents)
    
    # use U-Net to predict the added noises
    with(tf$GradientTape() %as% tape, {
      # train the network to separate noisy images to their components
      pred_noises <- self$DiT(list(noisy_latents, diffusion_times, condition), training=TRUE)
      noise_loss <- self$loss(noises, pred_noises)  # used for training
      if (mixed_precision){
        scaled_loss <- self$optimizer$get_scaled_loss(noise_loss)
      }
    })
    if (mixed_precision){
      scaled_gradients <- tape$gradient(scaled_loss, self$DiT$trainable_weights)
      gradients <- self$optimizer$get_unscaled_gradients(scaled_gradients)
    } else{
      gradients <- tape$gradient(noise_loss, self$DiT$trainable_variables)
    }
    self$optimizer$apply_gradients(
      zip_lists(gradients, self$DiT$trainable_variables)
    )
    
    #ema
    for(w in zip_lists(self$DiT$weights,self$EMA_DiT$weights)){
      w[[2]]$assign(ema * w[[2]] + (1 - ema) * w[[1]])
    }
    
    self$noise_loss_tracker$update_state(noise_loss)
    results <- list()
    for (m in self$metrics)
      results[[m$name]] <- m$result()
    results
  },
  
  test_step = function(data){
    c(images, text) %<-% data
    batch_size <- tf$shape(images)[1]
    # to latent space
    latents <- self$to_latents(images)
    
    # set training = TRUE to compare val_loss with train_loss 
    condition <- self$text2index(text, training = TRUE)
    
    # forward diffusion, add noises to the images
    c(noisy_latents, diffusion_times, noises) %<-% self$forward_diffusion(latents)
    
    # use U-Net to predict the added noises
    pred_noises <- self$DiT(list(noisy_latents, diffusion_times, condition), training=FALSE)
    noise_loss <- self$loss(noises, pred_noises)
    
    self$noise_loss_tracker$update_state(noise_loss)
    results <- list()
    for (m in self$metrics)
      results[[m$name]] <- m$result()
    results
  }
)
