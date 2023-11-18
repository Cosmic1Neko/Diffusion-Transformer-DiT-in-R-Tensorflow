KLVAE <- new_model_class(
  "KLVAE",
  initialize = function(VAE, EMA_VAE) {
    super$initialize()
    self$VAE <- VAE
    self$EMA_VAE <- EMA_VAE
    self$EMA_VAE$set_weights(self$VAE$get_weights())
  },
  
  compile = function(optimizer,reco_loss_fn, recp_loss_fn) {
    super()$compile()
    self$optimizer <- optimizer
    self$reco_loss_fn <- reco_loss_fn
    self$recp_loss_fn <- recp_loss_fn
    
    self$reco_loss_tracker <- keras$metrics$Mean(name = "reco_loss")
    self$recp_loss_tracker <- keras$metrics$Mean(name = "recp_loss")
    self$kl_loss_tracker <- keras$metrics$Mean(name = "kl_loss")
  }, 
  
  metrics = mark_active(function(){
    list(self$reco_loss_tracker,
         self$recp_loss_tracker,
         self$kl_loss_tracker)
  }),
  
  train_step = function(images){
    batch_size <- tf$shape(images)[1]
    
    # images augmentation
    images <- tf$image$random_flip_left_right(images)
    
    # Train VAE 
    with(tf$GradientTape() %as% tape, {
      reconstructed_images <- self$VAE(images, training = TRUE)     
      
      # Calculate the loss
      reco_loss <- self$reco_loss_fn(images, reconstructed_images)
      recp_loss <- self$recp_loss_fn(images, reconstructed_images)
      kl_loss <- self$VAE$losses[[1]]
      total_loss <- (10.0 * reco_loss) + (1.0 * recp_loss) + (lambda * kl_loss)
      if (mixed_precision){
        scaled_loss <- self$optimizer$get_scaled_loss(total_loss)
      }
    })
    if (mixed_precision){
      scaled_gradients <- tape$gradient(scaled_loss, self$VAE$trainable_weights)
      gradients <- self$optimizer$get_unscaled_gradients(scaled_gradients)
    } else{
      gradients <- tape$gradient(total_loss, self$VAE$trainable_variables)
    }
    self$optimizer$apply_gradients(
      zip_lists(gradients, self$VAE$trainable_variables)
    )
    
    #ema
    for(w in zip_lists(self$VAE$weights,self$EMA_VAE$weights)){
      w[[2]]$assign(ema * w[[2]] + (1 - ema) * w[[1]])
    }
    
    self$reco_loss_tracker$update_state(reco_loss)
    self$recp_loss_tracker$update_state(recp_loss)
    self$kl_loss_tracker$update_state(kl_loss)
    results <- list()
    for (m in self$metrics)
      results[[m$name]] <- m$result()
    results
  },
  
  test_step = function(images){
    batch_size <- tf$shape(images)[1]
    
    reconstructed_images <- self$VAE(images, training = FALSE)
    
    # Calculate the loss
    reco_loss <- self$reco_loss_fn(images, reconstructed_images)
    recp_loss <- self$recp_loss_fn(images, reconstructed_images)
    kl_loss <- self$VAE$losses[[1]]
    
    self$reco_loss_tracker$update_state(reco_loss)
    self$recp_loss_tracker$update_state(recp_loss)
    self$kl_loss_tracker$update_state(kl_loss)
    results <- list()
    for (m in self$metrics)
      results[[m$name]] <- m$result()
    results
  }
)