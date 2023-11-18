gan_monitor <- new_callback_class(
  "gan_monitor",
  initialize = function(save_epochs, plot) {
    super$initialize()
    self$save_epochs <- save_epochs
    self$plot <- plot
    if (!fs::dir_exists("gen_images(KLVAE)")) fs::dir_create("gen_images(KLVAE)")
    if (!fs::dir_exists("saved_model(KLVAE)")) fs::dir_create("saved_model(KLVAE)")
  },
  on_epoch_end = function(epoch, logs = NULL) {
    #generate images
    if(self$plot){
      original_images <- test_dataset %>% 
        reticulate::as_iterator() %>% 
        reticulate::iter_next()
      original_images <- original_images[1:8,,,]
      reconstructed_images <- model$EMA_VAE(original_images, training = FALSE)
      original_images <- as.array(original_images)
      reconstructed_images <- as.array(reconstructed_images)
      images <- EBImage::abind(original_images, reconstructed_images, along = 1)
      png(paste0("gen_images(KLVAE)/gen_image(Epoch ",epoch, ").png"))
      par(mfrow=c(4,4))
      for(i in 1:16){
        EBImage::Image(images[i,,,],colormode = "Color") %>% 
          EBImage::transpose() %>% 
          EBImage::normalize() %>% 
          plot(margin = 1)
      }
      dev.off()
    }
    
    #save model
    if((epoch + 1) %% self$save_epochs == 0){
      save_model_tf(model$EMA_VAE,
                    file = paste0("saved_model(KLVAE)/ema_vae(Epoch ",epoch + 1,")"))
    }
  }
)