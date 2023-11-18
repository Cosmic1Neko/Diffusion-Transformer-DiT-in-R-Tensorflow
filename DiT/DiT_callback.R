gan_monitor <- new_callback_class(
  "gan_monitor",
  initialize = function(save_epochs,plot_diffusion_steps,plot) {
    super$initialize()
    self$save_epochs <- save_epochs
    self$plot_diffusion_steps <- plot_diffusion_steps
    self$plot <- plot
    if (!fs::dir_exists("gen_images(DiT)")) fs::dir_create("gen_images(DiT)")
    if (!fs::dir_exists("saved_model(DiT)")) fs::dir_create("saved_model(DiT)")
  },
  on_epoch_end = function(epoch, logs = NULL) {
    if(self$plot){
      # conditional generate images
      generated_images <- model$generate(num_images = 25L, 
                                         diffusion_steps = self$plot_diffusion_steps,
                                         condition = "black hair, long hair, smile, blush, bare shoulders, bangs")
      generated_images <- as.array(generated_images)
      png(paste0("gen_images(DiT)/gen_image_cond(Epoch ",epoch + 1, ").png"))
      par(mfrow=c(5,5))
      for(i in 1:25){
        EBImage::Image(generated_images[i,,,],colormode = "Color") %>% 
          EBImage::transpose() %>% 
          EBImage::normalize() %>% 
          plot(margin = 1)
      }
      dev.off()
      
      # unconditional generate images
      generated_images <- model$generate(num_images = 25L, 
                                         diffusion_steps = self$plot_diffusion_steps,
                                         condition = "black hair, long hair, smile, blush, bare shoulders, bangs",
                                         CFG_scale = 0.0)
      generated_images <- as.array(generated_images)
      png(paste0("gen_images(DiT)/gen_image_uncond(Epoch ",epoch + 1, ").png"))
      par(mfrow=c(5,5))
      for(i in 1:25){
        EBImage::Image(generated_images[i,,,],colormode = "Color") %>% 
          EBImage::transpose() %>% 
          EBImage::normalize() %>% 
          plot(margin = 1)
      }
      dev.off()
    }
    
    #save model
    if((epoch + 1) %% self$save_epochs == 0){
      save_model_tf(model$EMA_DiT,
                    file = paste0("saved_model(DiT)/ema_DiT(Epoch ",epoch + 1,")"))
    }
  }
)
