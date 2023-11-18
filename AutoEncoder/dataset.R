load_image <- function(img_path) {
  img <-  img_path %>% 
    tf$io$read_file() %>% 
    tf$image$decode_jpeg(channels = 3) %>% 
    tf$image$resize(size = c(image_size,image_size),antialias = T) %>% 
    layer_rescaling(scale = 1.0/127.5, offset = -1.0, dtype = "float32")
  return(tf$clip_by_value(img,-1.0,1.0))
}

prepare_dataset <- function(file_path,validation_split,seed,subset){
  image_count <- list.files(file_path) %>% length()
  list_ds <- tf$data$Dataset$list_files(paste0(file_path,"/*.jpg"),shuffle = T, seed = seed)
  print(paste0("Total Number of Images: ",image_count))
  val_size <- as.integer(image_count * validation_split)
  if(subset == "training"){
    return(list_ds %>% dataset_skip(val_size) %>% 
             dataset_map(load_image,num_parallel_calls=tf$data$AUTOTUNE) %>% 
             dataset_shuffle(buffer_size = 1024L) %>% 
             dataset_batch(batch_size) %>% 
             dataset_prefetch(buffer_size = tf$data$AUTOTUNE))
  } else if(subset == "validation"){
    return(list_ds %>% dataset_take(val_size) %>% 
             dataset_map(load_image,num_parallel_calls=tf$data$AUTOTUNE) %>%
             dataset_shuffle(buffer_size = 1024L) %>% 
             dataset_batch(batch_size) %>% 
             dataset_prefetch(buffer_size = tf$data$AUTOTUNE))
  }
}