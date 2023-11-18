load_image <- function(img_path) {
  img <-  img_path %>% 
    tf$io$read_file() %>% 
    tf$image$decode_jpeg(channels = 3) %>% 
    tf$image$resize(size = c(image_size,image_size),antialias = T) %>% 
    layer_rescaling(scale = 1.0/127.5, offset = -1.0, dtype = "float32")
  return(tf$clip_by_value(img,-1.0,1.0))
}

load_files <- function(img_path, text_path) {
  img <-  img_path %>% 
    tf$io$read_file() %>% 
    tf$image$decode_jpeg(channels = 3) %>% 
    tf$image$resize(size = c(image_size,image_size),antialias = T) %>% 
    layer_rescaling(scale = 1.0/127.5, offset = -1.0, dtype = "float32") %>% 
    tf$clip_by_value(-1.0,1.0)
  
  text <- text_path %>% 
    tf$io$read_file() %>% 
    tf$strings$strip() 
  
  return(list(img, text))
}

prepare_dataset <- function(file_path,validation_split,seed,subset){
  image_path <- paste0(file_path, "/image")
  image_files <- tf$data$Dataset$list_files(paste0(image_path,"/*.jpg"), shuffle = T, seed = seed)
  text_path <- paste0(file_path, "/text")
  text_files <- tf$data$Dataset$list_files(paste0(text_path,"/*.txt"), shuffle = T, seed = seed)
  dataset <- tf$data$Dataset$zip(reticulate::tuple(image_files, text_files))
  
  image_count <- list.files(image_path, pattern = "jpg") %>% length()
  print(paste0("Total Number of Images: ",image_count))
  val_size <- as.integer(image_count * validation_split)
  if(subset == "training"){
    return(dataset %>% dataset_skip(val_size) %>% 
             dataset_map(load_files,num_parallel_calls=tf$data$AUTOTUNE) %>% 
             dataset_shuffle(buffer_size = 1024L) %>% 
             dataset_batch(batch_size) %>% 
             dataset_prefetch(buffer_size = tf$data$AUTOTUNE))
  } else if(subset == "validation"){
    return(dataset %>% dataset_take(val_size) %>% 
             dataset_map(load_files,num_parallel_calls=tf$data$AUTOTUNE) %>%
             dataset_shuffle(buffer_size = 1024L) %>% 
             dataset_batch(batch_size) %>% 
             dataset_prefetch(buffer_size = tf$data$AUTOTUNE))
  }
}

make_vocabulary <- function(file_path){
  text_path <- paste0(file_path, "/text")
  text_files <- list.files(text_path, pattern = ".txt", full.names = TRUE)
  tmp <- c()
  for (i in 1:length(text_files)){
    if (i %% as.integer(length(text_files)/100) == 0){message(i)}
    content <- text_files[i] %>% 
      readLines(warn = TRUE)
    if (length(content) == 0){
      content = NULL
    } else{
      content <- content %>% 
        {strsplit(., split = ", ")[[1]]} %>% 
        unlist()
    }
    tmp <- c(tmp, content)
    tmp <- unique(tmp)
  }
  tokens_list <- sort(tmp)
  tokens_list <- c("[cls]", tokens_list)
}
