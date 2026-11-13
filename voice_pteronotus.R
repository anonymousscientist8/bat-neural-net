rm(list = ls())
gc()

# Load packages
library(reticulate)
library(keras3)
library(signal)
library(tidyverse)
library(tuneR)
library(caret)
library(uwot)
library(RColorBrewer)
library(reticulate)
library(tensorflow)

# Folder containing the CSV and WAV files
folder <- "C:\\Users\\crave\\Downloads\\AudioBattyExports\\AudioBattyExports"

# Get all CSV files
csv_files <- list.files(
  path = folder,
  pattern = "\\.csv$",
  full.names = TRUE
)

# Read each CSV and add the associated WAV filename
data_list <- lapply(csv_files, function(csv_file) {
  
  # Read annotation table
  x <- read.csv(csv_file, header = TRUE)
  
  # Get the filename without the .csv extension
  base_name <- tools::file_path_sans_ext(basename(csv_file))
  
  # Add associated WAV filename
  x$Recording <- paste0(base_name, ".wav")
  
  return(x)
})

# Combine all annotation tables
d <- do.call(rbind, data_list)

# Encode call types
d <- d[d$Label != "Noise",]
d <- d[d$Label != "Unknown",]
d <- d[d$Label %in% c("DC", "CC", "LHQCFC", "LQCFC"),]
call_map <- c(
  "DC" = 1, "CC" = 2, "LHQCFC" = 3, "LQCFC" = 3
)
d$Label <- as.numeric(call_map[d$Label])
length(unique(d$Label))

# Get sample rate
sr <- readWave("C:\\Users\\crave\\Downloads\\ANGIE2_20230426_104356.wav")@samp.rate
fs <- sr

# File paths
paths <- paste0(
  "C:/Users/crave/Downloads/pterwavs/pterwavs/",
  d$Recording
)

# Audio loader: extract only the annotated segment
get_wave <- function(fname, on, off, fs) {
  wave <- readWave(fname, from = on, to = off, units = "seconds")
  wave@left
}

# Extract waveform for each annotation
waves <- map2(
  paths,
  seq_len(nrow(d)),
  ~ get_wave(.x, d$Onset..s.[.y], d$Offset..s.[.y], fs)
)

# Get lengths
lengths <- sapply(waves, length)
max_samples <- 5000

keep <- lengths <= max(lengths) # Currently set to keep all calls
#keep <- lengths <= max_samples # Currently set to restrict calls we look at
d     <- d[keep, ]
waves <- waves[keep]
waves_ds <- waves

# # Normalize the amplitude by call
for (i in 1:length(d$Recording)) {
  mu <- mean(waves_ds[[i]], na.rm = TRUE)
  sigma <- sd(waves_ds[[i]], na.rm = TRUE)
  waves_ds[[i]] <- (waves_ds[[i]] - mu) / sigma
}

# Stratified 80/20 split
# train_idx <- unlist(
#   lapply(unique(d$CallType), function(k) {
#     idx <- which(d$CallType == k)
#     sample(idx, size = floor(0.8 * length(idx)))
#   })
# )
# test_idx <- setdiff(seq_len(nrow(d)), train_idx)

# Get unique recording files
recordings <- unique(d$Recording)

# Randomly select 80% of recordings for training
train_files <- sample(
  recordings,
  size = floor(0.80 * length(recordings))
)

# Get row indices
train_idx <- which(d$Recording %in% train_files)
test_idx  <- which(!d$Recording %in% train_files)


# Prepare waveform
max_len <- floor(5000)
waveforms <- matrix(0, length(waves_ds), max_len) # Zero pad
for (i in seq_along(waves_ds)) { # Fill in with sound
  waveforms[i, 1:min(max_len, length(waves_ds[[i]]))] <- head(waves_ds[[i]], max_len)
}

# Train / Test split
x_train_wave <- array_reshape(
  waveforms[train_idx, ],
  c(length(train_idx), max_len, 1)
)

x_test_wave <- array_reshape(
  waveforms[test_idx, ],
  c(length(test_idx), max_len, 1)
)

y_train <- d$Label[train_idx] - 1
y_test  <- d$Label[test_idx] - 1

num_classes <- 3#length(unique(c(y_train, y_test)))
y_train <- to_categorical(y_train, num_classes)
y_test  <- to_categorical(y_test, num_classes)

# Class weights
counts <- as.numeric(table(apply(y_train, 1, which.max) - 1))
weights <- max(counts)/counts
class_weights <- as.list(weights)
names(class_weights) <- 0:(length(weights)-1)

# =========================
# CNN model (2D)
# =========================
# Waveform branch
wave_input <- layer_input(shape = c(max_len, 1), name = "wave_input")
wave_branch <- wave_input %>%
  layer_conv_1d(8, 16, activation = "gelu") %>%
  layer_max_pooling_1d(8) %>%
  layer_conv_1d(16, 32, activation = "gelu") %>%
  layer_max_pooling_1d(8) %>%
  # layer_conv_1d(32, 64, activation = "gelu") %>%
  # layer_max_pooling_1d(8) %>%
  layer_gru(32)

# Combine
combined <- layer_concatenate(list(wave_branch)) %>%#, spec_branch)) %>%
  layer_dense(32, activation = "gelu") %>%
  #layer_dropout(0.3) %>%
  layer_dense(num_classes, activation = "softmax")

#model <- keras_model(inputs = list(wave_input, spec_input), outputs = combined)
model <- keras_model(inputs = wave_input, outputs = combined)

model %>% compile(
  optimizer = optimizer_adam(1e-3),
  loss = "categorical_crossentropy",
  metrics = "accuracy"
)

summary(model)

# =========================
# Train

history <- model %>% fit(
  x = list(x_train_wave),#, x_train_spec),
  y = y_train,
  batch_size = 16,
  epochs = 10,
  #class_weight = class_weights,
  validation_split = 0
)

# =========================
# Evaluate
# =========================
y_pred_probs <- model %>% predict(list(x_test_wave))#, x_test_spec))
max_probs <- y_pred_probs[max.col(y_pred_probs)]
y_pred <- apply(y_pred_probs, 1, which.max) - 1
y_true <- apply(y_test, 1, which.max) - 1
mean(y_pred == y_true)
mean(apply(y_pred_probs, 1, max) > 0.25)
mean(y_pred[apply(y_pred_probs, 1, max) > 0.25] == y_true[apply(y_pred_probs, 1, max) > 0.25])
mean(apply(y_pred_probs, 1, max) > 0.5)
mean(y_pred[apply(y_pred_probs, 1, max) > 0.50] == y_true[apply(y_pred_probs, 1, max) > 0.50])
mean(apply(y_pred_probs, 1, max) > 0.75)
mean(y_pred[apply(y_pred_probs, 1, max) > 0.75] == y_true[apply(y_pred_probs, 1, max) > 0.75])

k <- 2
y_true_vec <- apply(y_test, 1, which.max) - 1  # 0-based
top_k_correct <- sapply(1:nrow(y_pred_probs), function(i) {
  top_indices <- order(y_pred_probs[i, ], decreasing = TRUE)[1:k] - 1
  y_true_vec[i] %in% top_indices
})

top_k_accuracy <- mean(top_k_correct)
cat("Top-2 accuracy: ", top_k_accuracy, "\n")

confusionMatrix(factor(y_pred), factor(y_true))
