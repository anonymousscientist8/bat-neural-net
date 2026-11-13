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
dir <- "D://for_gabby"

d <- tibble(
  file_name = list.files(dir, pattern = "\\.wav$", full.names = FALSE)
) %>%
  mutate(
    call_number = str_extract(file_name, "^[^_]+"),
    call_type   = str_extract(file_name, "(?<=_)[^.]+")
  )

# Encode call types
d <- d[d$call_type != "Unsure",]
call_map <- c(
  "Echo" = 1, "SFM" = 2, "sHFM" = 3, "LDFM" = 4, "FMB" = 5, "QCF" = 6,      
  "DFM" = 7, "LFM" = 8, "CS" = 9, "LsDFM-LFM" = 10, "LQCF-CS" = 11
)
d$call_type <- as.numeric(call_map[d$call_type])

# Get sample rate
sr <- readWave("D:\\for_gabby\\0_Echo.wav")@samp.rate
fs <- sr

# File paths
paths <- paste0(
  "D:/for_gabby/",
  d$file_name
)

# Audio loader: extract only the annotated segment
get_wave <- function(fname) {
  readWave(fname)@left / 32767
}
waves <- map(paths, get_wave)


# Get lengths
lengths <- sapply(waves, length)
max_samples <- 6400 # Set max we are look at to half second calls

keep <- lengths <= max(lengths) # Currently set to keep all calls
#keep <- lengths <= max_samples # Currently set to restrict calls we look at
d     <- d[keep, ]
waves <- waves[keep]
waves_ds <- waves

# # Normalize the amplitude by call
for (i in 1:length(d$file_name)) {
  mu <- mean(waves_ds[[i]], na.rm = TRUE)
  sigma <- sd(waves_ds[[i]], na.rm = TRUE)
  waves_ds[[i]] <- (waves_ds[[i]] - mu) / sigma
}

# Stratified 80/20 split
train_idx <- unlist(
  lapply(unique(d$call_type), function(k) {
    idx <- which(d$call_type == k)
    sample(idx, size = floor(0.7 * length(idx)))
  })
)
test_idx <- setdiff(seq_len(nrow(d)), train_idx)


# Prepare waveform
max_len <- floor(max_samples)
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

y_train <- d$call_type[train_idx] - 1
y_test  <- d$call_type[test_idx] - 1

num_classes <- 11#length(unique(c(y_train, y_test)))
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
  layer_conv_1d(32, 64, activation = "gelu") %>%
  layer_max_pooling_1d(8) %>%
  layer_gru(32)
  # layer_global_max_pooling_1d()

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
  epochs = 25,
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
