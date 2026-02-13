rm(list = ls())
gc()

library(keras)
library(signal)
library(tidyverse)
library(tuneR)
library(caret)

d <- read.csv("C:/Users/Raven/AnnotationTable2.csv", header = TRUE)
unique(d$Caller)

d <- d %>%
  mutate(Session = cumsum(
    Receiver != lag(Receiver, default = first(Receiver))
  ))

FOCAL_CALLER <- "Yikes"

# Keep only focal caller
d <- d[d$Caller == FOCAL_CALLER,]
d <- d[d$Receiver != "Lurch",]
unique(d$Receiver)
d <- d[d$Receiver %in% c("Beast", "Quark"),]

splits <- d %>%
  group_by(Caller, Receiver, Session) %>%
  summarise(session_size = n(), .groups = "drop") %>%   # number of rows per session
  group_by(Caller, Receiver) %>%
  filter(n() > 1) %>%                                   # only dyads with >1 session
  mutate(
    test_session = Session[which.min(session_size)],    # pick smallest session
    split = if_else(Session == test_session, "test", "train")
  ) %>%
  select(-test_session, -session_size)

# Join back to row-level data
d_split <- d %>%
  inner_join(splits, by = c("Caller", "Receiver", "Session"))

train_idx <- which(d_split$split == "train")
test_idx  <- which(d_split$split == "test")

# =========================
# Encode CR
# =========================
#call_map <- c("Beast" = 1, "Quark" = 2, "Creep" = 3, "Kazul" = 4, "Mochi" = 5, "Alana" = 6)
call_map <- c("Beast" = 1, "Quark" = 2)

d_split$Receiver <- as.numeric(call_map[d_split$Receiver])

#=========================
# File paths
# =========================
paths <- paste0(
  "C:/Users/Raven/Vampire_audio/",
  d_split$Filename, ".wav"
)

# =========================
# Audio loader
# =========================
get_wave <- function(fname) {
  readWave(fname)@left / 32768
}

waves <- map(paths, get_wave)

lengths <- sapply(waves, length)

max_samples <- max(lengths)

# Downsample
waves_ds <- lapply(waves, function(x) decimate(x, 4))
sr <- readWave(paths[1])@samp.rate
fs <- sr / 4

# =========================
# Spectrogram parameters
# =========================
win_length <- 64
hop_length <- 16
n_fft <- 128

max_time_bins <- 256   # time dimension
max_freq_bins <- n_fft / 2 + 1

# =========================
# Spectrogram function
# =========================
make_spectrogram <- function(x) {
  
  spec <- specgram(
    x,
    n = n_fft,
    Fs = fs,
    window = hamming(win_length),
    overlap = win_length - hop_length
  )
  
  S <- spec$S
  if (is.null(dim(S))) {
    S <- matrix(S, ncol = 1)
  }
  
  S <- abs(S)
  
  # Protect against zeros / invalid values
  S[!is.finite(S)] <- 0
  
  S <- log10(S + 1e-8)
  
  S[!is.finite(S)] <- 0
  
  S_fixed <- matrix(0, max_freq_bins, max_time_bins)
  
  f <- min(nrow(S), max_freq_bins)
  t <- min(ncol(S), max_time_bins)
  
  t_start <- floor((max_time_bins - t) / 2) + 1
  S_fixed[1:f, t_start:(t_start + t - 1)] <- S[1:f, 1:t]
  
  # ----- SAFE normalization -----
  mu <- mean(S_fixed, na.rm = TRUE)
  sigma <- sd(S_fixed, na.rm = TRUE)
  
  if (!is.na(sigma) && sigma > 0) {
    S_fixed <- (S_fixed - mu) / sigma
  } else {
    S_fixed[,] <- 0
  }
  
  return(S_fixed)
}

# =========================
# Build spectrogram array
# =========================
specs <- array(
  0,
  dim = c(length(waves_ds), max_freq_bins, max_time_bins, 1)
)

for (i in seq_along(waves_ds)) {
  specs[i, , , 1] <- make_spectrogram(waves_ds[[i]])
}

# =========================
# Prepare waveform array
# =========================
max_len <- floor(fs * 0.5) # 1 second max
waveforms <- matrix(0, length(waves_ds), max_len)
for (i in seq_along(waves_ds)) {
  waveforms[i, 1:min(max_len, length(waves_ds[[i]]))] <- head(waves_ds[[i]], max_len)
}

# =========================
# Train / Test split
# =========================
x_train_wave <- array_reshape(
  waveforms[train_idx, ],
  c(length(train_idx), max_len, 1)
)

x_test_wave <- array_reshape(
  waveforms[test_idx, ],
  c(length(test_idx), max_len, 1)
)

x_train_spec <- specs[train_idx, , , ]
x_test_spec  <- specs[test_idx,  , , ]

y_train <- d_split$Receiver[train_idx] - 1
y_test  <- d_split$Receiver[test_idx] - 1

num_classes <- 6
y_train <- to_categorical(y_train, num_classes)
y_test  <- to_categorical(y_test, num_classes)

# # Class weights
counts <- as.numeric(table(apply(y_train, 1, which.max) - 1))
#weights <- sqrt(max(counts)/counts)
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
  layer_gru(32)

# Spectrogram branch
spec_input <- layer_input(shape = c(max_freq_bins, max_time_bins, 1), name = "spec_input")
spec_branch <- spec_input %>%
  layer_conv_2d(8, c(8,8), activation = "gelu") %>%
  layer_max_pooling_2d(c(2,2)) %>%
  layer_conv_2d(16, c(16,16), activation = "gelu") %>%
  layer_global_average_pooling_2d()

# Combine
combined <- layer_concatenate(list(wave_branch, spec_branch)) %>%
  layer_dense(32, activation = "gelu") %>%
  layer_dropout(0.3) %>%
  layer_dense(num_classes, activation = "softmax")

model <- keras_model(inputs = list(wave_input, spec_input), outputs = combined)

model %>% compile(
  optimizer = optimizer_adam(1e-3),
  loss = "categorical_crossentropy",
  metrics = "accuracy"
)

# =========================
# Train
# =========================
history <- model %>% fit(
  x = list(x_train_wave, x_train_spec),
  y = y_train,
  batch_size = 16,
  epochs = 10,
  class_weight = class_weights,
  validation_split = 0
)

# =========================
# Evaluate
# =========================
y_pred_probs <- model %>% predict(list(x_test_wave, x_test_spec))
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

