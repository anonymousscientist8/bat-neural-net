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

# =========================
# Read annotation table
# =========================
d <- read.csv("C:\\Users\\crave\\Downloads\\AnnotationTable2.csv", header = TRUE)
unique(d$Caller)
d <- d[is.na(d$Caller) == F,]

d <- d %>%
  mutate(Session = cumsum(
    Caller != lag(Caller, default = first(Caller)) |
      Receiver != lag(Receiver, default = first(Receiver))
  ))

d <- d[!duplicated(d$Filename), ]
unique(d$Receiver)

d <- d[d$Caller == "Yikes",]

# =========================
# Stratified 80/20 split
# =========================

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
#d_split <- d_split[sample(nrow(d_split)), ]

d_split <- d_split[d_split$Receiver %in% c("Mochi", "Beast"),]

a <- unique(d_split$Receiver)
a
# for (i in 1:length(a)) {
#   temp1 <- d_split[d_split$Caller != a[i] & d_split$split == "train",]
#   temp2 <- d_split[d_split$Caller == a[i] & d_split$split == "train",]
#   temp3 <- d_split[d_split$split == "test",]
#   temp2 <- temp2[sample(1:length(temp2$Caller), min(length(temp2$Caller), 400)),]
#   d_split <- rbind(temp1, temp2, temp3)
# }

train_idx <- which(d_split$split == "train")
test_idx  <- which(d_split$split == "test")


# Stratified 80/20 split
# train_idx <- unlist(
#   lapply(unique(d$Caller), function(k) {
#     idx <- which(d$Caller == k)
#     sample(idx, size = floor(0.8 * length(idx)))
#   })
# )
# test_idx <- setdiff(seq_len(nrow(d)), train_idx)

# =========================
# Encode caller
# =========================
# call_map <- c("Lurch" = 1,
#               "Beast" = 2,
#               "Mochi" = 3,
#               "Creep" = 4,
#               "Yikes" = 5,
#               "Kazul" = 6,
#               "Alana" = 7)

call_map <- c("Mochi" = 1, "Beast" = 2)

d_split$Receiver <- as.numeric(call_map[d_split$Receiver])

# =========================
# Match annotation filenames to audio files
# =========================

paths <- list.files(
  "C:/Users/crave/Downloads/Vampire_Audio/",
  pattern = "\\.wav$",
  full.names = TRUE,
  recursive = TRUE
)

# Make a lookup table from the actual audio files
paths_df <- tibble(
  path = paths,
  Filename = basename(paths)
)

# Clean filenames on both sides
d2 <- d_split %>%
  mutate(
    Filename_clean = tolower(trimws(basename(Filename))),
    Filename_clean = tools::file_path_sans_ext(Filename_clean)
  )

paths_df <- paths_df %>%
  mutate(
    Filename_clean = tolower(trimws(Filename)),
    Filename_clean = tools::file_path_sans_ext(Filename_clean)
  )

# Match annotations to audio files
d2 <- d2 %>%
  left_join(
    paths_df %>% select(path, Filename_clean),
    by = "Filename_clean"
  )

# Check the result
cat("Total annotation rows:", nrow(d2), "\n")
cat("Rows with matched audio:", sum(!is.na(d2$path)), "\n")
cat("Rows WITHOUT audio:", sum(is.na(d2$path)), "\n")

# Look at unmatched filenames
head(
  d2 %>%
    filter(is.na(path)) %>%
    select(Filename),
  20
)

d_split <- d2

# =========================
# Audio loader
# =========================
get_wave <- function(fname) {
  readWave(fname)@left / 32768
}

waves <- map(d_split$path, get_wave)

lengths <- sapply(waves, length)

max_samples <- max(lengths)

waves_ds <- waves
sr <- readWave(paths[1])@samp.rate
fs <- sr

# =========================
# Prepare waveform array
# =========================
# # Normalize the amplitude by call
for (i in 1:length(d_split$Filename)) {
  mu <- mean(waves_ds[[i]], na.rm = TRUE)
  sigma <- sd(waves_ds[[i]], na.rm = TRUE)
  waves_ds[[i]] <- (waves_ds[[i]] - mu) / sigma
}

target_len <- 10000

time_normalize <- function(x, target_len) {
  
  old_time <- seq(0, 1, length.out = length(x))
  new_time <- seq(0, 1, length.out = target_len)
  
  approx(
    x = old_time,
    y = x,
    xout = new_time,
    method = "linear"
  )$y
}

waveforms <- matrix(
  0,
  nrow = length(waves_ds),
  ncol = target_len
)

for (i in seq_along(waves_ds)) {
  waveforms[i, ] <- time_normalize(
    waves_ds[[i]],
    target_len
  )
}

# =========================
# Train / Test split
# =========================
x_train_wave <- array_reshape(
  waveforms[train_idx, ],
  c(length(train_idx), target_len, 1)
)

x_test_wave <- array_reshape(
  waveforms[test_idx, ],
  c(length(test_idx), target_len, 1)
)

y_train <- d_split$Receiver[train_idx] - 1
y_test  <- d_split$Receiver[test_idx] - 1
# y_train <- d$Caller[train_idx] - 1
# y_test  <- d$Caller[test_idx] - 1

num_classes <- 2
y_train <- to_categorical(y_train, num_classes)
y_test  <- to_categorical(y_test, num_classes)

# # Class weights
counts <- as.numeric(table(apply(y_train, 1, which.max) - 1))
weights <- sqrt(max(counts)/counts)
class_weights <- as.list(weights)
names(class_weights) <- 0:(length(weights)-1)

# =========================
# CNN model (2D)
# =========================
# Waveform branch
wave_input <- layer_input(shape = c(target_len, 1), name = "wave_input")
wave_branch <- wave_input %>%
  layer_conv_1d(4, 8, activation = "gelu") %>%
  layer_max_pooling_1d(8) %>%
  layer_conv_1d(4, 16, activation = "gelu") %>%
  layer_max_pooling_1d(8) %>%
  # layer_conv_1d(32, 64, activation = "gelu") %>%
  # layer_max_pooling_1d(8) %>%
  layer_gru(128)

# Combine
combined <- layer_concatenate(list(wave_branch)) %>%#, spec_branch)) %>%
  layer_dense(32, activation = "gelu") %>%
  #layer_dropout(0.3) %>%
  layer_dense(num_classes, activation = "softmax")

model <- keras_model(inputs = wave_input, outputs = combined)

model %>% compile(
  optimizer = optimizer_adam(1e-3),
  loss = "categorical_crossentropy",
  metrics = "accuracy"
)

summary(model)

# =========================
# Train
# =========================
history <- model %>% fit(
  x = x_train_wave,
  y = y_train,
  batch_size = 16,
  epochs = 100,
  #class_weight = class_weights,
  validation_split = 0
)

# =========================
# Evaluate
# =========================
y_pred_probs <- model %>% predict(list(x_test_wave))
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
