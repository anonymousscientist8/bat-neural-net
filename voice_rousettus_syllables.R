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

# Read annotation table
d <- read.csv("C:\\Users\\crave\\Downloads\\AnnotationTable.csv", header = TRUE)
d <- d[d$CallType != "", ] # Get rid of calls without annotation

# Extract sex, individual IDs, and hearing capability
d$Sex <- substr(d$MicFileName, 1, 1)
d$Individual <- substr(d$MicFileName, 1, 2)
d$Hearing <- substr(d$MicFileName, 4, 4)

d <- d[d$Hearing == "H",]

# Encode call types
d <- d[d$CallType != "Un",]
call_map <- c(
  "BB" = 1, "Pi" = 2, "Ch" = 3, "Sq" = 4,
  "Bu" = 5, "LT" = 6, "Ra" = 7, "Pa" = 8, "Ba" = 9,
  "SB" = 10, "PB" = 11, "Tr" = 12, "LPi" = 13
)
# call_map3 <- c(
#   "M" = 1, "F" = 2
# )
# call_map3 <- c(
#   "D" = 1, "H" = 2
# )
# call_map3 <- c(
#   "M3" = 1, "M4" = 2, "F4" = 3, "F5" = 4, "F6" = 5
# )
# call_map3 <- c(
#   "M3" = 1, "M4" = 2, "F4" = 3, "F5" = 4, "F6" = 5,
#   "F3" = 6, "F1" = 7, "M2" = 8, "F2" = 9, "M1" = 10
# )
d$CallType <- as.numeric(call_map[d$CallType])
unique(d$CallType)

#d <- d[d$Individual == "F6",]
#d <- d[d$CallType == 13,]

# for (i in 1:length(unique(d$CallType))) {
#   if (length(d$CallType[d$CallType == i]) < 200) {
#     d <- d[d$CallType != i,]
#   }
# }

# d1 <- c("M3", length(d$CallType[d$CallType == 1 & d$Individual == "M3"]),
#         length(d$CallType[d$CallType == 2 & d$Individual == "M3"]),
#         length(d$CallType[d$CallType == 5 & d$Individual == "M3"]),
#         length(d$CallType[d$CallType == 6 & d$Individual == "M3"]),
#         length(d$CallType[d$CallType == 8 & d$Individual == "M3"]),
#         length(d$CallType[d$CallType == 13 & d$Individual == "M3"]))
# d2 <- c("M4", length(d$CallType[d$CallType == 1 & d$Individual == "M4"]),
#         length(d$CallType[d$CallType == 2 & d$Individual == "M4"]),
#         length(d$CallType[d$CallType == 5 & d$Individual == "M4"]),
#         length(d$CallType[d$CallType == 6 & d$Individual == "M4"]),
#         length(d$CallType[d$CallType == 8 & d$Individual == "M4"]),
#         length(d$CallType[d$CallType == 13 & d$Individual == "M4"]))
# d3 <- c("F4", length(d$CallType[d$CallType == 1 & d$Individual == "F4"]),
#         length(d$CallType[d$CallType == 2 & d$Individual == "F4"]),
#         length(d$CallType[d$CallType == 5 & d$Individual == "F4"]),
#         length(d$CallType[d$CallType == 6 & d$Individual == "F4"]),
#         length(d$CallType[d$CallType == 8 & d$Individual == "F4"]),
#         length(d$CallType[d$CallType == 13 & d$Individual == "F4"]))
# d4 <- c("F5", length(d$CallType[d$CallType == 1 & d$Individual == "F5"]),
#         length(d$CallType[d$CallType == 2 & d$Individual == "F5"]),
#         length(d$CallType[d$CallType == 5 & d$Individual == "F5"]),
#         length(d$CallType[d$CallType == 6 & d$Individual == "F5"]),
#         length(d$CallType[d$CallType == 8 & d$Individual == "F5"]),
#         length(d$CallType[d$CallType == 13 & d$Individual == "F5"]))
# d5 <- c("F6", length(d$CallType[d$CallType == 1 & d$Individual == "F6"]),
#         length(d$CallType[d$CallType == 2 & d$Individual == "F6"]),
#         length(d$CallType[d$CallType == 5 & d$Individual == "F6"]),
#         length(d$CallType[d$CallType == 6 & d$Individual == "F6"]),
#         length(d$CallType[d$CallType == 8 & d$Individual == "F6"]),
#         length(d$CallType[d$CallType == 13 & d$Individual == "F6"]))
# df <- as.data.frame(rbind(d1,d2,d3,d4,d5))
# rownames(df) <- NULL
# df[,2:7] <- lapply(df[,2:7], as.numeric)
# chisq.test(as.matrix(df[,2:7]))

# d <- d[d$CallType == 13,]

# Re-number remaining call types consecutively from 1
#d$CallType <- match(d$CallType, sort(unique(d$CallType)))

#d$CallType <- as.numeric(call_map3[d$Hearing])
#d$CallType <- as.numeric(call_map3[d$Sex])
# d$CallType <- as.numeric(call_map3[d$Individual])

unique(d$CallType)

# File paths
paths <- paste0(
  "C:/Users/crave/Downloads/VocDataBase/VocDataBase/",
  d$ALFileName
)

# Audio loader
get_wave <- function(fname) {
  readWave(fname)@left / 32767
}
waves <- map(paths, get_wave)

# Get lengths
lengths <- sapply(waves, length)

# Get sample rate
sr <- readWave(paths[1])@samp.rate
fs <- sr
max_samples <- ceiling(25000) # Set max we are look at to half second calls

keep <- lengths <= max(lengths) # Currently set to keep all calls
#keep <- lengths <= max_samples # Currently set to restrict calls we look at
d     <- d[keep, ]
waves <- waves[keep]
waves_ds <- waves

# # Normalize the amplitude by call
for (i in 1:length(d$MicFileName)) {
  mu <- mean(waves_ds[[i]], na.rm = TRUE)
  sigma <- sd(waves_ds[[i]], na.rm = TRUE)
  waves_ds[[i]] <- (waves_ds[[i]] - mu) / sigma
}

# Stratified 80/20 split
train_idx <- unlist(
  lapply(unique(d$CallType), function(k) {
    idx <- which(d$CallType == k)
    sample(idx, size = floor(0.8 * length(idx)))
  })
)
test_idx <- setdiff(seq_len(nrow(d)), train_idx)

# test_individual <- "F6"
# 
# test_idx <- which(d$Individual == test_individual)
# train_idx <- setdiff(seq_len(nrow(d)), test_idx)


# Prepare waveform
max_len <- floor(25000)
waveforms <- matrix(0, length(waves_ds), max_len) # Zero pad
for (i in seq_along(waves_ds)) { # Fill in with sound
  waveforms[i, 1:min(max_len, length(waves_ds[[i]]))] <- head(waves_ds[[i]], max_len)
}

# max_len2 <- 250000
# step <- 10
# max_len <- ceiling(max_len2 / step)
# 
# waveforms <- matrix(
#   0,
#   length(waves_ds),
#   ceiling(max_len2 / step)
# )
# 
# for (i in seq_along(waves_ds)) {
#   x <- waves_ds[[i]]
#   
#   # Take every 10th sample from the first 250,000
#   idx <- seq(1, min(max_len2, length(x)), by = step)
#   sampled <- x[idx]
#   
#   # Fill waveform matrix
#   waveforms[i, 1:length(sampled)] <- sampled
# }

# Train / Test split
x_train_wave <- array_reshape(
  waveforms[train_idx, ],
  c(length(train_idx), max_len, 1)
)

x_test_wave <- array_reshape(
  waveforms[test_idx, ],
  c(length(test_idx), max_len, 1)
)

y_train <- d$CallType[train_idx] - 1
y_test  <- d$CallType[test_idx] - 1

num_classes <- 13#length(unique(c(y_train, y_test)))
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

cm <- confusionMatrix(
  factor(y_pred, levels = 0:12),
  factor(y_true, levels = 0:12)
)

cm_df <- as.data.frame.matrix(cm$table)

cm_df <- cbind(
  Prediction = rownames(cm_df),
  cm_df
)

write.csv(
  cm_df,
  "confusion_matrix.csv",
  row.names = FALSE
)
