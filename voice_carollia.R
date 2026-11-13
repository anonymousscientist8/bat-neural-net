rm(list = ls())
gc()

# Load packages
library(readxl)
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

# Folder containing your CSV files
files <- list.files(
  "C:/Users/crave/Box/imisstheserver/LamanaiColonyCage_Carollia/labels",
  pattern = "\\.csv$",
  full.names = TRUE
)

files <- files[basename(files) != "batch_summary.csv"]

# Read all CSVs and extract WAV filename + onset/offset/label
all_data <- map_dfr(files, function(f) {
  
  # Extract ANGIE1 through the end of the filename
  wav_name <- basename(f) |>
    str_extract("ANGIE1.*(?=\\.csv$)") |>
    paste0(".wav")
  
  # Read CSV
  dat <- read_csv(f, show_col_types = FALSE)
  
  # Extract the columns of interest
  dat |>
    select(`Onset (s)`, `Offset (s)`, Label) |>
    mutate(
      wav_file = wav_name,
      source_csv = basename(f),
      .before = 1
    )
})

d <- all_data

training <- d %>%
  mutate(
    dataset = "training"
  )

new_dir <- "C:/Users/crave/Box/imisstheserver/Emily Stuff/For Raven"

new_xlsx <- list.files(
  new_dir,
  pattern = "\\.xlsx$",
  full.names = TRUE
)

new_xlsx <- new_xlsx[
  basename(new_xlsx) != "Call Type Codes.xlsx"
]

testing <- map_dfr(new_xlsx, function(f) {
  
  dat <- read_excel(f)
  
  wav_file <- paste0(
    tools::file_path_sans_ext(basename(f)),
    ".wav"
  )
  
  dat %>%
    transmute(
      wav_file = wav_file,
      `Onset (s)` = `start time`,
      `Offset (s)` = `end time`,
      Label = call.type,
      dataset = "testing",
      wav_path = file.path(new_dir, wav_file)
    )
})

d <- bind_rows(
  training,
  testing
)

# Encode call types
d <- d[d$Label %in% c("aggressive down-sweeps (A)", "distress calls",
                      "echolocation calls", "DC", "EC", "ADC-CL", "ADS"),]
call_map <- c(
  "aggressive down-sweeps (A)" = 1, "distress calls" = 2,
  "echolocation calls" = 3, "DC" = 2, "EC" = 3, "ADC-CL" = 1, "ADS" = 1
)
d$Label <- as.numeric(call_map[d$Label])
length(unique(d$Label))

# Get sample rate
sr <- readWave("C:\\Users\\crave\\Box\\imisstheserver\\Emily Stuff\\For Raven\\ANGIE2_20260730_191516.wav")@samp.rate
fs <- sr

# Base directories
training_dir <- "C:/Users/crave/Box/imisstheserver/LamanaiColonyCage_Carollia"
testing_dir  <- "C:/Users/crave/Box/imisstheserver/Emily Stuff/For Raven"


# Create the appropriate WAV path for each row
paths <- ifelse(
  d$dataset == "training",
  file.path(training_dir, d$wav_file),
  file.path(testing_dir, d$wav_file)
)


# Read only the annotated segment
get_wave <- function(fname, on, off) {
  wave <- readWave(
    fname,
    from = on,
    to = off,
    units = "seconds"
  )
  
  wave@left
}


# Extract each annotated segment
waves <- map2(
  paths,
  seq_len(nrow(d)),
  ~ get_wave(
    .x,
    d$`Onset (s)`[.y],
    d$`Offset (s)`[.y]
  )
)
# Get lengths
lengths <- sapply(waves, length)
max_samples <- 6200 # Set max we are look at to half second calls

keep <- lengths <= max(lengths) # Currently set to keep all calls
#keep <- lengths <= max_samples # Currently set to restrict calls we look at
d     <- d[keep, ]
waves <- waves[keep]
waves_ds <- waves

# # Normalize the amplitude by call
for (i in 1:length(d$wav_file)) {
  mu <- mean(waves_ds[[i]], na.rm = TRUE)
  sigma <- sd(waves_ds[[i]], na.rm = TRUE)
  waves_ds[[i]] <- (waves_ds[[i]] - mu) / sigma
}

# Get row indices
train_idx <- which(d$dataset == "training")
test_idx  <- which(d$dataset == "testing")


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
  layer_global_max_pooling_1d()


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
  epochs = 50,
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
