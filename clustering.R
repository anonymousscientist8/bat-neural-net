rm(list = ls())
gc()

# =========================
# Load packages
# =========================
library(keras)
library(signal)
library(tidyverse)
library(tuneR)
library(caret)
library(uwot)
library(RColorBrewer)
library(dbscan)
library(irlba)
library(factoextra)

# =========================
# Read annotation table
# =========================
d <- read.csv("C:\\Users\\Raven\\AnnotationTable2.csv", header = TRUE)
unique(d$Guess)
d# <- d[!(d$Guess == ""),]

# =========================
# Encode caller
# =========================
call_map <- c(
  "dash" = 1, "down" = 2, "down3" = 3, "i" = 4, "i3" = 4,
  "n" = 5, "N" = 6, "r" = 7, "r2" = 8, "serifn" = 9,
  "sqrt" = 10, "v" = 11, "smalln1" = 12, "smalln" = 13, "r4" = 14,
  "halfn" = 15, "dash1" = 16, "noise" = 17)

#d$Guess <- as.numeric(call_map[d$Guess])

# =========================
# File paths
# =========================
paths <- paste0(
  "C:/Users/Raven/Vampire_audio/",
  d$Filename, ".wav"
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
n_fft <- 1024

max_time_bins <- 256   # time dimension
max_freq_bins <- n_fft / 2 + 1

# =========================
# Spectrogram function (no normalization yet)
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
  if (is.null(dim(S))) S <- matrix(S, ncol = 1)
  
  S <- abs(S)
  S[!is.finite(S)] <- 0
  S <- log10(S + 1e-8)
  S[!is.finite(S)] <- 0
  
  S_fixed <- matrix(0, max_freq_bins, max_time_bins)
  f <- min(nrow(S), max_freq_bins)
  t <- min(ncol(S), max_time_bins)
  t_start <- floor((max_time_bins - t) / 2) + 1
  S_fixed[1:f, t_start:(t_start + t - 1)] <- S[1:f, 1:t]
  
  return(S_fixed)
}

# =========================
# Build raw spectrograms (no normalization)
# =========================
specs_raw <- array(
  0,
  dim = c(length(waves_ds), max_freq_bins, max_time_bins)
)

for (i in seq_along(waves_ds)) {
  specs_raw[i,,] <- make_spectrogram(waves_ds[[i]])
}

# =========================
# Global normalization
# =========================
all_values <- as.vector(specs_raw)
all_values <- all_values[all_values != 0]
global_mu <- mean(all_values, na.rm = TRUE)
global_sigma <- sd(all_values, na.rm = TRUE)

specs <- array(
  0,
  dim = c(length(waves_ds), max_freq_bins, max_time_bins, 1)
)

for (i in seq_along(waves_ds)) {
  specs[i, , , 1] <- (specs_raw[i,,] - global_mu) / global_sigma
}

# =========================
# Prepare waveform array
# =========================
max_len <- floor(fs * 0.5) # 1 second max
waveforms <- matrix(0, length(waves_ds), max_len)
for (i in seq_along(waves_ds)) {
  waveforms[i, 1:min(max_len, length(waves_ds[[i]]))] <- head(waves_ds[[i]], max_len)
}




# Flatten spectrograms
specs_flat <- t(apply(specs, 1, function(x) as.vector(x[,,1])))

# Compute top 50 PCs using irlba
pca_res <- irlba(specs_flat, nv = 10)  # nv = # of PCs
pca_scores <- pca_res$u %*% diag(pca_res$d)

# Run k-means
# fviz_nbclust(pca_scores, kmeans, method = "gap_stat", k.max = 20)
#fviz_nbclust(pca_scores, kmeans, method = "silhouette", k.max = 50)
#k <- 13  # start with 2-4 clusters
k <- 8
#km <- kmeans(pca_scores, centers = k)
km <- hdbscan(pca_scores, minPts = 10)
clusters <- km$cluster
table(km$cluster)
# clusters_clean <- clusters
# # or assign to a new cluster if you want:
# clusters_clean[clusters_clean == 0] <- max(clusters) + 1
# clusters <- clusters_clean
# gmm_model <- Mclust(pca_scores, G = 10)  # G = number of clusters
# 
# # Cluster assignments (hard)
# gmm_clusters <- gmm_model$classification
# 
# # Soft assignments (probabilities)
# gmm_probs <- gmm_model$z  # N x G matrix

# Visualize with UMAP
umap_res <- umap(pca_scores)
umap_df <- data.frame(
  UMAP1 = umap_res[,1],
  UMAP2 = umap_res[,2],
  Cluster = factor(clusters),
  Guess = d$Guess,
  Caller = d$Caller
)

# ggplot(umap_df_gmm, aes(x = UMAP1, y = UMAP2, color = Cluster)) +
#   geom_point(size = 2, alpha = 0.8) +
#   theme_minimal(base_size = 14) +
#   labs(title = "GMM Clusters of Spectrograms", color = "Cluster") +
#   theme(plot.title = element_text(hjust = 0.5))
# umap_df <- data.frame(
#   UMAP1 = umap_res[,1],
#   UMAP2 = umap_res[,2],
#   Cluster = factor(km$cluster)  # factor for ggplot coloring
# )
ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = Cluster)) +
  geom_point(size = 2, alpha = 0.8) +
  theme_bw(base_size = 14) +
  labs(
    title = "UMAP of Spectrograms",
    x = "UMAP 1",
    y = "UMAP 2",
    color = "Cluster"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "right"
  )

umap_df <- umap_df[umap_df$Guess != "",]

ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = Caller)) +
  geom_point(size = 2, alpha = 0.8) +
  theme_bw(base_size = 14) +
  labs(
    title = "UMAP of Spectrograms",
    x = "UMAP 1",
    y = "UMAP 2",
    color = "Cluster"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "right"
  )

ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = Guess)) +
  geom_point(size = 2, alpha = 0.8) +
  theme_bw(base_size = 14) +
  labs(
    title = "UMAP of Spectrograms",
    x = "UMAP 1",
    y = "UMAP 2",
    color = "Cluster"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "right"
  )

# km <- kmeans(pca_scores, centers = k, nstart = 25)
#clusters <- gmm_clusters  # use the hard assignment

# Function to plot a spectrogram
# plot_spectrogram <- function(spec_matrix, title = "") {
#   image(
#     t(spec_matrix),   # transpose only, do NOT reverse
#     axes = FALSE,
#     col = heat.colors(100),
#     main = title
#   )
#   box()
# }

freqs <- seq(0, fs/2, length.out = max_freq_bins)
times <- seq(
  0,
  by = hop_length / fs,
  length.out = max_time_bins
)

plot_spectrogram <- function(spec_matrix, title = "") {
  
  freqs_khz <- seq(0, fs / 2, length.out = nrow(spec_matrix)) / 1000
  times <- seq(0, by = hop_length / fs, length.out = ncol(spec_matrix))
  
  par(mar = c(4.5, 5.5, 3, 1))
  
  image(
    x = times,
    y = freqs_khz,
    z = t(spec_matrix),
    col = heat.colors(100),
    axes = FALSE,
    xlab = "",
    ylab = "",
    main = title
  )
  
  axis(1)
  axis(2, las = 1)
  
  mtext("Frame time (s, centered window)", side = 1, line = 3)
  mtext("Frequency (kHz)", side = 2, line = 3.5)
  
  box()
}

show_cluster_samples <- function(cluster_id, n_samples = 5) {
  idxs <- which(clusters == cluster_id)
  if (length(idxs) == 0) {
    cat("No samples in cluster", cluster_id, "\n")
    return()
  }
  
  selected <- sample(idxs, min(n_samples, length(idxs)))
  
  # Arrange plots in a row
  par(mfrow = c(1, length(selected)), mar = c(2,2,2,2))
  selected <- c(1,11,24)
  for (i in selected) {
    plot_spectrogram(specs[i, , , 1])
  }
  
  npar(mfrow = c(1,1))  # reset layout
}

show_cluster_samples(cluster_id = 2, n_samples = 3)
