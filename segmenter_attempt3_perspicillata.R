rm(list = ls())
gc()

library(tuneR)
library(seewave)
library(tidyverse)
library(signal)

find_quiet_forward <- function(x, start, noise_rms, samples) {
  
  if (start > length(x)) {
    return(length(x))
  }
  
  # Need enough samples remaining to test a complete window
  last_start <- length(x) - samples + 1
  
  if (start > last_start) {
    return(length(x))
  }
  
  for (j in start:last_start) {
    
    window <- x[j:(j + samples - 1)]
    window_rms <- sqrt(mean(window^2))
    
    if (window_rms <= noise_rms) {
      return(j)
    }
  }
  
  return(length(x))
}


find_quiet_backward <- function(x, start, noise_rms, samples) {
  
  if (start < 1) {
    return(1)
  }
  
  first_start <- 1
  
  if (start < first_start) {
    return(1)
  }
  
  for (j in start:1) {
    
    if (j + samples - 1 > length(x)) {
      next
    }
    
    window <- x[j:(j + samples - 1)]
    window_rms <- sqrt(mean(window^2))
    
    if (window_rms <= noise_rms) {
      return(j + samples - 1)
    }
  }
  
  return(1)
}

# Read annotation table
wave <- readWave("D:\\Recordings 7.23-7.24.26\\Data\\ANGIE2_20260724_170703.wav")@left
fs <- readWave("D:\\Recordings 7.23-7.24.26\\Data\\ANGIE2_20260724_170703.wav")@samp.rate

# Build data frame and visualize
time <- 1:length(wave)/fs
d <- data.frame(wave, time)
#d$wave <- d$wave/max(abs(d$wave))
spectro(wave, fs, fastdisp = T, collevels = seq(-80, 0, by = 2))
# ggplot(data = d, mapping = aes(x = time, y = wave)) +
#   geom_line()

# Bandpass limits
low_cutoff <- 20000
high_cutoff <- 120000

# Design a 8th-order Butterworth bandpass filter
bf <- butter(
  n = 8,
  W = c(low_cutoff, high_cutoff) / (fs / 2),
  type = "pass"
)

# Apply the filter
wave_filtered <- filtfilt(bf, d$wave)
# spectro(wave_filtered, fs)
d$wave_filtered <- wave_filtered
#d$wave_filtered <- d$wave_filtered/max(abs(d$wave_filtered))

# Parameters for segmenter
thresh <- 1000 # Relative to maximum loudness
jump_forward <- 30 # in milliseconds
jump_backward <- 30

noise <- d$wave_filtered[1:100]
noise_samp <- 1:100
d_noise <- data.frame(noise, noise_samp)
# ggplot(data = d_noise, mapping = aes(x = noise_samp, y = noise)) +
#   geom_line()
noise_rms <- sqrt(mean(noise^2))

#  Require signal to remain below noise RMS for this long
duration <- 1
samples <- round(duration / 1000 * fs)

sound_num <- 0
i <- 1
final <- list()
times <- c()
freqs <- c()
while (i <= length(d$wave_filtered)) {
  if (abs(d$wave_filtered[i]) < thresh) {
    i <- i + 1
  } else {
    begin <- max(1, i - round(jump_backward / 1000 * fs))
    ending <- min(nrow(d), i + round(jump_forward / 1000 * fs))
    
    sound <- d[begin:ending,]
    E <- cumsum(sound$wave_filtered^2)
    E_norm <- E/max(E)
    
    PE_limits <- c(0.01, 0.95)
    PE_start <- which(E_norm >= PE_limits[1])[1]
    PE_end <- which(E_norm >= PE_limits[2])[1]
    
    pre_final <- sound$wave_filtered[PE_start:PE_end]
    
    peak <- which.max(abs(pre_final))
    mult <- 1
    
    forward_relative <- find_quiet_forward(
      x = pre_final,
      start = peak,
      noise_rms = noise_rms * mult,
      samples = samples
    )
    
    backward_relative <- find_quiet_backward(
      x = pre_final,
      start = peak - samples + 1,
      noise_rms = noise_rms * mult,
      samples = samples
    )
    
    final_start <- max(
      1,
      backward_relative
    )
    
    final_end <- min(
      length(pre_final),
      forward_relative + samples - 1
    )
    
    if (final_end <= final_start) {
      i <- i + round(jump_forward / 1000 * fs)
      next
    }
    
    final[[sound_num + 1]] <- pre_final[final_start:final_end]
    
    sig <- final[[sound_num + 1]]
    
    wl <- min(128, length(sig))
    wl <- wl - wl %% 2
    
    spec_data <- seewave::meanspec(
      sig,
      f = fs,
      wl = wl,
      plot = FALSE
    )
    
    peak_freq <- spec_data[which.max(spec_data[, 2]), 1]

    times <- c(times, d$time[i])
    freqs <- c(freqs, peak_freq)
    
    sound_num <- sound_num + 1
    i <- i + round(jump_forward/1000 * fs)
  }
}

wave_test <- final[[2]]
time_test <- 1:length(wave_test)/fs
df <- data.frame(wave_test, time_test)
ggplot(data = df, mapping = aes(x = time_test, y = wave_test)) +
  geom_line() +
  theme_bw() +
  ylab("Amplitude") +
  xlab("Time (s)")
spectro(wave_test, fs, wl = 32, collevels = seq(-20, 0, by = 1))

spectro(
  wave_filtered,
  fs,
  collevels = seq(-80, 0, by = 2),
  fastdisp = T
)
points(
  times,
  freqs,
  pch = 4,
  col = "red",
  cex = 1.5,
  lwd = 2
)
