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
wave <- readWave("C:\\Users\\crave\\Downloads\\ANGIE2_20240129_180332.wav")@left
fs <- readWave("C:\\Users\\crave\\Downloads\\ANGIE2_20240129_180332.wav")@samp.rate

# Build data frame and visualize
time <- 1:length(wave)/fs
d <- data.frame(wave, time)
d$wave <- d$wave/max(abs(d$wave))
# spectro(wave, fs)
# ggplot(data = d, mapping = aes(x = time, y = wave)) +
#   geom_line()

# Bandpass limits
low_cutoff <- 30000
high_cutoff <- 90000

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

# Parameters for segmenter
thresh <- 0.2 # Relative to maximum loudness
jump_forward <- 30 # in milliseconds
jump_backward <- 30

noise <- d$wave_filtered[497000:497100]
noise_samp <- 1:101
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
    begin <- i - round(jump_backward/1000 * fs)
    ending <- i + round(jump_forward/1000 * fs)
    
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
      i <- i + jump_forward_samples
      next
    }
    
    final[[sound_num + 1]] <- pre_final[final_start:final_end]
    
    spec_data <- seewave::meanspec(
      final[[sound_num + 1]],
      f = fs,
      wl = min(128, length(final[[sound_num + 1]])),
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
  collevels = seq(-40, 0, by = 2)
)
points(
  times,
  freqs,
  pch = 4,
  col = "red",
  cex = 1.5,
  lwd = 2
)



# 
# spec <- spectro(
#   wave_test,
#   f = fs,
#   wl = 32,
#   ovlp = 32,
#   plot = FALSE
# )
# 
# spec_df <- expand.grid(
#   time = spec$time,
#   frequency = spec$freq
# )
# 
# spec_df$amplitude <- as.vector(spec$amp)
# 
# ggplot(spec_df, aes(time, frequency, fill = amplitude)) +
#   geom_raster() +
#   scale_fill_viridis_c(
#     limits = c(-20, 0),
#     oob = scales::squish
#   ) +
#   theme_bw() +
#   labs(
#     x = "Time (s)",
#     y = "Frequency (Hz)",
#     fill = "Amplitude (dB)"
#   )
# 
# 
# spectro(
#   wave_test,
#   f = fs,
#   wl = 32,
#   ovlp = 32,
#   collevels = seq(-20, 0, by = 1)
# )
# 
# image(
#   x = spec$time,
#   y = spec$freq,
#   z = t(spec$amp),
#   col = viridis(100),
#   useRaster = TRUE
# )
