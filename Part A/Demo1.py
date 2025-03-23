import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import time
from encoders import RPE_frame_st_coder
from decoders import RPE_frame_st_decoder
from utils1 import read_split_file


if __name__ == "__main__": 
   sample_rate, frames, original_data, num_frames= read_split_file("ena_dio_tria.wav")
  # sample_rate, frames, original_data, num_frames = read_split_file(
  #  r"C:\Users\nirva\OneDrive\Υπολογιστής\MultimediaSystemsAssignment\ena_dio_tria.wav")

   # Play original sound
   print("Playing original audio...")
   sd.play(original_data, samplerate=sample_rate)
   time.sleep(len(original_data) / sample_rate)

   # Process each frame
   processed_frames = []
   prev_residual = None  # Initialize previous residual

   for frame in frames:
      LARc, curr_frame_st_resd, e_final, e = RPE_frame_st_coder(frame, prev_residual)
      reconstructed_frame = RPE_frame_st_decoder(LARc, curr_frame_st_resd)
      processed_frames.append(reconstructed_frame)
      prev_residual = curr_frame_st_resd

   # Reconstruct the full processed signal
   processed_audio = np.concatenate(processed_frames)

   # Play reconstructed sound
   print("Playing reconstructed audio...")
   sd.play(processed_audio, samplerate=sample_rate)
   time.sleep(len(processed_audio) / sample_rate)


duration = len(original_data) / sample_rate  # Total duration in seconds
time2 = np.linspace(0, duration, num=len(original_data))

# Plot the signal
plt.figure(figsize=(10, 4))
plt.plot(time2, original_data, label="Audio Signal", color="C0")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("Waveform of Audio Signal")
plt.legend()
plt.grid()
plt.show()
