import numpy as np
from scipy.signal import lfilter
from scipy.io import wavfile


# Constants
ALPHA = 32735 * (2 ** -15)  # Coefficient for pre-emphasis
BETA = 28180 * (2 ** -15)
FRAME_SIZE = 160  # Number of samples per frame
PREDICTOR_ORDER = 8  # Order of LPC predictor
A = np.array([20.000, 20.000, 20.000, 20.000,
        13.637, 15.000, 8.334, 8.824])

B = np.array([0.000, 0.000, 4.000, -5.000, 0.184,
        -3.500, -0.666, -2.235])


DLB = [0.2, 0.5, 0.8]
QLB = [0.10, 0.35, 0.65, 1.00]
SUBFRAME_SIZE = 40
NUM_SUBFRAMES = 4
FRAME_SIZE = NUM_SUBFRAMES * SUBFRAME_SIZE

def read_split_file(file):
   sample_rate, data = wavfile.read(file)

   if data.dtype == np.int16:
      data = data / 32768.0

   num_frames = len(data) // FRAME_SIZE
   remainder = len(data) % FRAME_SIZE
   
   # Perform Zero Padding if necessary (if last frame has less than 160 samples)
   if remainder != 0:
      padding_length = FRAME_SIZE - remainder
      data = np.pad(data, (0, padding_length), mode='constant', constant_values=0)

   num_frames = len(data) // FRAME_SIZE
   frames = np.array_split(data, num_frames)
   return sample_rate, frames, data, num_frames



def offset_compensation(signal):
  sof = np.zeros_like(signal)
  z0 = 0 
  z1 = 0
  for i in range(len(signal)):
   s1 = signal[i] - z0
   z0 = signal[i]
   sof[i] = s1 + ALPHA * z1
   z1 = sof[i] 
  return sof 

def pre_emphasis(sof):
   s = np.zeros_like(sof)
   z = 0
   for i in range (len(sof)):
      s[i] = sof[i] - z * BETA
      z = sof[i]
   return s

def autocorrelation_coeff(sof, order):
    acf = np.zeros(order +1)
    for k in range (order + 1):
        acf[k] = np.sum(sof[k:] * sof[:FRAME_SIZE - k])
    return acf

def levinson_durbin(r, order):

    a = np.zeros(order + 1, dtype=float)
    a[0] = 1.0            # Force the polynomial to be monic from the start.
    e = float(r[0])        # Initial error.

    for i in range(1, order + 1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i-j]
        k = -acc / e

        a_old = a.copy()
        for j in range(1, i):
            a[j] = a_old[j] + k * a_old[i - j]
        a[i] = k
        e = e * (1 - k**2)

    return a, e

def reflection_coeff_to_LAR(rc):
   lar = np.zeros_like(rc)
   for k in range (len(rc)):
      if np.abs(rc[k]) < 0.675:
         lar[k] = rc[k] 
      elif 0.950 <= np.abs(rc[k]) <=1.000:
         lar[k] = np.sign(rc[k]) * (8 * np.abs(rc[k]) * (-6.375))
      else :
         lar[k] = np.sign(rc[k]) * (2 * np.abs(rc[k]) * (-0.675))
   return lar

def quantization_encode(LAR, A, B):
   LARC = np.zeros_like(LAR)
   LARC = ((A * LAR + B) + np.sign(A * LAR + B) * 0.5).astype(int) 
   return LARC

def decode_LARC (LARC,A ,B):
   LAR_prime = np.zeros_like(LARC)
   LAR_prime = (LARC - B) / A
   return LAR_prime
      
def LAR_to_reflection_coeff_prime(LAR):
   r_prime = np.zeros_like(LAR)
   for k in range (len(LAR)):
      if np.abs(LAR[k]) < 0.675:
         r_prime[k] = LAR[k] 
      elif 1.225 <= np.abs(LAR[k]) <= 1.625:
         r_prime[k] = np.sign(LAR[k]) * (0.125 * np.abs(LAR[k]) + 0.796875)
      else :
         r_prime[k] = np.sign(LAR[k]) * (0.5 * np.abs(LAR[k]) + 0.337500)
   return r_prime

def compute_residual(signal, a_prime):
   if len(signal) == 0:
      return np.zeros(160)
   else: 
      # Construct FIR filter coefficients [1, -a'_1, -a'_2, ..., -a'_8]
      fir_coefficients = np.hstack(([1], -a_prime))

      # Apply the FIR filter
      residual = lfilter(fir_coefficients, [1], signal)

   return residual
   
