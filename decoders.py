import numpy as np
from scipy.signal import lfilter
from hw_utils import reflection_coeff_to_polynomial_coeff
from utils1 import decode_LARC, LAR_to_reflection_coeff_prime
from utils2 import decode_N, decode_b, synthesis

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

def RPE_frame_st_decoder(
   LARc : np.ndarray, 
   curr_frame_st_resd : np.ndarray):
   LAR_prime = decode_LARC(LARc, A, B)
   r_prime  = LAR_to_reflection_coeff_prime(LAR_prime)
   a_prime,e_final = reflection_coeff_to_polynomial_coeff(r_prime)
   fir_coefficients = np.hstack(([1], -a_prime))
   s0 = lfilter([1], fir_coefficients, curr_frame_st_resd)
   return s0


def RPE_frame_slt_decoder(LARc: np.ndarray, Nc: int, bc: int, curr_frame_ex_full: np.ndarray, prev_frame_st_resd: np.ndarray):
  
   # Decoded N, b Values 
   N_dec = decode_N(Nc) 
   b_dec = decode_b(bc) 
   curr_frame_st_resd_array = []
   for subframe_id in range(NUM_SUBFRAMES):
      start_idx = 40 * subframe_id
      end_idx = 40 * (subframe_id + 1)
      curr_frame_st_resd = synthesis(curr_frame_ex_full[start_idx:end_idx], b_dec, prev_frame_st_resd, subframe_id, N_dec)
      prev_frame_st_resd = curr_frame_st_resd
      curr_frame_st_resd_array.append(curr_frame_st_resd)
   s0 = RPE_frame_st_decoder(LARc, curr_frame_st_resd_array)
      
   return  curr_frame_st_resd




