import numpy as np

from utils1 import offset_compensation, pre_emphasis, autocorrelation_coeff, levinson_durbin, reflection_coeff_to_LAR, \
quantization_encode, decode_LARC, LAR_to_reflection_coeff_prime, compute_residual
from hw_utils import polynomial_coeff_to_reflection_coeff
from hw_utils import reflection_coeff_to_polynomial_coeff
from utils2 import synthesis, RPE_subframe_slt_lte, encode_b, encode_N, decode_b, decode_N

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


   
def RPE_frame_st_coder(
    s0: np.ndarray, 
    prev_frame_st_resd: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
  
   sof = offset_compensation(s0)
   s = pre_emphasis(sof)
   r = autocorrelation_coeff(s,8)
   a,e = levinson_durbin(r,8)
   rc = polynomial_coeff_to_reflection_coeff(a)
   lar = reflection_coeff_to_LAR(rc)
   LARc = quantization_encode(lar, A, B)
   LAR_prime = decode_LARC(LARc, A, B)
   r_prime = LAR_to_reflection_coeff_prime(LAR_prime)
   a_prime,e_final = reflection_coeff_to_polynomial_coeff(r_prime)
   curr_frame_st_resd = compute_residual(s0, a_prime)
   return LARc, curr_frame_st_resd, e_final, e




def RPE_frame_slt_coder(s0, prev_frame_st_resd):

   # Call function for Short Term Encoding
   LARc, d_frame, _ ,_ = RPE_frame_st_coder(s0, prev_frame_st_resd)

   curr_frame_ex_full = np.zeros(FRAME_SIZE)
   curr_frame_st_resd = np.zeros(FRAME_SIZE)

   for subframe_id in range(NUM_SUBFRAMES):
      start_idx = subframe_id * SUBFRAME_SIZE
      end_idx = (subframe_id + 1) * SUBFRAME_SIZE

      d_curr_subframe = d_frame[start_idx:end_idx]

      # Define Search Spaces for Long Term Correlation
      if start_idx == 0:
         past_dprime = prev_frame_st_resd[40:160] # 120 

      elif start_idx == 40:
        part1 = prev_frame_st_resd[80:160]       
        part2 = curr_frame_st_resd[0:40]         
        past_dprime = np.concatenate([part1, part2]) # 80 + 40 = 120

      elif start_idx == 80:
         part1 = prev_frame_st_resd[120:160]
         part2 = curr_frame_st_resd[0:80]
         past_dprime = np.concatenate([part1, part2]) # 40 + 80 = 120

      else:
         past_dprime = curr_frame_st_resd[0:120] # 120
         
      past_dprime = np.array(past_dprime)

      # Get pitch lag N and b for current subframe
      N_curr , b_curr = RPE_subframe_slt_lte(d_curr_subframe, past_dprime)
      
      # Quantize and Dequantize N and b
      N_encoded = encode_N(N_curr)
      Nc = decode_N(N_encoded)
      Nc = int(Nc)
      b_encoded = encode_b(b_curr)
      bc = decode_b(b_encoded)
      
      # Initialize Residual Array for Current Subframe (40 samples)
      e_sub = np.zeros(SUBFRAME_SIZE, dtype = float)

      # Prediction
      for i in range(SUBFRAME_SIZE):
         
         current_id = start_idx + i 
         ref_id = current_id - Nc
         
         # if ref_id >= 0 we are in the same frame, no need to access prev_frame_st_resd
         if ref_id >= 0:
            e_sub[i] = d_curr_subframe[i] - bc * curr_frame_st_resd[ref_id]
         else: #This accesses the previous frame
            ref_id_prev = ref_id + FRAME_SIZE
            e_sub[i] = d_curr_subframe[i] - bc * prev_frame_st_resd[ref_id_prev]
           
      curr_frame_ex_full[start_idx:end_idx] = e_sub
      e_prime_sub = e_sub #This is given by the instructions, ignore sub-routine (γ)

      # Synthesis
      curr_frame_st_resd = synthesis(e_prime_sub, bc, prev_frame_st_resd, start_idx, Nc)
      
   return LARc, Nc, bc, curr_frame_ex_full, curr_frame_st_resd






