import numpy as np

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

# SECOND PART
def RPE_subframe_slt_lte(d: np.ndarray, prev_d: np.ndarray):
    
    N_values = np.arange(40, 121)  # Possible values of N (40 ≤ N ≤ 120)
    R = np.zeros(len(N_values), dtype=float)

    # Compute R[i] for every possible lag in N_values
    for i, lag in enumerate(N_values):
        R[i] = np.sum(d * prev_d[120 - lag : 120 - lag + SUBFRAME_SIZE])

    # Select best lag (N) that maximizes R
    best_idx = np.argmax(R)
    N = N_values[best_idx]

    # Compute b 
    denominator = np.sum(prev_d[120 - N : 120 - N + SUBFRAME_SIZE] ** 2)
    if denominator == 0:
        b = 0  # Prevent division by zero
    else:
        b = np.sum(d * prev_d[120 - N : 120 - N + SUBFRAME_SIZE]) / denominator

    return N, b


def encode_N(n):
    return n & 0b1111111

def decode_N(encoded_value):
    return encoded_value

def encode_b(b):
    if b < DLB[0]:
        encoded_b = 0
    elif DLB[0] <= b < DLB[1]:
        encoded_b = 1
    elif DLB[1] <= b < DLB[2]:
        encoded_b = 2
    else:
        encoded_b = 3
    return int(encoded_b)

def decode_b(encoded_b):
    return QLB[int(encoded_b)]

def synthesis(e_prime_sub, bc, prev_frame_st_resd, start_id, Nc):
   curr_frame_st_resd = np.zeros(160)
   
   # Synthesis does the reverse process of Prediction, it uses calculated residual to compute the signal
   for k in range(len(e_prime_sub)):
      current_id = start_id + k 
      ref_id = current_id - Nc

      # Same train of thought here with the indexing, if ref_id > 0, no need to access the previous frame
      if ref_id >= 0:
         curr_frame_st_resd[k] = e_prime_sub[k] + bc * curr_frame_st_resd[ref_id]
      else: # indexing here is recalculated since there is need to access the previous frame
         ref_id_prev = ref_id + len(prev_frame_st_resd)
         curr_frame_st_resd[k] = e_prime_sub[k] + bc * prev_frame_st_resd[ref_id_prev]
   
   return curr_frame_st_resd


