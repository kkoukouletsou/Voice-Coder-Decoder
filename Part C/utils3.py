def weighting_filter(input_signal):
    H = [-134, -374, 0, 2054, 5741, 8192, 5741, 2054, 0, -374, -134];
    H = np.array(H)/2**13;
    filtered_signal = np.convolve(input_signal, H, mode='same');
    return filtered_signal;


import numpy as np
def downsampling(subframe):
    seq1 = subframe[0::3]  # x(k0 + 40j + 0), x(k0 + 40j + 3), ..., x(k0 + 40j + 36)
    seq2 = subframe[1::3]  # x(k0 + 40j + 1), x(k0 + 40j + 4), ..., x(k0 + 40j + 37)
    seq3 = subframe[2::3]  # x(k0 + 40j + 2), x(k0 + 40j + 5), ..., x(k0 + 40j + 38)
    seq4 = subframe[3::3]  # x(k0 + 40j + 3), x(k0 + 40j + 6), ..., x(k0 + 40j + 39)
    return seq1, seq2, seq3, seq4


def get_Mc(seq):
    power1 = sum(x**2 for x in seq[0])
    power2 = sum(x**2 for x in seq[1])
    power3 = sum(x**2 for x in seq[2])
    power4 = sum(x**2 for x in seq[3])
    powers = [power1, power2, power3, power4];
    max_power = max(powers);
    Mc = powers.index(max_power);
    return Mc

def quantize_dequantize_xmax(xmax):
    levels = [0, 31, 63, 95, 127, 159, 191, 223, 255, 287, 319, 351, 383, 415, 447, 479, 511, 575, 639, 703, 767, 831, 895, 
              959, 1023, 1151, 1279, 1407, 1535, 1663, 1791, 1919, 2047, 2303, 2559, 2815, 3071, 3327, 3583, 3839, 4095, 
              4607, 5119, 5631, 6143, 6655, 7167, 7679, 8191, 9215, 10239, 11263, 12287, 13311, 14335, 15359, 16383, 18431, 
              20479, 22527, 24575, 26623, 28671, 30719, 32767]
    
    xmaxq = levels[-1]  # Default to the maximum level
    xmaxc = len(levels) - 1  # Default index
    
    for i in range(len(levels)):
        if xmax <= levels[i]:
            xmaxq = levels[i]
            xmaxc = i
            break
    
    return xmaxq, xmaxc




def quantize_dequantize_excitation_sequence(sequence):
   quantized_sequence = np.zeros(len(sequence))
   xMC = np.zeros(7)
   sequence = sequence * (2**15)
   interval_limits = [-32768, -24576, -16384, -8192, 0, 8192, 16384, 24576]

   for sample in sequence:
    for i in range(len(interval_limits) - 1):
        if interval_limits[i] <= sample < interval_limits[i + 1]:
            quantized_value = (interval_limits[i] + interval_limits[i + 1]) / 2
            quantized_sequence[i] = quantized_value
            xMC[i] = i
            break
   xM_prime = np.array(quantized_sequence) * (2**(-15))      
   xMC_binary = [format(int(i), '03b') for i in xMC]

   return xM_prime, xMC, xMC_binary

   
def excitation_computation(curr_frame_ex_full):
   Mc_array = []
   x_max_c_arr = []
   best_sequencies_for_each_subframe = []

   for j in range(4):
      filtered_err = weighting_filter(curr_frame_ex_full[40 * j:40 * (j + 1)]) # τα σφάλματα μετά την εφαρμογή του φίλτρου
      sequences = downsampling(filtered_err) # downsampling function returns the 4 downsampled sequences.
      Mc = get_Mc(sequences) # Find Mc
      max_power_sequence = sequences[Mc] # Chosen sequence, that happens for Mc
      x_max = max(abs(max_power_sequence))
      x_max_quant, x_max_dequant = quantize_dequantize_xmax(x_max)
      x_prime = max_power_sequence / x_max_dequant
      xM_prime, xMC, xMC_binary = quantize_dequantize_excitation_sequence(x_prime) # κβαντισμος της x'(i)=x_M(i)/x'max
      best_sequencies_for_each_subframe.append(xM_prime)
   return best_sequencies_for_each_subframe, xMC, xMC_binary





def compose_frame(LARc, Nc_arr, bc_arr, Mc_arr, x_max_c_arr, best_sequence):
   binary_block = [];
   for i in range(4):
      binary_block += LARc[i * 8 :i * 8 + 7]; # από τον πίνακα τον LAR ολου του frame επιλέγονται καθε φορα τα 8 LAR του συγκεκριμενου subframe
      binary_block += [Nc_arr[i], bc_arr[i], Mc_arr[i], x_max_c_arr[i]];
      binary_block += binary_block + best_sequence[i]
   return binary_block

      
    
def binaryblock_to_parameters(block):
   LARc = []
   Nc = []
   bc = []
   M = []
   Xmax = []
   x = []

   # Extract LARc values
   LARc.append(int(''.join(map(str, block[0:6])), 2))
   LARc.append(int(''.join(map(str, block[6:12])), 2))
   LARc.append(int(''.join(map(str, block[12:17])), 2))
   LARc.append(int(''.join(map(str, block[17:22])), 2))
   LARc.append(int(''.join(map(str, block[22:26])), 2))
   LARc.append(int(''.join(map(str, block[26:30])), 2))
   LARc.append(int(''.join(map(str, block[30:33])), 2))
   LARc.append(int(''.join(map(str, block[33:36])), 2))

   # Extract Nc, bc, M, Xmax, and x values
   for j in range(4):
      s = 56 * j + 37
      Nc.append(int(''.join(map(str, block[s:s + 7])), 2))
      bc.append(int(''.join(map(str, block[s + 7:s + 9])), 2))
      M.append(int(''.join(map(str, block[s + 9:s + 11])), 2) + 1)
      Xmax.append(int(''.join(map(str, block[s + 11:s + 17])), 2))
      x_j = []
      for i in range(13):
         x_j.append(int(''.join(map(str, block[s + 17 + 3 * i:s + 19 + 3 * i])), 2))
      x.append(x_j)

      return LARc, Nc, bc, M, Xmax, x
   

def params2bin(LARc, Nc, bc, M, Xmax, x):
   block = [0] * 260

   # Assign LARc values
   block[0:6] = [int(bit) for bit in format(LARc[0], '06b')]
   block[6:12] = [int(bit) for bit in format(LARc[1], '06b')]
   block[12:17] = [int(bit) for bit in format(LARc[2], '05b')]
   block[17:22] = [int(bit) for bit in format(LARc[3], '05b')]
   block[22:26] = [int(bit) for bit in format(LARc[4], '04b')]
   block[26:30] = [int(bit) for bit in format(LARc[5], '04b')]
   block[30:33] = [int(bit) for bit in format(LARc[6], '03b')]
   block[33:36] = [int(bit) for bit in format(LARc[7], '03b')]

   # Process Nc, bc, M, Xmax, and x values
   for j in range(4):
      k = 56 * j + 37
        
      block[k:k + 7] = [int(bit) for bit in format(Nc[j], '07b')]
      block[k + 7:k + 9] = [int(bit) for bit in format(bc[j], '02b')]
      block[k + 9:k + 11] = [int(bit) for bit in format(M[j] - 1, '02b')]
      block[k + 11:k + 17] = [int(bit) for bit in format(Xmax[j], '06b')]

      for i in range(13):
         block[k + 17 + 3 * i : k + 19 + 3 * i] = [int(bit) for bit in format(x[j][i], '03b')]

   return block


def compute_residual_from_excitation_sequence(Mc, x_maxc, xc):
    xc = np.array(xc).flatten()

    
    levels1 = np.array([-0.875, -0.625, -0.375, -0.125, 0.125, 0.375, 0.625, 0.875])
    levels2 = np.array([31, 63, 95, 127, 159, 191, 223, 255, 287, 319, 351, 383, 415, 447, 479, 511,
                    575, 639, 703, 767, 831, 895, 959, 1023, 1151, 1279, 1407, 1535, 1663, 1791, 1919, 2047,
                    2303, 2559, 2815, 3071, 3327, 3583, 3839, 4095, 4607, 5119, 5631, 6143, 6655, 7167, 7679, 8191,
                    9215, 10239, 11263, 12287, 13311, 14335, 15359, 16383, 18431, 20479, 22527, 24575, 26623, 28671,
                    30719, 32767])
    xm = levels1[xc]
    x_max = levels2[x_maxc]
    x = xm * x_max * 2**-15
    residual = np.zeros(40)
    if Mc == 1:
        residual[int(Mc)-1::3] = x
    else:
        residual[int(Mc)::3] = x
    return residual

