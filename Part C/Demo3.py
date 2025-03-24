import sounddevice as sd
import time
import numpy as np
from decoders import RPE_frame_slt_decoder 
from encoders import RPE_frame_slt_coder
from utils1 import read_split_file
from utils3 import compute_residual_from_excitation_sequence, excitation_computation

DLB = [0.2, 0.5, 0.8]
QLB = [0.10, 0.35, 0.65, 1.00]
SUBFRAME_SIZE = 40
NUM_SUBFRAMES = 4
FRAME_SIZE = NUM_SUBFRAMES * SUBFRAME_SIZE


if __name__ == "__main__":
    sample_rate, frames, original_data, num_frames = read_split_file("ena_dio_tria.wav")
    #sample_rate, frames, original_data, num_frames = read_split_file(
    #r"C:\Users\nirva\OneDrive\Υπολογιστής\MultimediaSystemsAssignment\ena_dio_tria.wav")

    # Play original sound
    print("Playing original audio...")
    sd.play(original_data, samplerate=sample_rate)
    time.sleep(len(original_data) / sample_rate)

    LARc_list = []
    curr_frame_ex_full_list = []
    curr_frame_st_resd_list = []
    Nc_list = []  
    bc_list = []  
    block = np.zeros(260)
    blocks = []
    prev_frame_st_resd = np.zeros(160)
    count = 0

    Mc_array = []
    best_seq_arr = []
    xMC_arr = []

    # Coder loop
    for frame in frames:
        LARc, Nc, bc, curr_frame_ex_full, curr_frame_st_resd = RPE_frame_slt_coder(frame, prev_frame_st_resd)
        
        LARc_list.extend(LARc)
        curr_frame_ex_full_list.extend(curr_frame_ex_full)
        curr_frame_st_resd_list.extend(curr_frame_st_resd)
        Nc_list.append(Nc)
        bc_list.append(bc)
        
        # Call for Excitation Computation
        best_sequencies_for_each_subframe, xMC, xMC_binary = excitation_computation(curr_frame_ex_full)
        Mc_array.append(xMC)
        xMC_arr.append(xMC)
        best_seq_arr.append(best_sequencies_for_each_subframe)
        #block = params2bin(LARc, Nc_list, bc_list, Mc_array, x_max_c_arr, xc_arr)
        
        prev_frame_st_resd = curr_frame_st_resd
        count = count + 1

    store_processed_frames = np.array([])
    count = 0
    prev_frame_st_resd_batch = 160 * [0]


    count1 = 0 
    # Decoder loop
'''    for frame in frames:
        # Extract 8 LARc values
        LARc_batch = LARc_list[count * 8 : (count + 1) * 8]
        curr_frame_ex_full_batch = curr_frame_ex_full_list[count * 160 : (count + 1) * 160]
        curr_frame_st_resd_batch = curr_frame_st_resd_list[count * 160 : (count + 1) * 160]
        Nc = Nc_list[count]
        bc = bc_list[count]
        print(np.size(Mc_array))
        print(np.size(xMC_arr))
        print(len(best_seq_arr))
        curr_frame_ex_full_batch = compute_residual_from_excitation_sequence(Mc_array[1:8], xMC_arr[1:8], best_seq_arr)
        curr_frame_st_resd = RPE_frame_slt_decoder(LARc_batch, Nc, bc, curr_frame_ex_full_batch, curr_frame_st_resd_batch)

        store_processed_frames = np.concatenate((store_processed_frames, curr_frame_st_resd))
        prev_frame_st_resd = curr_frame_st_resd_batch
        count1 += 1

    # Play reconstructed sound
    print("Playing reconstructed audio...")
    sd.play(store_processed_frames, samplerate=sample_rate)
    time.sleep(len(store_processed_frames) / sample_rate)'''
