import numpy as np
import os
import pandas as pd

def refine_formants_row_by_row(predicted_formants, spectral_peaks):
    """
    Refines each row in predicted_formants using the corresponding row in spectral_peaks.
    Assumes both inputs are 2D arrays or DataFrames with aligned rows.
    """
    refined_formants = []
    
    for frame_formants, peaks in zip(predicted_formants, spectral_peaks):
        assigned_peaks = set()
        refined_frame = []
        for formant in frame_formants:
            closest_peak = min(
                (peak for peak in peaks if peak not in assigned_peaks),
                key=lambda peak: abs(peak - formant)
            )
            refined_frame.append(closest_peak)
            assigned_peaks.add(closest_peak)
        refined_formants.append(refined_frame)
    return np.array(refined_formants)

def refine_formants_row_by_row(predicted_formants, spectral_peaks):
    """
    Refines each row in predicted_formants using the corresponding row in spectral_peaks.
    Assumes both inputs are 2D arrays or DataFrames with aligned rows.
    """
    refined_formants = []
    
    for frame_formants, peaks in zip(predicted_formants, spectral_peaks):
        assigned_peaks = set()
        refined_frame = []
        for formant in frame_formants:
            closest_peak = min(
                (peak for peak in peaks if peak not in assigned_peaks),
                key=lambda peak: abs(peak - formant)
            )
            refined_frame.append(closest_peak)
            assigned_peaks.add(closest_peak)
        refined_formants.append(refined_frame)
    return np.array(refined_formants)



def load_formants_from_directory(directory_path):
    """
    Loads all formant data from files in a directory.
    Assumes files are in .txt or .csv format with rows as frames.
    """
    all_formants = []
    file_names = sorted(f for f in os.listdir(directory_path) if not f.startswith('.'))  # Sort to align files between directories

    for file_name in file_names:
        if file_name.endswith('.txt') or file_name.endswith('.csv'):
            file_path = os.path.join(directory_path, file_name)
            formants = pd.read_csv(file_path)
            predicted = formants.iloc[:, 1:4]
            all_formants.append(predicted)

    return all_formants, file_names

def load_peaks_from_directory(directory_path):

    all_peaks = []
    file_names = sorted(f for f in os.listdir(directory_path) if not f.startswith('.')) 

    for file_name in file_names:
        if file_name.endswith('.txt') or file_name.endswith('.csv'):
            file_path = os.path.join(directory_path, file_name)
            peaks = pd.read_csv(file_path)
            all_peaks.append(peaks)

    return all_peaks, file_names

def save_refined_formants(refined_formants_list, qcp_peaks, output_directory, file_names):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for refined, peaks_df, file_name in zip(refined_formants_list, qcp_peaks, file_names):

        first_col = peaks_df.iloc[:, 0]
        first_col_name = peaks_df.columns[0]
        
        new_col_names = ["F1", "F2", "F3"]  
        refined_df = pd.DataFrame(refined, columns=new_col_names)
        
        final_df = pd.concat([first_col, refined_df], axis=1)
        
        output_path = os.path.join(output_directory, "Refined "+file_name)
        final_df.to_csv(output_path, index=False)


deep_formant_dir = "./DFEtoCSV"
qcp_peaks_dir = "./QCPEstimation"
output_dir = "./refined_formants"

deep_formants, file_names1 = load_formants_from_directory(deep_formant_dir)
qcp_peaks, file_names2 = load_peaks_from_directory(qcp_peaks_dir)

assert file_names2 == file_names1, "File names in the two directories do not match."

if len(deep_formants) != len(qcp_peaks):
    raise ValueError("Mismatch in number of files between directories.")

refined_formants_list = []

for predicted, peaks in zip(deep_formants, qcp_peaks):
    if predicted.shape[0] != peaks.shape[0]:
        raise ValueError("Mismatch in number of rows between predicted formants and spectral peaks.")
        
    numeric_peaks = peaks.iloc[:, 1:].to_numpy()
    predicted_array = predicted.to_numpy()
    
    refined_formants = refine_formants_row_by_row(predicted_array, numeric_peaks)
    refined_formants_list.append(refined_formants)

save_refined_formants(refined_formants_list, qcp_peaks, output_dir, file_names1)
