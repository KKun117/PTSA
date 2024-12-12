import os
import pandas as pd
import numpy as np
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

def load_ground_truth_from_directory(directory_path):
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
            predicted = formants.iloc[:, 2:5]
            all_formants.append(predicted)

    return all_formants, file_names


deep_formant_dir = "./DFEtoCSV"
refined_formant_dir = "./refined_formants"
ground_truth_dir = "./dataset/csv"
deep_formants, _ = load_formants_from_directory(deep_formant_dir)
refined_formants, _ = load_formants_from_directory(refined_formant_dir)
ground_truth, _ = load_ground_truth_from_directory(ground_truth_dir)


def FEE(deep_formants, refined_formants):
    """
    Computes the sum of absolute differences between deep formants and refined formants.
    Assumes both are lists of DataFrames with aligned rows and columns.
    """
    total_abs_diff = 0
    total_formant_count = 0

    for deep, refined in zip(deep_formants, refined_formants):
        if deep.shape != refined.shape:
            raise ValueError("Mismatch in shape between deep formants and refined formants.")
        
        # Compute the absolute difference and sum it
        abs_diff = np.abs(deep.values - refined.values).sum()
        total_abs_diff += abs_diff
        total_formant_count += deep.size
    

    FEE = total_abs_diff/total_formant_count
    return FEE, total_formant_count




def FEE_row_by_row(deep_formants, refined_formants):
    """
    Computes the sum of absolute differences between deep formants and refined formants row by row.
    Assumes both are lists of DataFrames with aligned rows and columns.

    Parameters:
    - deep_formants: List of DataFrames representing deep formants.
    - refined_formants: List of DataFrames representing refined formants.

    Returns:
    - total_abs_diff: The total sum of absolute differences.
    - row_differences: List of row-wise absolute differences.
    """
    total_abs_diff = 0
    row_differences = []

    for deep, refined in zip(deep_formants, refined_formants):
        if deep.shape != refined.shape:
            raise ValueError("Mismatch in shape between deep formants and refined formants.")
        row = []
        # Compute row-by-row absolute differences
        for row_idx in range(deep.shape[0]):
            row_diff = np.abs(deep.iloc[row_idx].values - refined.iloc[row_idx].values).sum()
            row.append(row_diff)
            total_abs_diff += row_diff
        row_differences.append(row)

    return total_abs_diff, row_differences


_,total_formant_count = FEE(deep_formants, ground_truth)
# refinedFEE = FEE(refined_formants, ground_truth)
# print("Deep formant FEE: ",dfFEE)
# print("Refined FEE: ",refinedFEE)

dfFEE, df_row = FEE_row_by_row(deep_formants, ground_truth)
refinedFEE, refined_row = FEE_row_by_row(refined_formants, ground_truth)
row_difference = np.array(df_row) - np.array(refined_row)
row_difference = pd.DataFrame(pd.DataFrame(row_difference).values/total_formant_count)


print("dfFEE: ",dfFEE)
print(refinedFEE)

#row_difference = row_difference.drop(index=1)

print(row_difference)
column_sums = row_difference.sum(axis=1)

print("Sum of each column:")
print(column_sums)

print("Error Difference: ",row_difference.values.sum())
sum_less_than_1000 = row_difference.values[row_difference.values < 5].sum()
print("Error Difference without Outlier: ",sum_less_than_1000)



