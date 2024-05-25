import os
from typing import Text
import pandas as pd

def split_csv(
    file_path: Text
    ) -> None:
    """
    Splits a CSV file into multiple smaller files, each containing 1000 lines.
    The smaller files are saved in the same directory as the original file.

    Args:
        file_path (str): The path to the original CSV file.

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
        Exception: For any other exceptions that may occur during file operations.
    """
    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    # Read the original CSV file
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"An error occurred while reading the file: {e}")

    # Determine the number of files needed
    num_files = len(df) // 1000 + (1 if len(df) % 1000 != 0 else 0)
    
    # Extract the directory and base name of the original file
    base_dir = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Split the dataframe and save smaller files
    for i in range(num_files):
        start_row = i * 1000
        end_row = start_row + 1000
        chunk_df = df[start_row:end_row]
        
        # Define the output file name
        output_file = os.path.join(base_dir, f"{base_name}_part_{i + 1}.csv")
        
        # Save the chunk to a new CSV file
        try:
            chunk_df.to_csv(output_file, index=False)
            print(f"Saved file: {output_file}")
        except Exception as e:
            raise Exception(f"An error occurred while saving the file {output_file}: {e}")

if __name__ == "__main__":
    split_csv('src/text_cls/dataset/VNTC/train.csv')
