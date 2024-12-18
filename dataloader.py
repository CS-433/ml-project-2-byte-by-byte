import csv
import pandas as pd
import glob
import os

def parse_a3m(file_path):
    """
    Parses an A3M file and extracts sequences.

    Parameters
    ----------
    file_path : str
        Path to the A3M file.

    Returns
    -------
    dict
        Dictionary where keys are sequence headers and values are the corresponding 
        sequences from the file.
    """
    sequences = {}
    with open(file_path, "r") as file:
        header = None
        for line in file:
            line = line.strip()         # Remove leading and trailing white space
            if line.startswith(">"):    # Header detection
                header = line[1:]       # Remove '>'
                sequences[header] = ""
            elif header:
                sequences[header] += line
    return sequences

def save_to_csv(sequences, output_path):
    """
    Saves sequences to a csv file.

    Parameters
    ----------
    sequences: dict
        Protein sequences with headers as keys and sequences as values.

    output_path : str
        Path for the desired output file.

    Returns
    -------
    None
    """
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Header", "Sequence"]) 
        for header, sequence in sequences.items():
            writer.writerow([header, sequence]) 


def load_a3m_files(root_dir, output_csv):
    """
    Load A3M files from all proteins from a directory and save the sequences to a CSV file.

    Parameters
    ----------
    root_dir : str
        Root directory where protein subdirectories and A3M files are located.
    output_csv : str
        Path to save the CSV file with sequences.

    Returns
    -------
    None
    """
    all_protein_sequences = {}
    
    # Gets to the different protein folders
    for protein_dir in glob.glob(os.path.join(root_dir, '*'), recursive=False):
        if os.path.isdir(protein_dir):  # Check if it's a subdirectory
            protein_name = os.path.basename(protein_dir) # Assign the name of the dir to the sequences
            protein_sequences = {}

            # Find all A3M files in the subdirectory
            for file_path in glob.glob(os.path.join(protein_dir, '*.a3m')):
                sequences = parse_a3m(file_path)  
                protein_sequences.update(sequences)  # Add sequences to the protein's group

        all_protein_sequences[protein_name] = protein_sequences  # Store sequences grouped by protein

    # Save to CSV file
    save_to_csv(all_protein_sequences, output_csv)


load_a3m_files("~/openfold","all_sequences")

pd.read_csv("all_sequences")