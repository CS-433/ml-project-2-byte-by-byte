import pandas as pd
import numpy as np
import torch 

def mask_column_MSA_prot(prot_sequences, index):
    """
    Masks a chosen residue of all aligned sequences of a protein. 

    Parameters
    ----------
    prot_sequences : DataFrame
        DataFrame of the sequences of the protein in which we mask a column.
    index:
        Position to mask.

    Returns
    -------
    DataFrame
        DataFrame with the masked position for all sequences.
    """
    if index < len(prot_sequences['Sequence'].iloc[0]):
        prot_sequences['Sequence'] = prot_sequences['Sequence'].apply(lambda seq: seq[:index] + '<mask>' + seq[index+1:])
    else: print('The index indicated does not exist in these sequences')

    return prot_sequences


def mask_column_MSA(input_data):
    """
    Masks a randomly chosen position in all aligned sequences of the proteins in the given file. 

    Parameters
    ----------
    input_data : str or DataFrame
        Either the file path of the proteins MSA file to mask, or a DataFrame containing the MSA.

    Returns
    -------
    DataFrame
        DataFrame of the MSA with the randomly chosen masked position for each protein.
    """
    if isinstance(input_data, str):
        df = pd.read_csv(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data
    else:
        raise ValueError("Input must be a file path (str) or a DataFrame.")

    # Group by the first 6 characters of the "Header", the anme of the protein
    grouped_dfs = df.groupby(df['Header'].str[:6], sort=False)

    df_new=pd.DataFrame()
    for _, group in grouped_dfs:
        index = np.random.randint(len(group['Sequence'].iloc[0]))
        prot_df=mask_column_MSA_prot(group, index)
        df_new=pd.concat([df_new, prot_df], ignore_index=True)

    return df_new


def mask(sequence, masked_index):
    """
    Replaces a character at the specified index with a mask token ('X').
    
    Parameters:
        sequence (str): The original sequence.
        masked_index (int): The index of the character to mask.

    Returns:
        list of tuples: Each tuple is a sequence identifier and sequence (required by batch_converter).
    """
    if isinstance(masked_index, int):
        return sequence[:masked_index] + '<mask>' + sequence[masked_index + 1:]
    elif isinstance(masked_index, list):
        masked_sequence=sequence
        for i in masked_index:
            masked_sequence = masked_sequence[:i] + 'X' + masked_sequence[i + 1:]
        return masked_sequence.replace('X', '<mask>')


def mask_MSA(sequences, masked_index):
    """
    Replaces a character at the specified index with a mask token ('X').
    
    Parameters:
        sequence (list): MSA.
        masked_index (int): The index of the character to mask.

    Returns:
        sequence with mask.
    """
    return [sequence[:masked_index] + '<mask>' + sequence[masked_index + 1:] for sequence in sequences]
    