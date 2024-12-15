import torch
import torch.nn as nn
import pandas as pd
import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained(
    'mlm-baby-bert/tokenizer/protein_tokenizer',
    use_fast=True,
    unk_token="<unk>",
    mask_token="<mask>",
    pad_token="<pad>"
)

def blosum62_matrix():
    """
    Defines and normalizes by column the BLOSUM62 matrix from the blosum62.csv file and returns it as a PyTorch tensor.
    
    Parameters
    ----------
    None 

    Returns
    -------
    torch.Tensor
        BLOSUM62 matrix as a tensor.
    """
    blosum62_df = pd.read_csv('../input/blosum62.csv', delimiter=";", header = None)
    # Normalization by column
    for column in blosum62_df.columns: 
        blosum62_df[column] = (blosum62_df[column] - blosum62_df[column].min()) / (blosum62_df[column].max() - blosum62_df[column].min())     
    return torch.tensor(blosum62_df.to_numpy(), dtype=torch.float32, requires_grad=True)

class BLOSUMLoss(nn.Module):
    """
    Loss function using the BLOSUM62 matrix instead of cross-entropy.
    https://www.labxchange.org/library/items/lb:LabXchange:24d0ec21:lx_image:1 
    """
    def __init__(self, blosum_matrix):
        """
        Initialize the BLOSUM based loss 

        Parameters
        ----------
        blosum_matrix : torch.tensor 
            Square matrix with substitution scores between tokens according to BLOSUM62. 

        Returns
        -------
        None
        """
        super().__init__()
        self.blosum_matrix = blosum_matrix
    
    def forward(self, logits, labels):
        """
        Computes the BLOSUM-based loss.

        Parameters
        ----------
        logits: torch.Tensor (batch_size, seq_length, vocab_size)
            predicted logits
        labels:  torch.Tensor (batch_size, seq_length)
            expected token indices

        Returns
        -------
        torch.Tensor: 
            Loss value according to BLOSUM62 score.
        """
        predicted_indices = logits.argmax(dim=-1)  # dim (batch_size, seq_length)

        blosum_scores = self.blosum_matrix[labels.view(-1)-3, predicted_indices.view(-1)-3] #-3 since <unk>, <mask> and <pad> are not in blosum

        # Convert high scores into low loss
        penalty_loss = 1 - blosum_scores
        return penalty_loss.mean()
