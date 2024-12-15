import torch
import pandas as pd
from sklearn.model_selection import train_test_split

class MLMDataset:
    def __init__(self, sequences, tokenizer, max_length=1000):
        """
        Initialize the dataset with sequences (from the 'Sequence' column).
        """
        self.sequences = sequences  # Sequences from the DataFrame
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        """
        Return the number of sequences in the dataset.
        """
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Retrieve the token IDs and corresponding labels for the sequence at the given index.
        """
        sequence = self.sequences.iloc[idx]  # Retrieve the sequence from the Series
        # sequence = sequence.lstrip('-').replace('-', '[UNK]')    # Replace '-' with <UNK>
        
        encoded = self.tokenizer.encode_plus(
            sequence,  # The sequence to tokenize
            add_special_tokens=True,  # Add special tokens like <CLS>, <SEP>, etc.
            truncation=True,  # Truncate if necessary (for long sequences)
            max_length=self.max_length,  # Adjust as per your model's context size
            padding="max_length",  # Pad sequences to the max length
            return_tensors="pt",  # Return PyTorch tensors
        )
        ids = encoded["input_ids"].squeeze(0).tolist()  # Token IDs as a list
    
        assert all(0 <= token_id < len(self.tokenizer.vocab) for token_id in ids), "Invalid token ID found"
    
        if all(token == self.tokenizer.unk_token_id or token == self.tokenizer.pad_token_id for token in ids):
            return None
         
        labels = ids.copy()  # Copy the token IDs to use as labels
        return ids, labels

def collate_fn(batch):
    if not batch:
        print("Empty batch received!")

    batch = [item for item in batch if item is not None]

    input_ids = [torch.tensor(i[0]) for i in batch]
    labels = [torch.tensor(i[1]) for i in batch]
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)
    # mask 15% of text leaving <pad> and unkown <unk>
    mlm_mask = torch.rand(input_ids.size()) < 0.15 * (input_ids!=0) * (input_ids!=1) # 0:<pad>, 1:<unk>
    masked_tokens = input_ids * mlm_mask # the masked tokens will have value True (==1). We retrieve their original value here
    # labels[masked_tokens==0]=-100 # set all tokens except masked tokens to -100
    input_ids[masked_tokens!=0]=2 # 2: <mask>
    return input_ids, labels

def load_dataset(input_path, tokenizer, batch_size):
    data = pd.read_csv(input_path)
    data = data[data['Header'].str.contains("query")]
    data['Sequence'] = data['Sequence'].str.upper()
    sequences = data["Sequence"]
    
    # Split the data into training and testing sets (97% train, % val, % test)
    train_sequences, val_sequences = train_test_split(sequences, test_size=0.1, random_state=42, shuffle=True)
    val_sequences, test_sequences = train_test_split(val_sequences, test_size=0.2, random_state=42, shuffle=True)

    print(f'     Train set: {train_sequences.size}\n     Validation set: {val_sequences.size}\n     Test set: {test_sequences.size}')
    # Create MLMDataset instances for training, validation and testing
    train_dataset = MLMDataset(train_sequences, tokenizer)
    val_dataset = MLMDataset(val_sequences, tokenizer)
    test_dataset = MLMDataset(test_sequences, tokenizer)

    # Create DataLoader for training and testing
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_dl, val_dl, test_dl