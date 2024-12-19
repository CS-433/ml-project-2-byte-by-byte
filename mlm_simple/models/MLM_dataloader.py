import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class MLMDataset:
    def __init__(self, sequences, tokenizer, max_length=1000):
        """
        Initialize the dataset with sequences (from the 'Sequence' column).
        """
        self.sequences = sequences  
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
        data = self.sequences.iloc[idx,0] 
        sequence = data[0]
        MSAconf = data[1]

        encoded = self.tokenizer.encode_plus(
            sequence,  
            add_special_tokens=True,  
            truncation=True,  
            max_length=self.max_length, 
            padding="max_length", 
            return_tensors="pt", 
        )
        ids = encoded["input_ids"].squeeze(0).tolist()  # Token IDs as a list
    
        assert all(0 <= token_id < len(self.tokenizer.vocab) for token_id in ids), "Invalid token ID found"
    
        # if all(token == self.tokenizer.unk_token_id or token == self.tokenizer.pad_token_id for token in ids):
        #     return None
         
        labels = ids.copy()  
        return ids, MSAconf, labels

def collate_fn(batch):
    if not batch:
        print("Empty batch received!")

    batch = [item for item in batch if item is not None]

    # Separate the input_ids, MSAconf, and labels from the batch
    input_ids = [torch.tensor(i[0]) for i in batch]  
    msa_confs = [torch.tensor(i[1]) for i in batch]  
    labels = [torch.tensor(i[2]) for i in batch]  

    max_length = len(input_ids[0])

    # Padding the sequences to max_length
    input_ids = [torch.cat([ids, torch.zeros(max_length - len(ids))]) if len(ids) < max_length else ids[:max_length] for ids in input_ids]
    msa_confs = [torch.cat([conf, torch.zeros(max_length - len(conf))]) if len(conf) < max_length else conf[:max_length] for conf in msa_confs]
    labels = [torch.cat([lbl, torch.zeros(max_length - len(lbl))]) if len(lbl) < max_length else lbl[:max_length] for lbl in labels]

    # Stack the tensors to form a batch
    input_ids = torch.stack(input_ids)
    msa_confs = torch.stack(msa_confs)
    labels = torch.stack(labels)

    # Apply MLM masking (15% of tokens are masked)
    mlm_mask = torch.rand(input_ids.size()) < 0.15 * (input_ids != 0) * (input_ids != 1)  
    masked_tokens = input_ids * mlm_mask  
    labels[masked_tokens == 0] = -100  
    input_ids[masked_tokens != 0] = 2 

    return input_ids, msa_confs, labels

def load_dataset(input_path, tokenizer, config):
    batch_size = config['batch_size']
    max_len = config['max_len']
    if "queries" in input_path:
        data = pd.read_csv(input_path)
        data = data[data['Header'].str.contains("query")]
        data['Sequence'] = data['Sequence'].str.upper()
        df_sequences = data["Sequence"][:-1000]  # we leave the 100 first ones for testing
        df_sequences = pd.DataFrame(df_sequences.apply(lambda x: [x, np.ones(len(x))]))
    else:
        data = pd.read_csv('../input/all_sequences.csv')

        ## For all  MSA, create a single string with most present amino acid and certainty
        grouped_dfs = data.groupby(data['Header'].str[:6], sort=False)
        df_sequences=pd.DataFrame(columns=['MSA'])
        for _, group in grouped_dfs:
            MSA = group.Sequence
            MSA = MSA.apply(lambda x: pd.Series(list(x)))
            MSA = MSA.map(lambda x: np.nan if x == '-' else x)
            MSA = MSA.dropna(axis = 1, how='all')
            mode_values = []
            mode_counts = []
            for col in MSA.columns:
            # Get the mode (most common value)
                non_nan_count = MSA[col].count()
                mode_value = MSA[col].mode().iloc[0]  # mode() returns a series, so we use iloc to get the first mode value
                # Count the occurrences of the mode
                mode_count = MSA[col].value_counts().get(mode_value, 0)
                
                # Append the results to the lists
                mode_values.append(mode_value)
                mode_counts.append(mode_count/non_nan_count)

            #  Create a DataFrame to show the results
            df_sequences.loc[_] = [[''.join(mode_values)] + [mode_counts]]   

    # Split the data into training and testing sets (97% train, % val, % test)
    train_sequences, val_sequences = train_test_split(df_sequences, test_size=0.3, random_state=42, shuffle=True)
    val_sequences, test_sequences = train_test_split(val_sequences, test_size=0.5, random_state=42, shuffle=True)

    print(f'     Train set: {train_sequences.size}\n     Validation set: {val_sequences.size}\n     Test set: {test_sequences.size}')
    # Create MLMDataset instances for training, validation and testing
    train_dataset = MLMDataset(train_sequences, tokenizer, max_len)
    val_dataset = MLMDataset(val_sequences, tokenizer, max_len)
    test_dataset = MLMDataset(test_sequences, tokenizer, max_len)

    # Create DataLoader for training and testing
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_dl, val_dl, test_dl
