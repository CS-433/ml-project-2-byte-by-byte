#%%
""" 
    modified from https://www.kaggle.com/code/shreydan/masked-language-modeling-from-scratch/notebook
"""
#%% import libraries
import torch
import numpy as np
import transformers
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
from itertools import cycle
import time
from datetime import timedelta

from models.utils import load_dataset
from models.BERTquery import MLMBERT

#%%
print(f'cuda is {'available' if torch.cuda.is_available() else 'not available'}') 

#%% load tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
        'mlm-baby-bert/tokenizer/protein_tokenizer',
        use_fast=True,
        unk_token="<unk>",
        mask_token="<mask>",
        pad_token="<pad>"
    )

#%%
config = {
    'batch_size' : 32,
    'dim': 512,
    'n_heads': 8,
    'attn_dropout': 0.1,
    'mlp_dropout': 0.1,
    'depth': 4,
    'vocab_size': len(tokenizer.get_vocab()),
    'max_len': 1000,
    'pad_token_id': tokenizer.pad_token_id,
    'mask_token_id': tokenizer.mask_token_id,
    # 'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    'device': 'cpu'
}

train_dl, val_dl, test_dl = load_dataset("../input/all_queries.csv", tokenizer, config['batch_size'])

# define the model architecture
model = MLMBERT(config).to(config['device'])
print('trainable:',sum([p.numel() for p in model.parameters() if p.requires_grad]))

# load the model parameters
state_dict = torch.load('./mlm-baby-bert/BERT_depth4_embed512_steps2000.pt')
model.load_state_dict(state_dict, strict=False)

#%% calculate test accuracy
test_predictions = []
test_actuals = []
i = 0
for input_ids, labels in test_dl:
    input_ids = input_ids.to(config['device'])
    labels = labels.to(config['device'])
    out_ = model(input_ids)
    mask_preds = out_['mask_predictions']
    test_predictions.extend(mask_preds.detach().numpy())             
    test_actuals.extend(labels[np.where(input_ids==tokenizer.mask_token_id)].detach().numpy())
    if i == 5:
        break
    i += 1

# Calculate accuracy
test_predictions = [pred.item() for pred in test_predictions]
test_actuals = [int(pred) for pred in test_actuals]
tacc = accuracy_score(test_actuals, test_predictions)
        
#%%
print(tacc)
#%% tryout on 
import pandas as pd
from sklearn.model_selection import train_test_split
import random

data = pd.read_csv('../input/all_queries.csv')
data = data[data['Header'].str.contains("query")]
data['Sequence'] = data['Sequence'].str.upper()
sequences = data["Sequence"]

# Split the data into training and testing sets (97% train, % val, % test)
train_sequences, val_sequences = train_test_split(sequences, test_size=0.1, random_state=42, shuffle=True)
val_sequences, test_sequences = train_test_split(val_sequences, test_size=0.2, random_state=42, shuffle=True)

special_tokens = [tokenizer.pad_token_id, tokenizer.unk_token_id, tokenizer.mask_token_id]

def predict_mask(sentence):
    sentence = sentence.lstrip('-').replace('-', '<unk>')[:config["max_len"]] 
    x = tokenizer.encode(
        sentence,  # The sequence to tokenize
        add_special_tokens=True,  # Add special tokens like <CLS>, <SEP>, etc.
        truncation=True,  # Truncate if necessary (for long sequences)
        max_length=config['max_len'],  # Adjust as per your model's context size
        padding="max_length",  # Pad sequences to the max length
        return_tensors="pt",  # Return PyTorch tensors
        )[0]
    # Create a mask: 1 for special tokens, 0 for non-special tokens
    special_tokens_mask = np.array([1 if token in special_tokens else 0 for token in x])
    special_tokens_mask_tensor = torch.tensor(special_tokens_mask)

    non_special_token_indices = torch.where(special_tokens_mask_tensor == 0)[0]
    if non_special_token_indices.numel() > 0:
        # Randomly choose a non-special token
        # idx = torch.randint(0, non_special_token_indices.size(0), (1,)).item()
        idx = torch.randint(0, non_special_token_indices.size(0), (5,))#.item() # choose the amount of values to be masked, eg. 5
        idx = idx.unique()
        idx = non_special_token_indices[idx]
    else:
        return False
    input_ids = x
    predicted = x.clone().detach()
    masked_ID = input_ids[idx] # eg. 20
    # print(masked_ID)
    masked_token = tokenizer.decode(masked_ID) # true value, eg. W
    # print(masked_token)
    # masking
    input_ids[idx] = 2 # idx -> [MASK]
    masked_sentence = input_ids.clone().detach()
    # preparing input
    input_ids = input_ids.clone().detach().unsqueeze(0).to('cpu')
    
    # extracting the predicted token
    out = model(input_ids)
    ##
    import matplotlib.pyplot as plt
    logitz = out['logits']
    logitz = logitz.view(-1, 1000, 23).clone().detach().numpy()
    j=0
    IDS = input_ids[j]==2
    for i in range(3,23):
        plt.plot(logitz[j,IDS,i])

    for i in range(0,3):
    # i = 2
        plt.plot(logitz[j,IDS, i], color= 'red', label=i)
    plt.show()
    ##
    predicted[idx] = out['mask_predictions']#.item()
    
    predicted_token = tokenizer.decode(out['mask_predictions'])
    print(f'masked: {masked_token} predicted: {predicted_token}')
    masked_sentence = tokenizer.decode(masked_sentence,skip_special_tokens=False)
    masked_sentence = masked_sentence.replace('<pad>','')
    masked_sentence = masked_sentence.replace('<mask>','x')
    masked_place = ''.join('^' if char == 'x' else '-' for char in masked_sentence)
    print('ACTUAL:',sentence)
    print('MASKED:',masked_sentence)
    print(' MODEL:',tokenizer.decode(predicted, skip_special_tokens=True))
    print('      :',masked_place)
    similar = masked_ID == out['mask_predictions']
    total_c = similar.sum().item()
    total_pred = out['mask_predictions'].size()[0]
    return total_c, total_pred

torch.manual_seed(1420)
correct = 0
# train_sequences = train_sequences.reset_index(drop=True)
# test_sequences = test_sequences.reset_index(drop=True)
combined_sequences = pd.concat([train_sequences, val_sequences, test_sequences]).reset_index(drop=True)

total_correct = 0
total_predicted = 0
for sentence in random.choices(combined_sequences,k=1):
    total_c, total_pred = predict_mask(sentence)
    total_correct += total_c
    total_predicted += total_pred
print(f'CORRECT:{total_correct}/{total_predicted}')

# %%
combined_sequences

# %%

# %%
35/491
# %%
1/23
# %%
