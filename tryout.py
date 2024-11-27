#%%
""" 
    modified from https://www.kaggle.com/code/shreydan/masked-language-modeling-from-scratch/notebook
"""
#%%

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import transformers
from pathlib import Path
from tqdm.auto import tqdm
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns

data = pd.read_csv("input/all_sequences.csv")
data['Sequence'] = data['Sequence'].str.upper()
data['Sequence'].shape # (156679,)
#%%
# Look at distribution of sequences
data["Letter_Count"] = data["Sequence"].apply(lambda seq: sum(1 for char in seq if char.isalpha() and char != '-'))
expected_value = data["Letter_Count"].mean()

# Plot the distribution
sns.histplot(data["Letter_Count"], kde=True, bins=100)#, color="blue")
plt.vlines(x = expected_value, ymin=0, ymax = 20000, color = 'red')

# Add labels and title
plt.xlabel("Letter Count (Excluding '-')")
plt.ylabel("Frequency")
plt.title("Sequence length")
plt.show()

#%%
# vocab = {'[MASK]': 2,
#  'T': 21,
#  'V': 24,
#  'Q': 10,
#  '1': 25,
#  'S': 20,
#  'D': 6,
#  'C': 8,
#  'E': 9,
#  'M': 17,
#  'W': 22,
#  '[UNK]': 0,
#  '3': 27,
#  'Y': 23,
#  '[PAD]': 1,
#  'B': 7,
#  'K': 16,
#  '2': 26,
#  'F': 18,
#  'I': 14,
#  'Z': 11,
#  'L': 15,
#  'P': 19,
#  'G': 12,
#  'H': 13,
#  'N': 5,
#  'R': 4,
#  'A': 3}

tokenizer = transformers.AutoTokenizer.from_pretrained(
    'mlm-baby-bert/tokenizer/protein_tokenizer',
    use_fast=True,
    unk_token="[UNK]",
    mask_token="[MASK]",
    pad_token="[PAD]"
)
tokenizer.get_vocab()

# tokenizer._convert_id_to_token(8)
# tokenizer.pad_token
# tokenizer.encode(text)

# Define the dataset class
class MLMDataset:
    def __init__(self, sequences):
        """
        Initialize the dataset with sequences (from the 'Sequence' column).
        """
        self.sequences = sequences  # Sequences from the DataFrame

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
        sequence = sequence.lstrip('-').replace('-', '[UNK]')    # Replace '-' with <UNK>
        
        encoded = tokenizer.encode_plus(
            sequence,  # The sequence to tokenize
            add_special_tokens=True,  # Add special tokens like <CLS>, <SEP>, etc.
            truncation=True,  # Truncate if necessary (for long sequences)
            max_length=500,  # Adjust as per your model's context size
            padding="max_length",  # Pad sequences to the max length
            return_tensors="pt",  # Return PyTorch tensors
        )
        ids = encoded["input_ids"].squeeze(0).tolist()  # Token IDs as a list
    
        assert all(0 <= token_id < len(tokenizer.vocab) for token_id in ids), "Invalid token ID found"
    
        if all(token == tokenizer.unk_token_id or token == tokenizer.pad_token_id for token in ids):
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
    # mask 15% of text leaving <PAD> and unkown <UNK>'
    mlm_mask = torch.rand(input_ids.size()) < 0.15 * (input_ids!=tokenizer.pad_token_id) * (input_ids!=tokenizer.unk_token_id) # 0:<PAD>, 26:<UNK>
    masked_tokens = input_ids * mlm_mask # the masked tokens will have value True (==1). We retrieve their original value here
    labels[masked_tokens==0]=-100 # set all tokens except masked tokens to -100
    input_ids[masked_tokens!=0]=tokenizer.mask_token_id # 27:[MASK] 
    return input_ids, labels

sequences = data["Sequence"][:10000]
# print(sequences.shape)

# # Use the 'Sequence' column for the dataset
# dataset = MLMDataset(sequences)
# dl = torch.utils.data.DataLoader(dataset,batch_size=2,shuffle=True,collate_fn=collate_fn)

# Split the data into training and testing sets (90% train, 8% val, 2% test)
train_sequences, val_sequences = train_test_split(sequences, test_size=0.1, random_state=42, shuffle=True)
val_sequences, test_sequences = train_test_split(val_sequences, test_size=0.2, random_state=42, shuffle=True)

print(len(train_sequences), len(val_sequences), len(test_sequences)) # 141011 12534 3134

# Create MLMDataset instances for training, validation and testing
train_dataset = MLMDataset(train_sequences)
val_dataset = MLMDataset(val_sequences)
test_dataset = MLMDataset(test_sequences)

# Create DataLoader for training and testing
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
## CHANGE BATCH_SIZE

# %%

# Encoder-Only Transformer Model
    # decoder == tokenizer

class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed
    
class MultiheadAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        assert dim % n_heads == 0, 'dim should be div by n_heads'
        self.head_dim = self.dim // self.n_heads
        self.in_proj = nn.Linear(dim,dim*3,bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        self.out_proj = nn.Linear(dim,dim)
        
    def forward(self,x,mask=None):
        b,t,c = x.shape
        q,k,v = self.in_proj(x).chunk(3,dim=-1)
        q = q.view(b,t,self.n_heads,self.head_dim).permute(0,2,1,3)
        k = k.view(b,t,self.n_heads,self.head_dim).permute(0,2,1,3)
        v = v.view(b,t,self.n_heads,self.head_dim).permute(0,2,1,3)
        
        qkT = torch.matmul(q,k.transpose(-1,-2)) * self.scale
        qkT = self.attn_dropout(qkT)
        
        if mask is not None:
            mask = mask.to(dtype=qkT.dtype,device=qkT.device)
            qkT = qkT.masked_fill(mask==0,float('-inf'))
              
        qkT = F.softmax(qkT,dim=-1)
        attn = torch.matmul(qkT,v)
        attn = attn.permute(0,2,1,3).contiguous().view(b,t,c)
        out = self.out_proj(attn)
        
        return out
    
class FeedForward(nn.Module):
    def __init__(self,dim,dropout=0.):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(dim,dim*4),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim*4,dim)
        )
        
    def forward(self, x):
        return self.feed_forward(x)
    
class EncoderBlock(nn.Module):
    def __init__(self, dim, n_heads, attn_dropout=0., mlp_dropout=0.):
        super().__init__()
        self.attn = MultiheadAttention(dim,n_heads,attn_dropout)
        self.ffd = FeedForward(dim,mlp_dropout)
        self.ln_1 = RMSNorm(dim)
        self.ln_2 = RMSNorm(dim)
        
    def forward(self,x,mask=None):
        x = self.ln_1(x)
        x = x + self.attn(x,mask)
        x = self.ln_2(x)
        x = x + self.ffd(x)
        return x
    
class Embedding(nn.Module):
    def __init__(self,vocab_size,max_len,dim):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.class_embedding = nn.Embedding(vocab_size,dim)
        self.pos_embedding = nn.Embedding(max_len,dim)
    def forward(self,x):
        # print('x',x)
        # print('Embedding ',self.vocab_size)
        x = self.class_embedding(x)
        pos = torch.arange(0,x.size(1),device=x.device)
        x = x + self.pos_embedding(pos)
        return x
    
class MLMBERT(nn.Module):
    def __init__(self, config):
        
        super().__init__()
        
        self.embedding = Embedding(config['vocab_size'],config['max_len'],config['dim'])
        
        self.depth = config['depth']
        self.encoders = nn.ModuleList([
            EncoderBlock(
                dim=config['dim'],
                n_heads=config['n_heads'],
                attn_dropout=config['attn_dropout'],
                mlp_dropout=config['mlp_dropout']
            ) for _ in range(self.depth)
        ])
        
        self.ln_f = RMSNorm(config['dim'])
        
        self.mlm_head = nn.Linear(config['dim'],config['vocab_size'],bias=False)
        
        self.embedding.class_embedding.weight = self.mlm_head.weight # weight tying
        
        self.pad_token_id = config['pad_token_id']
        self.mask_token_id = config['mask_token_id']
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def create_src_mask(self,src):
        return (src != self.pad_token_id).unsqueeze(1).unsqueeze(2) # N, 1, 1, src_len
    
    def forward(self,input_ids,labels=None):
        
        src_mask = self.create_src_mask(input_ids)
        # print('MLMBERT', input_ids)
        enc_out = self.embedding(input_ids)
        for layer in self.encoders:
            enc_out = layer(enc_out,mask=src_mask)
        
        enc_out = self.ln_f(enc_out)
        
        logits = self.mlm_head(enc_out)
        
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),labels.view(-1))
            return {'loss': loss, 'logits': logits}
        else:
            # assuming inference input_ids only have 1 [MASK] token
            mask_idx = (input_ids==self.mask_token_id).flatten().nonzero().item()
            logits[:, mask_idx, self.mask_token_id] = -float('inf')
            mask_preds = F.softmax(logits[:,mask_idx,:],dim=-1).argmax(dim=-1)
            # print(mask_preds)
            return {'mask_predictions':mask_preds}

    # def forward(self, input_ids, labels=None):
    #     src_mask = self.create_src_mask(input_ids)
    #     enc_out = self.embedding(input_ids)
        
    #     for layer in self.encoders:
    #         enc_out = layer(enc_out, mask=src_mask)
        
    #     enc_out = self.ln_f(enc_out)
    #     logits = self.mlm_head(enc_out)
        
    #     if labels is not None:
    #         # Compute loss for training
    #         loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    #         return {'loss': loss, 'logits': logits}
    #     else:
    #         # Locate all positions of the [MASK] token
    #         mask_indices = (input_ids == self.mask_token_id).nonzero(as_tuple=True)
            
    #         # Gather predictions for all mask positions
    #         mask_logits = logits[mask_indices]  # Shape: (num_masks, vocab_size)
    #         mask_preds = F.softmax(mask_logits, dim=-1).argmax(dim=-1)  # Shape: (num_masks,)
            
    #         return {'mask_predictions': mask_preds, 'mask_indices': mask_indices}

        
config = {
    'dim': 256,
    'n_heads': 8,
    'attn_dropout': 0.1,
    'mlp_dropout': 0.1,
    'depth': 6,
    'vocab_size': len(tokenizer.get_vocab()),
    'max_len': 500,
    'pad_token_id': tokenizer.pad_token_id,
    'mask_token_id': tokenizer.mask_token_id
}

#%%
model = MLMBERT(config).to('cpu')
print('trainable:',sum([p.numel() for p in model.parameters() if p.requires_grad]))
# %%
import torch
print(torch.cuda.is_available()) 

#%%
# TEST : SINGLE TOKEN MASKING
test_sequences.shape

special_tokens = [tokenizer.pad_token_id, tokenizer.unk_token_id, tokenizer.mask_token_id]
test_actuals = []
test_batches = []

for seq in tqdm(test_sequences):
    seq = seq.lstrip('-').replace('-', '[UNK]')[:config["max_len"]] 
    # print('length seq: ',len(seq))
    tokenized_seq = tokenizer.encode(seq)

    # Create a mask: 1 for special tokens, 0 for non-special tokens
    special_tokens_mask = np.array([1 if token in special_tokens else 0 for token in tokenized_seq])

    tokenized_seq_tensor = torch.tensor(tokenized_seq)
    special_tokens_mask_tensor = torch.tensor(special_tokens_mask)

    non_special_token_indices = torch.where(special_tokens_mask_tensor == 0)[0]

    if non_special_token_indices.numel() > 0:
        # Randomly choose a non-special token
        random_index = torch.randint(0, non_special_token_indices.size(0), (1,)).item()
        # print(random_index)
        random_index = non_special_token_indices[random_index]
        test_actuals.append(tokenized_seq_tensor[random_index])
        tokenized_seq_tensor[random_index] = tokenizer.mask_token_id
        test_batches.append(tokenized_seq_tensor)
#%%
# #%%
# tokenizer.get_vocab()
# # print(test_batches[0][220])
# #%%
# print(special_tokens)
# for seq in tqdm(test_sequences):
#     print(seq)
#     seq = seq.replace('-', '<UNK>')
#     tokenized_seq = tokenizer.encode(seq)
#     fi = len(tokenized_seq)
#     special_tokens_mask = np.array([0 if token in special_tokens else 1 for token in tokenized_seq])
#     tokenized_seq_tensor = torch.tensor(tokenized_seq)
#     special_tokens_mask_tensor = torch.tensor(special_tokens_mask)
#     masked_seq = tokenized_seq_tensor * special_tokens_mask_tensor
#     non_special_token_indices = torch.where(special_tokens_mask_tensor == 0)[0]
#     if non_special_token_indices.numel() > 0:
#         m = non_special_token_indices[torch.randint(0, non_special_token_indices.size(0), (1,)).item()]
    
    
#     print("Masked Sequence:", masked_seq)

#     print(tokenized_seq*special_tokens_mask)
#     m = torch.randint(...).item()
#     ###
#     tokenized_seq_tensor = torch.tensor(tokenized_seq)
#     special_tokens_mask_tensor = torch.tensor(special_tokens_mask)

#     # Mask the sequence (element-wise multiplication)
#     masked_seq = tokenized_seq_tensor * special_tokens_mask_tensor
#     print("Masked Sequence:", masked_seq)

#     # Randomly select one index with value 0 in the special_tokens_mask
    
#     break
# #%%
# test_actuals = []
# test_batches = []

# # Use DataLoader for test batching
# for input_ids, labels in tqdm(test_dl):
#     # Here, we will iterate over the test dataset and mask a single token from each sequence

#     for i in range(input_ids.size(0)):  # Loop over each batch element
#         seq = input_ids[i]  # Get the current sequence
        
#         # Ensure the sequence is a tensor and is on the right device (CPU for this case)
#         seq = seq.tolist()
        
#         # Find all tokens that are not padding and not <UNK> (in your case <UNK> is 8)
#         non_pad_tokens = [idx for idx, token in enumerate(seq) if token != tokenizer.pad_token_id and token != tokenizer.unk_token_id]
    
#         # if only padded tokens or UNK, continue to next 
#         # if not len(non_pad_tokens):
#         #     continue
        
#         # Randomly select one token to mask
#         m = random.choice(non_pad_tokens)  # Mask a random token

#         # Store the actual token for evaluation
#         test_actuals.append(seq[m])

#         # Mask the token in the sequence by setting it to the mask token
#         seq[m] = tokenizer.mask_token_id  # Set to [MASK] token id
        
#         # Add the modified sequence to test_batches
#         test_batches.append(torch.tensor(seq))
# print(test_actuals)
# print(test_batches)
# Now, test_batches holds the input sequences with a single token replaced with [MASK]
# test_actuals holds the true value of the token that was masked in each sequence.

# %% TRAINING LOOP
# config

# tokenizer.vocab
# print(f"Max token ID in input_ids: {input_ids.max()}")
# print(f"Vocabulary size: {config.vocab_size}")
#%%
import time
from datetime import timedelta
epochs = 250
train_losses = []
valid_losses = []
test_accuracies = []
best_val_loss = 1e9

optim = torch.optim.Adam(model.parameters(), lr=6e-4 / 25.)
sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=6e-4, steps_per_epoch=len(train_dl), epochs=epochs)

for ep in tqdm(range(epochs)):
    t00 = time.time()
    t0 = time.time()
    model.train()
    trl = 0.
    tprog = tqdm(enumerate(train_dl), total=len(train_dl))
    for i, (input_ids, labels) in tprog:
        input_ids = input_ids.to('cpu')
        labels = labels.to('cpu')
        loss = model(input_ids, labels)['loss']
        loss.backward()
        optim.step()
        optim.zero_grad()
        sched.step()
        trl += loss.item()
        tprog.set_description(f'train step loss: {loss.item():.4f}')
    train_losses.append(trl / len(train_dl))
    print("     Training:", timedelta(seconds=time.time() - t0))
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        vrl = 0.
        vprog = tqdm(enumerate(val_dl), total=len(val_dl))
        for i, (input_ids, labels) in vprog:
            input_ids = input_ids.to('cpu')
            labels = labels.to('cpu')
            loss = model(input_ids, labels)['loss']
            vrl += loss.item()
            vprog.set_description(f'valid step loss: {loss.item():.4f}')
        vloss = vrl / len(val_dl)
        valid_losses.append(vloss)
        print(f'epoch {ep} | train_loss: {train_losses[-1]:.4f} valid_loss: {valid_losses[-1]:.4f}')
        print("     Validation:", timedelta(seconds=time.time() - t0))
        t0 = time.time()
        if vloss < best_val_loss:
            best_val_loss = vloss
            print('PREDICTING!')
            test_predictions = []
            for input_ids in tqdm(test_batches):
                input_ids = input_ids.unsqueeze(0)  # Add batch dimension
                input_ids = input_ids.to('cpu')
                # print(input_ids)
                # ####
                # print("Input IDs:", input_ids)
                # print("Max input ID:", input_ids.max())
                # print("Unique token IDs:", input_ids.unique())
                # print("Input IDs:", input_ids)
                # print("Unique token IDs:", input_ids.unique())

                # print('Token? ', torch.any(input_ids == 27))
                # flattened_input_ids = input_ids.flatten()

                # # Count occurrences of each token ID using bincount
                # token_counts = torch.bincount(flattened_input_ids)

                # # Print the distribution
                # for token_id, count in enumerate(token_counts):
                #     print(f"Token ID {token_id}: {count.item()} occurrences")
                # plt.plot(input_ids.flatten().T, 'o')
                ###
                mask_preds = model(input_ids)['mask_predictions']
                test_predictions.extend(mask_preds.detach().cpu().numpy())
            
            # Calculate accuracy
            test_predictions = [pred.item() for pred in test_predictions]
            # print('accuracy')
            # print(test_actuals, test_predictions)
            # print(len(test_actuals), len(test_predictions))
            tacc = accuracy_score(test_actuals, test_predictions)
            test_accuracies.append(tacc)
            print(f'SINGLE MASK TOKEN PREDICTION ACCURACY: {tacc:.4f}')
            print('saving best model...')
            sd = model.state_dict()
            torch.save(sd, './mlm-baby-bert/model.pt')
            print("     Prediction:", timedelta(seconds=time.time() - t0))

print("     Running the model:", timedelta(seconds=time.time() - t00))

#%%

# %%
plt.plot(train_losses,color='red',label='train loss')
plt.plot(valid_losses,color='orange',label='valid loss')
plt.legend()
plt.show()

plt.plot(test_accuracies)
plt.title('single mask token prediction accuracy')
plt.show()

#%%
# len(test_predictions)
# len(test_actuals)
# tacc

# test_accuracies
# #%%
# len(test_batches)
# for pred in test_predictions:
#     print(pred)
# #%%
# test_predictions = []
# test_accuracies = []
# for input_ids in tqdm(test_batches):
#     input_ids = input_ids.unsqueeze(0)  # Add batch dimension
#     input_ids = input_ids.to('cpu')
#     mask_preds = model(input_ids)['mask_predictions']
#     # print(mask_preds)
#     test_predictions.extend(mask_preds.detach().cpu().numpy())

# # Calculate accuracy
# test_predictions = [pred.item() for pred in test_predictions]
# # print('accuracy')
# # print(test_actuals, test_predictions)
# # print(len(test_actuals), len(test_predictions))
# tacc = accuracy_score(test_actuals, test_predictions)
# test_accuracies.append(tacc)
# print(f'SINGLE MASK TOKEN PREDICTION ACCURACY: {tacc:.4f}')
# print('saving best model...')
# # sd = model.state_dict()
# # torch.save(sd, './mlm-baby-bert/model.pt')
# print("     Prediction:", timedelta(seconds=time.time() - t0))


#%%
# sd = torch.load('./mlm-baby-bert/model.pt')
# model.load_state_dict(sd)

# %%
def predict_mask(sentence):
    # print(sentence)
    sentence = sentence.lstrip('-').replace('-', '[UNK]')[:config["max_len"]] 
    # print('length seq: ',len(seq))
    x = tokenizer.encode(sentence)
    # print('x',x)
    # Create a mask: 1 for special tokens, 0 for non-special tokens
    special_tokens_mask = np.array([1 if token in special_tokens else 0 for token in x])

    # x_tensor = torch.tensor(x)
    special_tokens_mask_tensor = torch.tensor(special_tokens_mask)

    non_special_token_indices = torch.where(special_tokens_mask_tensor == 0)[0]
    if non_special_token_indices.numel() > 0:
        # Randomly choose a non-special token
        idx = torch.randint(0, non_special_token_indices.size(0), (1,)).item()
        idx = non_special_token_indices[idx]

    input_ids = x
    # print(input_ids[idx])
    masked_token = tokenizer.decode([input_ids[idx]])
    # print(masked_token)

    # masking
    input_ids[idx] = 2 # idx -> [MASK]
    masked_sentence = input_ids.copy()
    
    # print(input_ids)
    # preparing input
    input_ids = torch.tensor(input_ids,dtype=torch.long).unsqueeze(0).to('cpu')
    
    # extracting the predicted token
    out = model(input_ids)
    # print(out)
    predicted = x.copy()
    predicted[idx] = out['mask_predictions'].item()
    predicted_token = tokenizer.decode([out['mask_predictions'].item()])
    
    print(f'masked: {masked_token} predicted: {predicted_token}')
    masked_sentence = tokenizer.decode(masked_sentence,skip_special_tokens=False)
    masked_sentence = masked_sentence.replace('[PAD]','')
    print('ACTUAL:',sentence)
    print('MASKED:',masked_sentence)
    print(' MODEL:',tokenizer.decode(predicted))
    
    return int(masked_token == predicted_token)

torch.manual_seed(1420)
correct = 0
train_sequences = train_sequences.reset_index(drop=True)
for sentence in random.choices(train_sequences,k=1000):
    correct += predict_mask(sentence)
    print('\n\n')
print(f'CORRECT:{correct}/{1000}')

#%%
train_sequences = train_sequences.reset_index(drop=True)
for sentence in random.choices(train_sequences,k=1000):
    correct += predict_mask(sentence)
    print('\n\n')
print(f'TRAIN CORRECT:{correct}/{1000}')

for sentence in random.choices(test_sequences,k=1000):
    correct += predict_mask(sentence)
    print('\n\n')
print(f'TEST CORRECT:{correct}/{1000}')
#%%
# train_sequences+test_sequences
# # %%
# idx = 'o'
# sentence = '--------ADY--------JSK'
# sentence = sentence.lstrip('-').replace('-', '[UNK]')[:config["max_len"]] 
#     # print('length seq: ',len(seq))
# x = tokenizer.encode(sentence)
# print(x)
# print(tokenizer.decode(x))

# special_tokens_mask = np.array([1 if token in special_tokens else 0 for token in x])

# # x_tensor = torch.tensor(x)
# special_tokens_mask_tensor = torch.tensor(special_tokens_mask)

# non_special_token_indices = torch.where(special_tokens_mask_tensor == 0)[0]
# print(special_tokens_mask_tensor)
# print(non_special_token_indices)
# if non_special_token_indices.numel() > 0:
#     # Randomly choose a non-special token
#     idx = torch.randint(0, non_special_token_indices.size(0), (1,)).item()
#     idx = non_special_token_indices[idx]
# print(idx)
# # %%
# tokenizer.mask_token_id
# %%
