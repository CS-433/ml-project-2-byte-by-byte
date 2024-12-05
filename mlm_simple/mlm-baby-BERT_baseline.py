#%%
""" 
    modified from https://www.kaggle.com/code/shreydan/masked-language-modeling-from-scratch/notebook
"""
#%%

import torch
from torch.utils.data import Subset
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

import time
from datetime import timedelta

print(torch.cuda.is_available()) 

data = pd.read_csv("../input/all_sequences.csv")
data = data[data['Header'].str.contains("query")]
data['Sequence'] = data['Sequence'].str.upper()
data['Sequence'].shape # (100,)

# # Look at distribution of sequences
# data["Letter_Count"] = data["Sequence"].apply(lambda seq: sum(1 for char in seq if char.isalpha() and char != '-'))
# expected_value = data["Letter_Count"].mean()

# # Plot the distribution
# sns.histplot(data["Letter_Count"], kde=True, bins=100)#, color="blue")
# plt.vlines(x = expected_value, ymin=0, ymax = 11, color = 'red')

# # Add labels and title
# plt.xlabel("Letter Count (Excluding '-')")
# plt.ylabel("Frequency")
# plt.title("Sequence length")
# plt.show()
#%%

tokenizer = transformers.AutoTokenizer.from_pretrained(
    'mlm-baby-bert/tokenizer/protein_tokenizer',
    use_fast=True,
    unk_token="<unk>",
    mask_token="<mask>",
    pad_token="<pad>"
)

# tokenizer._convert_id_to_token(8)
# tokenizer.pad_token
# tokenizer.encode(text)

config = {
    'batch_size' : 32,
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
            max_length=config['max_len'],  # Adjust as per your model's context size
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
    # mask 15% of text leaving <PAD> and unkown <UNK>
    mlm_mask = torch.rand(input_ids.size()) < 0.15 * (input_ids!=tokenizer.pad_token_id) * (input_ids!=tokenizer.unk_token_id) # 0:<PAD>, 26:<UNK>
    masked_tokens = input_ids * mlm_mask # the masked tokens will have value True (==1). We retrieve their original value here
    labels[masked_tokens==0]=-100 # set all tokens except masked tokens to -100
    input_ids[masked_tokens!=0]=tokenizer.mask_token_id # 27:[MASK] 
    return input_ids, labels

sequences = data["Sequence"]

# Split the data into training and testing sets (70% train, 15% val, 15% test)
train_sequences, val_sequences = train_test_split(sequences, test_size=0.7, random_state=42, shuffle=True)
val_sequences, test_sequences = train_test_split(val_sequences, test_size=0.5, random_state=42, shuffle=True)

print(len(train_sequences), len(val_sequences), len(test_sequences)) # 141011 12534 3134

# Create MLMDataset instances for training, validation and testing
train_dataset = MLMDataset(train_sequences)
val_dataset = MLMDataset(val_sequences)
test_dataset = MLMDataset(test_sequences)

# Create DataLoader for training and testing
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

#%%
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
        
        src_mask = self.create_src_mask(input_ids) # non-padding tokens are marked as true
        
        enc_out = self.embedding(input_ids)
        for layer in self.encoders:
            enc_out = layer(enc_out,mask=src_mask)
        
        enc_out = self.ln_f(enc_out)
        
        logits = self.mlm_head(enc_out)

        # preventing the model to predict 0,1,2 = unk, pad, mask
        logits[:, :, 0:3] = -float('inf')
        
        if labels is not None:
            logits = logits.view(-1, logits.size(-1))  # Flatten logits to [batch_size * seq_len, vocab_size]

            # we only look at the values that are masked and compare those
            mask = (labels != -100)
            masked_logits = logits[mask.view(-1)]

            labels = labels.view(-1)  
            masked_labels = labels[mask.view(-1)]
            
            loss = F.cross_entropy(masked_logits, masked_labels)
            return {'loss': loss, 'logits': masked_logits, 'labels': masked_labels}
        
        else:
            ## for multiple masks
            # ## F.softmax(logits, dim=-1).argmax(dim=-1) returns the predicted tokens for the masks

            # assuming inference input_ids only have 1 [MASK] token
            mask_idx = (input_ids==self.mask_token_id).flatten().nonzero().item()
            # logits[:, mask_idx, self.mask_token_id] = -float('inf')
            mask_preds = F.softmax(logits[:,mask_idx,:],dim=-1).argmax(dim=-1)
            # print(mask_preds)
            return {'mask_predictions': mask_preds, 'mask_idx': mask_idx}

#%%
model = MLMBERT(config).to('cpu')
print('trainable:',sum([p.numel() for p in model.parameters() if p.requires_grad]))

#%%
# TEST : SINGLE TOKEN MASKING

special_tokens = [tokenizer.pad_token_id, tokenizer.unk_token_id, tokenizer.mask_token_id]
test_actuals = []
test_batches = []
for seq in tqdm(test_sequences):
    seq = seq.lstrip('-').replace('-', '<unk>')[:config["max_len"]] 
    tokenized_seq = tokenizer.encode(seq)
    # Create a mask: 1 for special tokens, 0 for non-special tokens
    special_tokens_mask = np.array([1 if token in special_tokens else 0 for token in tokenized_seq], dtype=np.int32)
    non_special_token_indices = np.where(special_tokens_mask == 0)[0]
    if non_special_token_indices.size > 0:
        # Randomly choose a non-special token
        random_index = np.random.randint(0, non_special_token_indices.size)
        random_index = non_special_token_indices[random_index]
        if tokenized_seq[random_index] in [0,1,2]:
            print('WRONG')
        test_actuals.append(tokenized_seq[random_index])
        tokenized_seq[random_index] = tokenizer.mask_token_id
        test_batches.append(torch.tensor(np.copy(tokenized_seq)))

#%%
# TRAINING
epochs = 400
train_losses = []
valid_losses = []
test_accuracies = []
best_val_loss = 1e9

batch_logits = []

optim = torch.optim.Adam(model.parameters(), lr=6e-4 / 25.)
sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=6e-4, steps_per_epoch=len(train_dl), epochs=epochs)

# state_dict = torch.load('./mlm-baby-bert/model_100epoch_100seq.pt')
# model.load_state_dict(state_dict, strict=False)

for ep in tqdm(range(epochs), disable = True):
    t0 = time.time()

    # no dynamical update necessary - train_set: 80 sequences

    model.train()
    trl = 0.
    tprog = tqdm(enumerate(train_dl), total=len(train_dl), desc = f'Epoch {ep}: training')
    for i, (input_ids, labels) in tprog:
        input_ids = input_ids.to('cpu')
        labels = labels.to('cpu')
        return_= model(input_ids, labels)
        loss = return_['loss']
        
        # if you want to look at the distribution of the predicted logits_
        # logits_ = return_['logits']
        # labels_ = return_['labels']
        # batch_logits.append(return_['logits'].detach().cpu().numpy())
        
        loss.backward()
        optim.step()
        optim.zero_grad()
        sched.step()
        trl += loss.item()
    train_loss = trl / len(train_dl)
    train_losses.append(train_loss)
    
    model.eval()
    with torch.no_grad():
        vrl = 0.
        vprog = tqdm(enumerate(val_dl), total=len(val_dl), desc = '           validation')
        for i, (input_ids, labels) in vprog:
            input_ids = input_ids.to('cpu')
            labels = labels.to('cpu')
            loss = model(input_ids, labels)['loss']
            vrl += loss.item()
        vloss = vrl / len(val_dl)
        valid_losses.append(vloss)
        if vloss < best_val_loss:
            # best_val_loss = vloss
            test_predictions = []
            for input_ids in tqdm(test_batches, desc= '           testing'):
                input_ids = input_ids.unsqueeze(0)  # Add batch dimension
                input_ids = input_ids.to('cpu')
                mask_preds = model(input_ids)['mask_predictions']
                test_predictions.extend(mask_preds.detach().numpy())
            # Calculate accuracy
            test_predictions = [pred.item() for pred in test_predictions]
            tacc = accuracy_score(test_actuals, test_predictions)
            test_accuracies.append(tacc)
            sd = model.state_dict()
            torch.save(sd, './mlm-baby-bert/model_500epoch_100seq_bis.pt')
            
        print(
            "          "
            f" --> training loss: {train_loss:.4f}   "
            f"validation loss: {vloss:.4f}   "
            f"accuracy: {tacc*100:.2f}%   "
            f"time: {timedelta(seconds=time.time() - t0)}"
        )

# %%
plt.plot(train_losses,color='red',label='train loss')
plt.plot(valid_losses,color='orange',label='valid loss')
plt.legend()
plt.show()

plt.plot(test_accuracies)
plt.title('single mask token prediction accuracy')
plt.show()

# plt.plot(np.histogram(np.array(test_predictions)))
#%%

test_accuracies
#%%
# EVALUATION

# validation set
val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# choose pretrained model
state_dict = torch.load('./mlm-baby-bert/model_500epoch_100seq.pt')
model.load_state_dict(state_dict, strict=False)

# evaluate
valid_losses = []
test_accuracies = []
best_val_loss = 1e9
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
    test_predictions = []
    for input_ids in tqdm(test_batches):
        # print(input_ids.shape)
        input_ids = input_ids.unsqueeze(0)  # Add batch dimension
        input_ids = input_ids.to('cpu')
        mask_preds = model(input_ids)['mask_predictions']
        test_predictions.extend(mask_preds.detach().cpu().numpy())
    # Calculate accuracy
    test_predictions = [pred.item() for pred in test_predictions]
    # print(test_actuals)
    # print(test_predictions)
    tacc = accuracy_score(test_actuals, test_predictions)
    test_accuracies.append(tacc)
print(
        f"validation loss: {vloss:.4f}    "
        f"accuracy: {tacc*100:.2f}%"
    )
# %%
def predict_mask(sentence):
    # print(sentence)
    sentence = sentence.lstrip('-').replace('-', '<UNK>')[:config["max_len"]] 
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

# torch.manual_seed(1420)
# correct = 0
# train_sequences = train_sequences.reset_index(drop=True)
# for sentence in random.choices(train_sequences,k=1000):
#     correct += predict_mask(sentence)
#     print('\n\n')
# print(f'CORRECT:{correct}/{1000}')

#%%
correct = 0
train_sequences = train_sequences.reset_index(drop=True)
for sentence in random.choices(train_sequences,k=1000):
    correct += predict_mask(sentence)
    print('\n\n')
print(f'TRAIN CORRECT:{correct}/{1000}')
correct = 0
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
