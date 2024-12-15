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
from itertools import cycle, islice
import time
from datetime import timedelta

# from models.utils import load_dataset
# from models.BERTquery import MLMBERT


##
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        if labels is not None:
            # punish the model if the model predicts 0,1,2 (= unk, pad, mask)
            logits[:, :, 0:3] = -100 #-float('inf')
    
            logits = logits.view(-1, logits.size(-1))  # Flatten logits to [batch_size * seq_len, vocab_size]
            labels = labels.view(-1) 

            loss = F.cross_entropy(logits, labels)
            return {'loss': loss, 'logits': logits, 'labels': labels}
        
        else:
            mask_idx = (input_ids==self.mask_token_id)
            mask_preds = F.softmax(logits[mask_idx],dim=-1)
            mask_preds = mask_preds.argmax(dim=-1)
            return {'mask_predictions': mask_preds, 'mask_idx': mask_idx}



##
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

model = MLMBERT(config).to(config['device'])
print('     trainable: ',sum([p.numel() for p in model.parameters() if p.requires_grad]))

#%%
# TRAINING 
    # 32 queries through the model in one training step
training_steps = 2000
train_losses = []
valid_losses = []
test_accuracies = []

best_val_loss = float('inf')

train_iter = cycle(train_dl)

batch_logits = []
N = 500
# step = 0
t00 =  time.time()
optim = torch.optim.Adam(model.parameters(), lr=4e-4 / 25.)
sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=4e-4, total_steps=training_steps)
model.train()
tprog = tqdm(range(training_steps), desc="Training Progress")
t0 = time.time()
for step in tprog:
    trl = 0.
    model.train()
    input_ids, labels = next(train_iter)  
    input_ids = input_ids.to(config['device'])
    labels = labels.to(config['device'])
    
    # Forward pass
    return_ = model(input_ids, labels)
    loss = return_['loss']
    loss.backward()
    
    # Optimization step
    optim.step()
    optim.zero_grad()
    sched.step()
    trl += loss.item()
    
    # Log every N steps
    if (step + 1) % N == 0:
        train_loss = trl / N
        train_losses.append(train_loss)
        # Validation phase
        model.eval()
        with torch.no_grad():
            vrl = 0.
            vprog = tqdm(enumerate(val_dl), total=min(len(val_dl), 50), desc = '           validation')
            for i, (input_ids, labels) in vprog:
                input_ids = input_ids.to(config['device'])
                labels = labels.to(config['device'])
                loss = model(input_ids, labels)['loss']
                vrl += loss.item()
                if i == 500:
                    break

            vloss = vrl / len(val_dl)
            valid_losses.append(vloss)
            
            # Save the best model
            if vloss < best_val_loss:
                best_val_loss = vloss
                test_predictions = []
                test_actuals = []

                teprog = tqdm(enumerate(val_dl), total=min(len(val_dl), 50), desc = '           testing')
                for i, (input_ids, labels) in teprog:
                    input_ids = input_ids.to(config['device'])
                    labels = labels.to(config['device'])
                    out_ = model(input_ids)
                    mask_preds = out_['mask_predictions']
                    test_predictions.extend(mask_preds.detach().numpy())             
                    test_actuals.extend(labels[np.where(input_ids==tokenizer.mask_token_id)].detach().numpy())
                    if i == 500:
                        break
                # Calculate accuracy
                test_predictions = [pred.item() for pred in test_predictions]
                test_actuals = [int(pred) for pred in test_actuals]
                tacc = accuracy_score(test_actuals, test_predictions)
                test_accuracies.append(tacc)
                torch.save(model.state_dict(), './mlm-baby-bert/BERT_depth4_embed512_steps2000.pt')
        
        print(
            "          "
            f" --> training loss: {train_loss:.4f}   "
            f"validation loss: {vloss:.4f}   "
            f"accuracy: {tacc*100:.2f}%   "
            f"time: {timedelta(seconds=time.time() - t0)}"
        )

print('Total time: ', {timedelta(seconds=time.time() - t00)} )

#%%
import matplotlib.pyplot as plt

plt.plot(train_losses,color='red',label='train loss')
plt.plot(valid_losses,color='orange',label='valid loss')
plt.legend()
plt.show()

plt.plot(test_accuracies)
plt.title('single mask token prediction accuracy')
plt.show()
# %%
