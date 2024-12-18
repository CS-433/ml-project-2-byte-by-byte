#%% import libraries
import torch
import numpy as np
import transformers
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
from itertools import cycle, islice
import time
from datetime import timedelta

## CHANGE TO THIS AFTER TRAINING EVERYTHING
# from models.utils import load_dataset
# from models.pBERT import MLMBERT

from models.MLM_dataloader import load_dataset
from models.pBERT_final import MLMBERT

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
    'device': 'cpu',
    # 'loss':'BLOSUM' # uncomment if you want to use BLOSUM loss
}

# load the data
train_dl, val_dl, test_dl = load_dataset("../input/all_queries.csv", tokenizer, config)

# define the model architecture
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
N = 500# after how many training steps you want to evaluate the model
t00 =  time.time()
optim = torch.optim.Adam(model.parameters(), lr=4e-4 / 25.)
sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=4e-4, total_steps=training_steps)
model.train()
tprog = tqdm(range(training_steps), desc="Training Progress")
t0 = time.time()
for step in tprog:
    trl = 0.
    model.train()
    input_ids, MSAconf, labels = next(train_iter)   ##
    input_ids = input_ids.to(config['device'])
    MSAconf = MSAconf.to(config['device'])
    labels = labels.to(config['device'])

    # Forward pass
    return_ = model(input_ids, MSAconf, labels)
    print(tokenizer.decode(return_['mask_pred']))
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
            i = 0
            vprog = tqdm(enumerate(val_dl), total=min(len(test_dl), 50), desc = '           validation')
            for i, (input_ids, MSAconf, labels) in vprog:
                input_ids = input_ids.to(config['device'])
                MSAconf = MSAconf.to(config['device'])
                labels = labels.to(config['device'])
                loss = model(input_ids, MSAconf, labels)['loss']
                vrl += loss.item()
                i+=1
                if i == 50:
                        break

            vloss = vrl / len(val_dl)
            valid_losses.append(vloss)
            
            # Save the best model
            if vloss < best_val_loss:
                # best_val_loss = vloss
                # test_predictions = []
                # test_actuals = []

                # teprog = tqdm(enumerate(test_dl), total=len(test_dl), desc = '           testing')
                # for i, (input_ids, MSAconf, labels) in teprog:
                #     input_ids = input_ids.to(config['device'])
                #     MSAconf = MSAconf.to(config['device'])
                #     labels = labels.to(config['device'])
                #     out_ = model(input_ids)
                #     mask_preds = out_['predictions']
                #     test_predictions.extend(mask_preds.detach().numpy())             
                #     test_actuals.extend(labels[np.where(input_ids==tokenizer.mask_token_id)].detach().numpy())
                # # Calculate accuracy
                # test_predictions = [pred.item() for pred in test_predictions]
                # test_actuals = [int(pred) for pred in test_actuals]
                # tacc = accuracy_score(test_actuals, test_predictions)
                # test_accuracies.append(tacc)
                torch.save(model.state_dict(), './mlm-baby-bert/BERT_depth4_embed512_steps2000.pt')

        print(
            "          "
            f" --> training loss: {train_loss:.4f}   "
            f"validation loss: {vloss:.4f}   "
            # f"accuracy: {tacc*100:.2f}%   "
            f"time: {timedelta(seconds=time.time() - t0)}"
        )

print('Total time: ', {timedelta(seconds=time.time() - t00)} )
# %%