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
    'dim': 320,
    'n_heads': 20,
    'attn_dropout': 0.1,
    'mlp_dropout': 0.1,
    'depth': 6,
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

# load the model parameters
state_dict = torch.load('./mlm-baby-bert/BERT_depth6_embed320_head20_steps1000.pt')
model.load_state_dict(state_dict, strict=False)

#%%
train_iter = cycle(train_dl)
input_ids, labels = next(train_iter)  

print(input_ids.shape)
print(input_ids)
#%%
# TRAINING 
    # 32 queries through the model in one training step
training_steps = 500
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

                teprog = tqdm(enumerate(test_dl), total=min(len(test_dl), 50), desc = '           testing')
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
                # torch.save(model.state_dict(), './mlm-baby-bert/BERT_depth6_embed320_head20_steps500_1000.pt')
        
        print(
            "          "
            f" --> training loss: {train_loss:.4f}   "
            f"validation loss: {vloss:.4f}   "
            f"accuracy: {tacc*100:.2f}%   "
            f"time: {timedelta(seconds=time.time() - t0)}"
        )

print('Total time: ', {timedelta(seconds=time.time() - t00)} )

#%%
# import matplotlib.pyplot as plt

# plt.plot(train_losses,color='red',label='train loss')
# plt.plot(valid_losses,color='orange',label='valid loss')
# plt.legend()
# plt.show()

# plt.plot(test_accuracies)
# plt.title('single mask token prediction accuracy')
# plt.show()
# %%


##############
##############
#%%

# TRAINING 
    # 32 queries through the model in one training step
##############
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import random

# data = pd.read_csv('../input/all_queries.csv')
# data = data[data['Header'].str.contains("query")]
# data['Sequence'] = data['Sequence'].str.upper()
# sequences = data["Sequence"]

# # Split the data into training and testing sets (97% train, % val, % test)
# train_sequences, val_sequences = train_test_split(sequences, test_size=0.1, random_state=42, shuffle=True)
# val_sequences, test_sequences = train_test_split(val_sequences, test_size=0.2, random_state=42, shuffle=True)

# special_tokens = [tokenizer.pad_token_id, tokenizer.unk_token_id, tokenizer.mask_token_id]

# def predict_mask(sentence):
#     sentence = sentence.lstrip('-').replace('-', '<unk>')[:config["max_len"]] 
#     x = tokenizer.encode(
#         sentence,  # The sequence to tokenize
#         add_special_tokens=True,  # Add special tokens like <CLS>, <SEP>, etc.
#         truncation=True,  # Truncate if necessary (for long sequences)
#         max_length=config['max_len'],  # Adjust as per your model's context size
#         padding="max_length",  # Pad sequences to the max length
#         return_tensors="pt",  # Return PyTorch tensors
#         )[0]
#     # Create a mask: 1 for special tokens, 0 for non-special tokens
#     special_tokens_mask = np.array([1 if token in special_tokens else 0 for token in x])
#     special_tokens_mask_tensor = torch.tensor(special_tokens_mask)

#     non_special_token_indices = torch.where(special_tokens_mask_tensor == 0)[0]
#     if non_special_token_indices.numel() > 0:
#         # Randomly choose a non-special token
#         # idx = torch.randint(0, non_special_token_indices.size(0), (1,)).item()
#         idx = torch.randint(0, non_special_token_indices.size(0), (1,))#.item() # choose the amount of values to be masked, eg. 5
#         idx = idx.unique()
#         idx = non_special_token_indices[idx]
#     else:
#         return 0,0
#     input_ids = x
#     predicted = x.clone().detach()
#     masked_ID = input_ids[idx] # eg. 20
#     # print(masked_ID)
#     masked_token = tokenizer.decode(masked_ID) # true value, eg. W
#     # print(masked_token)
#     # masking
#     input_ids[idx] = 2 # idx -> [MASK]
#     masked_sentence = input_ids.clone().detach()
#     # preparing input
#     input_ids = input_ids.clone().detach().unsqueeze(0).to('cpu')
    
#     # extracting the predicted token
#     out = model(input_ids)
#     ##
#     # import matplotlib.pyplot as plt
#     # logitz = out['logits']
#     # logitz = logitz.view(-1, 1000, 23).clone().detach().numpy()
#     # j=0
#     # IDS = input_ids[j]==2
#     # fig, axs = plt.subplots()
#     # for i in range(3,23):
#     #     axs.plot(logitz[j,IDS,i])
#     # for i in range(0,3):
#     #     axs.plot(logitz[j,IDS, i], color= 'red', linewidth = 5, label=i)
#     # plt.show()
#     ##
#     predicted[idx] = out['mask_predictions']#.item()
    
#     predicted_token = tokenizer.decode(out['mask_predictions'])
#     print(f'masked: {masked_token} predicted: {predicted_token}')
#     masked_sentence = tokenizer.decode(masked_sentence,skip_special_tokens=False)
#     masked_sentence = masked_sentence.replace('<pad>','')
#     masked_sentence = masked_sentence.replace('<mask>','x')
#     masked_place = ''.join('^' if char == 'x' else '-' for char in masked_sentence)
#     # print('ACTUAL:',sentence)
#     # print('MASKED:',masked_sentence)
#     # print(' MODEL:',tokenizer.decode(predicted, skip_special_tokens=True))
#     # print('      :',masked_place)
#     similar = masked_ID == out['mask_predictions']
#     total_c = np.array(similar.sum().item())
#     total_pred = np.array(out['mask_predictions'].size()[0])
#     # print(total_c)
#     # print(total_pred)
#     return total_c, total_pred

# ##############
# training_steps = 1000
# train_losses = []
# valid_losses = []
# test_accuracies = []

# best_val_loss = float('inf')

# train_iter = cycle(train_dl)
# combined_sequences = pd.concat([train_sequences, val_sequences, test_sequences]).reset_index(drop=True)

# batch_logits = []
# N = 50
# # step = 0
# t00 =  time.time()
# optim = torch.optim.Adam(model.parameters(), lr=4e-4 / 25.)
# sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=4e-4, total_steps=training_steps)
# model.train()
# tprog = tqdm(range(training_steps), desc="Training Progress")
# t0 = time.time()
# for step in tprog:
#     trl = 0.
#     model.train()
#     input_ids, labels = next(train_iter)  
#     input_ids = input_ids.to(config['device'])
#     labels = labels.to(config['device'])
    
#     # Forward pass
#     return_ = model(input_ids, labels)
#     loss = return_['loss']
#     loss.backward()
    
#     # Optimization step
#     optim.step()
#     optim.zero_grad()
#     sched.step()
#     trl += loss.item()
    
#     # Log every N steps
#     if (step + 1) % N == 0:
            
#         torch.manual_seed(1420)
#         correct = 0
#         # train_sequences = train_sequences.reset_index(drop=True)
#         # test_sequences = test_sequences.reset_index(drop=True)
#         total_correct = 0
#         total_predicted = 0
#         for sentence in random.choices(combined_sequences,k=100):
#             total_c, total_pred = predict_mask(sentence)
#             total_correct += total_c
#             total_predicted += total_pred
#         print(f'CORRECT:{total_correct}/{total_predicted}')

# print('Total time: ', {timedelta(seconds=time.time() - t00)} )
# ##############

# # # load the model parameters
# # state_dict = torch.load('./mlm-baby-bert/BERT_depth6_embed320_steps500.pt')
# # model.load_state_dict(state_dict, strict=False)

# # print('     trainable: ',sum([p.numel() for p in model.parameters() if p.requires_grad]))

# # #%%

# # # %%

# # for input_ids, labels in test_dl:
# #     input_ids = input_ids.to(config['device'])
# #     labels = labels.to(config['device'])
# #     out_ = model(input_ids)
# #     mask_preds = out_['mask_predictions']
# #     logitsz = out_['logits']
# #     break
# #     # test_predictions.extend(mask_preds.detach().numpy())             
# #     # test_actuals.extend(labels[np.where(input_ids==tokenizer.mask_token_id)].detach().numpy())
# #     # if i == 5:
# #     #     break
# #     # i += 1
# # #%%
# # #AFTER TRAINING, INFERENCE
# # logitsz.shape
# # labels.shape
# # specific_inference = logitsz.view(32, 1000, 23).clone().detach().numpy()

# # #%%
# # import matplotlib.pyplot as plt


# # IDS = input_ids[0]==2
# # IDSno = input_ids[0]!=2

# # j=0
# # IDS = input_ids[j]==2
# # for i in range(3,23):
# #     plt.plot(specific_inference[j,IDS,i])

# # for i in range(0,3):
# # # i = 2
# #     plt.plot(specific_inference[j,IDS, i], color= 'red', label=i)

# # #%%
# # logitsz[:, :, 0:3].shape
# # #%%
# # # DURING TRAINING
# # logits_.view(32, 1000, 23).shape
# # specific = logits_.view(32, 1000, 23).clone().detach().numpy()
# # # %%
# # input_ids[0].shape#[0,0]
# # # %%
# # specific[0,:, 0].shape
# # # %%
# # #%%
# # for i in range(3,23):
# #     plt.plot(specific[0,:, i])
# # plt.plot(-40+(input_ids[0]==2)*10)
# # #%%
# # for i in range(0,3):
# #     plt.plot(specific[0,:, i], label=i)
# # # plt.plot(-40+(input_ids[0]==2)*10)
# # plt.legend
# # #%%
# # IDS = input_ids[0]==2
# # IDSno = input_ids[0]!=2

# # specific[0,IDS, i]

# # for j in range(32):
# #     IDS = input_ids[j]==2
# #     for i in range(3,23):
# #         plt.plot(specific[j,IDS,i])

# #     for i in range(0,3):
# #         plt.plot(specific[j,IDS, i], color= 'red', label=i)

# # #%%
# # for j in range(32):
# #     IDS = input_ids[j]!=2
    
# #     for i in range(3,23):
# #         plt.plot(specific[j,IDS,i])

# #     for i in range(0,3):
# #         plt.plot(specific[j,IDS,i], color= 'red', label=i)

# # #%%
# # for i in range(3,23):
# #     plt.plot(specific[0,IDSno,i])

# # for i in range(0,3):
# #     plt.plot(specific[0,IDSno, i], color= 'red', label=i)

# # #%%
# # # give the ID where equal to 2


# # # %%
# # plt.plot(input_ids[0])
# # plt.plot(input_ids[0]==2)
# # # %%
# # tokenizer.get_vocab()
# # # %%

# # # %%

# # # %%
# # (input_ids[0]==2).sum()/(input_ids[0].shape[0]-(input_ids[0]==1).sum())
# # # %%

# # %%
