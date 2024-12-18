#%% import libraries
import torch
import numpy as np
import transformers
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

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
    'batch_size' : 1,
    'dim': 256,
    'n_heads': 8,
    'attn_dropout': 0.1,
    'mlp_dropout': 0.1,
    'depth': 4,
    'vocab_size': len(tokenizer.get_vocab()),
    'max_len': 1000,
    'pad_token_id': tokenizer.pad_token_id,
    'mask_token_id': tokenizer.mask_token_id,
    'device': 'cpu',
    # 'loss':'BLOSUM' 
}

# load the data
train_dl, val_dl, test_dl = load_dataset("../input/all_queries.csv", tokenizer, config)

# define the model architecture
model = MLMBERT(config).to(config['device'])
print('     trainable: ',sum([p.numel() for p in model.parameters() if p.requires_grad]))

# load the pretrained model parameters
saved_model = './mlm-baby-bert/BERTMSA_depth4_embed256_head8_steps2000.pt'
state_dict = torch.load(saved_model)
model.load_state_dict(state_dict, strict=False)

#%%
# calculate test accuracy
test_predictions = []
test_actuals = []

for input_ids, MSAconf, labels in test_dl:
    input_ids = input_ids.to(config['device'])
    labels = labels.to(config['device'])
    out_ = model(input_ids)
    mask_preds = out_['mask_predictions']
    test_predictions.extend(mask_preds.detach().numpy())             
    test_actuals.extend(labels[np.where(input_ids==tokenizer.mask_token_id)].detach().numpy())

    actual = labels.clone()
    masked = actual.clone()
    predicted = input_ids.clone()
    actual[labels == -100] = input_ids[labels == -100]
    actual = tokenizer.decode(actual[0], skip_special_tokens=True)
    masked = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    masked = masked.replace('<mask>', 'X')
    predicted[labels != -100] = mask_preds
    predicted = tokenizer.decode(predicted[0], skip_special_tokens=False)
    masked_place = ''.join('^' if char == 'X' else '-' for char in masked.replace('<pad>', ''))

    print('ACTUAL:', actual.replace('<pad>', ''))
    print('MASKED:', masked.replace('<pad>', ''))
    print('MODEL :', predicted.replace('<pad>', ''))
    print('      :', masked_place)

test_predictions = [pred.item() for pred in test_predictions]
test_actuals = [int(pred) for pred in test_actuals]
tacc = accuracy_score(test_actuals, test_predictions)

print(f'Accuracy of {tacc*100:.2f}% is achieved for the model saved in \n       {saved_model}' )
# %%
