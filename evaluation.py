import numpy as np
import pandas as pd
import torch
import utils
from tqdm import tqdm
import transformers
from mlm_simple.models.BERTquery import MLMBERT
from sklearn.model_selection import train_test_split

# [(id, seq), ...]

def single_mask_test(models, queries, config, model_names=None, outpath=None):
    if model_names is None:
        model_names=np.arange(len(models))
    if len(model_names) != len(models):
        print('len(model_names) != models. setting model names to index in list')
        model_names=np.arange(len(models))

    np.random.seed(0)
    mask_indices=[np.random.randint(0,len(seq)) for seq in queries['Sequence']]
    labels=[(id, seq) for i, (id, seq) in queries.iterrows()]
    batch = [
        (row["Header"], utils.mask(row["Sequence"], index))
        for (index, row), index in zip(queries.iterrows(), mask_indices)
    ]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'mlm_simple/mlm-baby-bert/tokenizer/protein_tokenizer',
        use_fast=True,
        unk_token="<unk>",
        mask_token="<mask>",
        pad_token="<pad>"
    )
    batch_ids=torch.Tensor([tokenizer.encode(seq)+(config['max_len']-len(tokenizer.encode(seq)))*[config['pad_token_id']] for _,seq in batch if len(seq)<1000]).int()
    results_list=[]
    for model, name in zip(models,model_names):
        # Inference
        with torch.no_grad():
            results = model(batch_ids)['logits'].argmax(dim=-1)

        # Evaluation
        for pred,(header, seq),i in tqdm(zip(results, labels, mask_indices)):
            results_list.append({'Name': name,'Header': header, 'Sequence': seq,
                                 'Mask': i, 'Prediction': tokenizer.decode(pred[i+1]),
                                 'Label': seq[i], 'Correct': int(seq[i]==tokenizer.decode(pred[i+1]))})
    

    results_df=pd.DataFrame(results_list)
    if outpath is not None:
        results_df.to_csv(outpath)
    return results_df



def single_mask_test_MSA(models, queries, config, model_names=None, outpath=None):
    if model_names is None:
        model_names=np.arange(len(models))
    if len(model_names) != len(models):
        print('len(model_names) != len(models). setting model names to index in list')
        model_names=np.arange(len(models))
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'mlm_simple/mlm-baby-bert/tokenizer/protein_tokenizer',
        use_fast=True,
        unk_token="<unk>",
        mask_token="<mask>",
        pad_token="<pad>"
    )

    np.random.seed(0)
    #[print(head.split('|')[1]) for i,(head,seq,query) in queries.iterrows()]
    mask_indices=[np.random.randint(0,len(MSA['MSA'][0])) for i, (id, MSA) in queries.iterrows()]
    labels=[(id, ['MSA'][0]) for i, (id, MSA) in queries.iterrows()]
    print(labels)
    batch = [(tokenizer.encode(list(utils.mask_MSA(['MSA'][0], index)))+(config['max_len']-len(tokenizer.encode(seq)))*[config['pad_token_id']], ['MSA'][1]) for i, (id, MSA) in queries.iterrows()]
    
    results_list=[]
    for model, name in zip(models,model_names):
        # Inference
        with torch.no_grad():
            results = model(batch)['logits'].argmax(dim=-1)

        # Evaluation
        for pred,(header, seq),i in tqdm(zip(results, labels, mask_indices)):
            results_list.append({'Name': name,'Header': header, 'Sequence': seq,
                                 'Mask': i, 'Prediction': tokenizer.decode(pred[i+1]),
                                 'Label': seq[i], 'Correct': int(seq[i]==tokenizer.decode(pred[i+1]))})
    

    results_df=pd.DataFrame(results_list)


    if outpath is not None:
        results_df.to_csv(outpath)
    return results_df



def load_model(filepath, config):
    model = MLMBERT(config).to(config['device'])
    state_dict = torch.load(filepath)
    model.load_state_dict(state_dict, strict=False) 
    return model

if __name__ == "__main__":
    np.random.seed(0)
    queries=pd.read_csv('input/all_queries.csv')
    _,queries=train_test_split(queries, test_size=0.3, random_state=42, shuffle=True)
    _,queries=train_test_split(queries, test_size=0.5, random_state=42, shuffle=True)
    print('data imported')
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'mlm_simple/mlm-baby-bert/tokenizer/protein_tokenizer',
        use_fast=True,
        unk_token="<unk>",
        mask_token="<mask>",
        pad_token="<pad>"
    )
    configs = [
        {
            'batch_size': 32,
            'dim': 256,
            'n_heads': 8,
            'attn_dropout': 0.1,
            'mlp_dropout': 0.1,
            'depth': 6,
            'vocab_size': len(tokenizer.get_vocab()),
            'max_len': 1000,
            'pad_token_id': tokenizer.pad_token_id,
            'mask_token_id': tokenizer.mask_token_id,
            'device': 'cpu'
        },
        {
            'batch_size': 32,
            'dim': 512,
            'n_heads': 8,
            'attn_dropout': 0.1,
            'mlp_dropout': 0.1,
            'depth': 4,
            'vocab_size': len(tokenizer.get_vocab()),
            'max_len': 1000,
            'pad_token_id': tokenizer.pad_token_id,
            'mask_token_id': tokenizer.mask_token_id,
            'device': 'cpu'
        },
        {
            'batch_size': 32,
            'dim': 256,
            'n_heads': 16,
            'attn_dropout': 0.1,
            'mlp_dropout': 0.1,
            'depth': 4,
            'vocab_size': len(tokenizer.get_vocab()),
            'max_len': 1000,
            'pad_token_id': tokenizer.pad_token_id,
            'mask_token_id': tokenizer.mask_token_id,
            'device': 'cpu'
        },
        {
            'batch_size': 32,
            'dim': 256,
            'n_heads': 8,
            'attn_dropout': 0.1,
            'mlp_dropout': 0.1,
            'depth': 4,
            'vocab_size': len(tokenizer.get_vocab()),
            'max_len': 1000,
            'pad_token_id': tokenizer.pad_token_id,
            'mask_token_id': tokenizer.mask_token_id,
            'device': 'cpu'
        }
        ]
    filepaths = [
        "mlm_simple/mlm-baby-bert/BERT_depth4_embed256_steps2000.pt",
        "mlm_simple/mlm-baby-bert/BERT_depth4_embed512_steps2000.pt",
        "mlm_simple/mlm-baby-bert/BERT_depth4_embed256_head16_steps1000.pt",
        'mlm_simple/mlm-baby-bert/BERTMSA_depth4_embed256_head8_steps2000.pt'
        ]
    models=[load_model(filepath, config) for config,filepath in zip(configs, filepaths)]
    print('models imported')
    names=[filepath.split('/')[-1].split('.')[0] for filepath in filepaths]
    queries=queries[queries['Sequence'].apply(len)<configs[0]['max_len']][:100]
    res=single_mask_test(models, queries, configs[0], names, outpath='model_evaluation.csv')
    np.random.seed(0)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'mlm_simple/mlm-baby-bert/tokenizer/protein_tokenizer',
        use_fast=True,
        unk_token="<unk>",
        mask_token="<mask>",
        pad_token="<pad>")
    config={
            'batch_size': 32,
            'dim': 256,
            'n_heads': 8,
            'attn_dropout': 0.1,
            'mlp_dropout': 0.1,
            'depth': 4,
            'vocab_size': len(tokenizer.get_vocab()),
            'max_len': 1000,
            'pad_token_id': tokenizer.pad_token_id,
            'mask_token_id': tokenizer.mask_token_id,
            'device': 'cpu'}
    
    res=single_mask_test(models, queries, config, names, outpath='model_evaluation.csv')

