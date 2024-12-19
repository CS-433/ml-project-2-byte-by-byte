<h1 align="center">
Language model for protein design: predicting masked residues using sequences
</h1>
<hr>
<h2 align="center">
Guillaume Belissent, Charles Stockman, Victoire Ringler
</h2>
<p align="center">
<em>Machine Learning CS-433, EPFL, Switzerland</em>
</p>

### About the project

Proteins are complex molecules essential for regulating numerous biological processes. They are composed of amino acid sequences that fold into diverse three-dimensional structures, enabling specific functions. The sequence of amino acids determines the structure and function of a protein, with each residue contributing properties such as charge, polarity, and hydrophobicity. Understanding these sequences is critical for uncovering protein behavior, designing new proteins, and advancing medicine and biological research.

In this repository, we provide tools for **predicting single masked amino acids in protein sequences using Machine Learning**. Our models are trained on large protein datasets, including Uniref90, MGnify, and the Big Fantastic Database (BFD). These models can process input in the form of a single query sequence or a Multiple Sequence Alignment (MSA). 
We investigate both state-of-the-art models, such as ESM, and our own in-house models. While we trained these models, we could not include the pre-trained versions directly in this GitHub repository due to their large size (stored as .pt files).

### Why is this interesting?
The models explored in this repository could contribute to protein engineering, particularly in the design of new proteins with desired properties (de novo design). By predicting amino acids in hypothetical sequences, these models help guide the composition of proteins with specific characteristics, such as improved stability or specificity.

The models additionally offer insights into protein properties by learning patterns in sequence relationships. Studying attention mechanisms or how embeddings encode biological information can help uncover fundamental principles about sequence structure, evolution, and functional motifs, advancing our understanding of proteins.


# Table of Contents
1. [Overview of the models](#Overview-of-the-models)
2. [Getting started](#Getting-started)
3. [Training](#training)
4. [Inference](#inference)
5. [Evaluation](#evaluation)

# Overview of the models

| Models with queries | ESM 8M esm2_t30_8M_UR50D | ESM 150M esm2_t30_150M_UR50D | pBERT comparison ESM | pBERT baseline | pBERT large embeddings| pBERT more attention heads|
| --------------------| ------------------------ | ---------------------------- | -------------------- |--------------- | --------------------- |-------------------------- | 
|Number of layers = depth|6|30|6|4|4|4|
|Embedding dimension|320|640|320|256|512|256|
|Attention heads|20|20|20|8|8|16|
|Training steps|500K|500K|500|2000|2000|1000|
|Learning rate|4e-4|4e-4|4e-4|4e-4|4e-4|4e-4|
|Criterion|Cross-entropy|Cross-entropy|Cross-entropy|Cross-entropy|Cross-entropy|Cross-entropy|

Note that the pBERT models presented here were specifically trained and evaluated for this project. However, all parameters are adjustable and can be fine-tuned according to your preferences.

# Getting started
## Dependencies
To run the training, inference and testing of our models, the following dependencies must be installed:
- Python 3.x
- PyTorch
- Pandas
- Transformers library by Hugging Face
- NumPy
- scikit-learn
- tqdm

You can install them using the following command:

```bash
pip install torch pandas transformers numpy scikit-learn tqdm
```
## Directory structure
You can clone this repository using:
```bash
git clone https://github.com/CS-433/ml-project-2-byte-by-byte.git
```

When you clone the repository, the directory structure will be as follows:

```bash
/project
├── /esm                       # Contains ESM-related models and resources
├── /input                     # Placeholder for your input data when using the csv option (see below)
├── /mlm_simple                # In-house models
├── dataloader.py              # Script for handling data loading
├── evaluation.py              # Script for evaluating the model
├── utils.py                   # Utility functions
└── README.md                  # Documentation
```

## Data Setup
### 1. **Included datasets**
This repository provides two datasets:
- all_queries.csv: Contains queries from 131'487 proteins, sourced from three databases:
	- UniRef90
	- MGnify
	- Big Fantastic Database (BFD)

- all_sequences.csv: Contains all Multiple Sequence Alignments (MSAs) for the proteins listed in all_queries.csv, derived from the same three databases.

### 2. **CSV Data Format**
To train the models on your own data, you can provide it as a `.csv` file in the `input` directory of the project. The CSV file should contain two columns: Header and Sequence, formatted as follows:

| Header | Sequence |
|--------|----------|
| seq1   | MKTAYIAKQRQISFVKSHFS... |
| seq2   | MNSMGHQRTLLPFGK... |

### 3. **A3M File Format**
Alternatively, if your protein data is in the A3M format (typically in folders), you can organize your data in a folder structure and use the provided script in `dataloader.py` to convert the A3M files to CSV format.

The organization of the folder should be as follows:
```bash
└── /openfold                  # Directory for A3M files
    ├── /protein1
    │   └── protein1.a3m           # A3M file for protein 1
    └── /protein2
        └── protein2.a3m           # A3M file for protein 2
```

The openfold folder should be at the same level as the `esm`, `input`, and `mlm_simple` folders in the project structure.

# Training
To train the models using your data and chosen configuration, follow these steps:

#### 1. Modify the configuration: open the script in `pBERT_training_final.py` and update the config dictionary as needed:

- `batch_size`: Number of sequences in each batch.
- `dim`: Size of the embedding vector.
- `n_heads`: Number of attention heads in the transformer layers.
- `attn_dropout`: 
- `mlp_dropout`:
- `depth`: Number of transformer layers.
- `max_len`: Maximum length of the sequences used for training.
- `device`: Choose 'cuda' if you have a GPU available; otherwise, use 'cpu'.
- `loss`: By default set to cross-entropy. However, we have also included an experimental BLOSUM loss function in the script. Note that the BLOSUM loss is not functional in this version of the model, but it is designed to be close to working. We have left the draft implementation in the repository for anyone who is interested in experimenting with it and potentially finding a solution. To use it, the `loss` should be set to 'BLOSUM'.

#### 2.Model evaluation and checkpoint:
Evaluation frequency can be modified in the script by modifying the parameter `N`.
The best model during training will be saved in the directory mlm-baby-bert/ as BERT_best_model.pt (modifiable if needed).
Any model can be reloaded for evaluation or predictions later using:
```bash
model.load_state_dict(torch.load('./mlm-baby-bert/BERT_best_model.pt'))
```

#### 3. Run the training:
Run the script in your Python environment using:
```bash
python pBERT_training_final.py
```

#### 4. Monitoring Training:
The training process will be displayed in the terminal with the current training loss, validation loss and elapsed time.

# Inference
After training the model, you can use it to make predictions on new protein sequences. The following instructions describe how to load the trained model, prepare the input data, and run inference to predict masked amino acids in protein sequences.

#### 1. Modify the configuration and load the models: update the script in `pBERT_inference_final.py`
As for the training the configuration should be set up according to your model.
The pretrained model should then be loaded using:

```bash
saved_model = './mlm-baby-bert/model_chosen.pt'
state_dict = torch.load(saved_model)
model.load_state_dict(state_dict, strict=False)
```

#### 2. Input data
The model can process both single query sequences and Multiple Sequence Alignments (MSAs). 
The dataset as presented in the section `Data setup` should be loaded as `.csv` using:
```bash
train_dl, val_dl, test_dl = load_dataset("../input/your_data.csv", tokenizer, config)
```

#### 3. Run the inference
Once the model is loaded, you can use it to predict masked residues in your protein sequences by running the script in your Python environment using:
```bash
python pBERT_inference_final.py
```

#### 4. Output
After running the inference loop, the model will output the actual, masked, and predicted sequences for each protein. It will also print the accuracy of the model on the test dataset.

# Evaluation
To evaluate the models trained, used the following steps: 

#### 1. Modify the configuration and load the models to evaluate: update the script in `evaluation.py`
To evaluate the models, the script needs to be adjusted to include the appropriate configurations for each model. These configurations can be modified directly in the `load_model` function within the `evaluation.py` script.
- Step 1.1: Adjust the configuration for each model to match the specific hyperparameters required for evaluation.
- Step 1.2: Specify the file paths to the pre-trained models you want to evaluate within the `load_model` function.

#### 2. Prepare the dataset on which the test will be performed
The dataset of protein sequences is loaded from a `.csv` file specified in the `load_model` function. By default, the script uses `all_queries.csv`, but you can replace it with any dataset of your choice. 
- Step 2.1: The script automatically splits the dataset into training and testing sets.
- Step 2.2: The sequences are processed to mask random positions.
- Step 2.3: The masked sequences are passed through each pre-trained model for inference.
- Step 2.4: The model’s predictions are compared to the ground truth, and accuracy is calculated for each model.

#### 3. Run the evaluation
You can then run the evaluation using:
```bash
python evaluation.py
```

#### 4. Output
The evaluation results will be saved to a CSV file (`model_evaluation.csv`). The CSV file contains the following columns:
- Name: The model name.
- Header: The header of the sequence.
- Sequence: The original sequence.
- Mask: The index of the masked amino acid.
- Prediction: The model's prediction for the masked amino acid.
- Label: The actual amino acid that was masked.
- Correct: A binary value indicating if the prediction was correct.

