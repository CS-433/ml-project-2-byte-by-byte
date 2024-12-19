<p align="center" style="font-weight: bold; font-size: 28px;">
Language model for protein design: predicting masked residues using sequences
</p>
<hr>
<p align="center" style="font-weight: bold; font-size: 20px;">
Guillaume Belissent, Charles Stockman, Victoire Ringler
</p>
<p align="center" style="font-style: italic; font-size: 16px;">
Machine Learning CS-433, EPFL, Switzerland
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
5. [Testing](#testing)

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
