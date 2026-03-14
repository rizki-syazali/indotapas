# Baseline Experiment 1: Weakly Supervised Semantic Parsing (Wang et al., 2019)

This repository contains the implementation and reproduction steps for the first baseline experiment used to evaluate the effectiveness of our proposed model, IndoTaPas. This baseline adopts the weakly supervised semantic parsing model introduced by Wang et al. (2019).

## Overview

This baseline approach is designed to transform natural language questions into executable programs (logical forms akin to SQL). These programs are then executed against a table to produce the final answer (denotation). 

Key characteristics of this model include:
* **Weakly Supervised Learning:** The model learns directly from question-denotation pairs, eliminating the need for explicit annotations of ground-truth programs.
* **Two-Stage Parsing Framework:** By introducing an inductive bias through latent structured alignments and abstract programs, the model achieves state-of-the-art execution accuracy on the WTQ dataset.
* **Search Space Reduction:** To maintain efficiency during program generation, the search space is constrained by predefined grammar rules, ensuring that only valid and meaningful programs are explored during training.

The primary focus of the implementation in this repository is adapting the **IndoHiTab** dataset into a format compatible with the WTQ dataset structure, allowing it to be processed and trained using this baseline architecture.


## Setup
Use conda to create a virtual environment and setup this package
    
    git clone https://github.com/berlino/weaksp_em19.git
    cd weaksp_em19
    conda create --name weaksp python=3.7
    conda activate weaskp
    pip install --user -e .

The preprocessed data is based on the one provided by [MAPO](https://github.com/crazydonkey200/neural-symbolic-machines). 
The preprocessed data is available in the [preprocessed_data](./preprocessed_data) folder.


*Note:* We replace the default GloVe word embeddings (`glove.42B.300d.txt`) with Indonesian FastText (`cc.id.300.vec`) to effectively accommodate the Indonesian language.

**Important Modifications:**
* Replace the `sketch.action` file in the cloned repository (`weaksp_em19/processed/sketch.action`) with the `sketch.action` file provided in this repository.
* We replace the default GloVe word embeddings (`glove.42B.300d.txt`) with Indonesian FastText (`cc.id.300.vec`) to effectively accommodate the Indonesian language.


## Reproducing Experiments on IndoHiTab

> **Shortcut:** If you prefer to skip the preprocessing steps (Steps 1–3), you can directly use the ready-to-use files in the [`preprocessed_data`](./preprocessed_data) folder and jump straight to **Step 4**.

**1. Generate Preprocessed Files** with the following script:
```
bash scripts/gen_processed_pkl.sh
```

**2.** Evaluate the coverage and generate consistent programs by:

```
python scripts/eval_coverage demo 9 
```

where demo is the experiemnt id and 9 the maximal length of a sketch. 

**3**. Cache the generated programs with:

```
python scripts/cache_lf.py processed/demo.train.programs.sketch.stat processed/demo.train.programs train processed/train.pkl
python scripts/cache_lf.py processed/demo.test.programs.sketch.stat processed/demo.test.programs test processed/test.pkl
```

**4. Train the Model**. Start the training process by specifying your experiment ID (`demo`):
```
python train_seq.py demo
```
The configs of the training is in `train_config/train_config`. Currently, two model types are included:

* seq: seq2seq with abstract programs
* struct: abstract programs with structured alignments

The checkpoints will be available in `checkpoints` folder