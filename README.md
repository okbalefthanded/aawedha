# Aawedha

***Aawedha*** (*عاودها* means repeate it or do it again in Algerian arabic) is a deep learning learning package based on [PyTorch](https://pytorch.org), for EEG based Brain-Computer Interface (BCI) decoding research and applications.


Compatible with **Python 3.8 and above**
---
## Disclaimer

The project is in a work-in-progress stage, the API and features will change often towards a stable release. use with caution (any contribution is welcomed).


## Motivation

The main goal for this package is to provide a flexible and complete analysis and benchmarking tool for Deep Learning research in BCI. 

---

## Features
Aawedha provides a complete set of operations from raw data preprocessing to model training, evaluation and results visualization. A regular workflow using this package consists of 5 instructions:
- Create a dataset: preprocess raw data to create epoched EEG trials (run once)
- Define an Evaluation : Train or Single subject or Cross Subject analysis with the data and model.
- Generate a random data split (optional, used when no independent test set is available in a given dataset).
- Run evaluation : train and test/validate model.
- Visualize the results and what the model has learnt.

The tables below show the available datasets and models, for a detailed tutorial on running the evaluations follow the colaboratory notebook in the examples folder. 
### Data


Check the complete list of supported datasets overhere: [DataSets](https://github.com/okbalefthanded/aawedha/wiki/DataSets-List) 

     

### Deep Learning Models
The supported DL models are available here: [Models](https://github.com/okbalefthanded/aawedha/wiki/Supported-Models)

---

## Installation

First, clone Aawedha using git:
```
git clone https://github.com/okbalefthanded/aawedha.git
```
Then, cd to the Aawedha folder, install requirements using pip then proceed to package setup:
```
cd aawedha

pip install -r requirements.txt

python setup.py install
```

---

## Usage
```
Follow the colab notebooks in /examples
```
---

## Citation
```
Coming soon...
```

---

## Acknowledgment 
This work would not see the light without the precious open source libraries and models made freely available by their authors. I sincerely thank them for helping advancing the BCI field. This is one tiny sand grain in the ecosystem dune.
