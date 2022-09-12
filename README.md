# Aawedha

***Aawedha*** (*عاودها* means repeate it or do it again in Algerian arabic) is a deep learning learning package based on [Keras](https://www.tensorflow.org/guide/keras/overview) with [Tensorflow](https://www.tensorflow.org/guide) backend and [PyTorch](https://pytorch.org), for EEG based Brain-Computer Interface (BCI) decoding research and application.

Compatible with **Python 3.6 and above**
---
## Disclaimer

The project is in a work-in-progress stage, the API and features will change often towards a stable release. use with caution.


## Motivation

The main goal for this package is to provide a flexible and complete analysis and benchmarking tool for Deep Learning research in BCI. 

---

## Features
Aawedha provides a complete set of operations from raw data preprocessing to model evaluation and results visualization. A regular workflow using this package consists of 5 instructions:
- Create a dataset: preprocess raw data to create epoched EEG trials (run once)
- Define an Evaluation : Train or Single subject or Cross Subject analysis with the data and model.
- Generate a random data split.
- Run evaluation : train and test/validate model.
- Visualize the results and what the model has learnt.

The tables below show the available datasets and models, for a detailed tutorial on running the evaluations follow the colaboratory notebook in the examples folder. 
### Data

|   Datasets               | Paradigm      | Participants(subjects)  | 
| -------------            |:-------------:| :-----:|
| [BCI Competetion IV 2a](http://www.bbci.de/competition/iv/)    | Motor Imagery | 9      | 
| [Exoskleton](https://github.com/sylvchev/dataset-ssvep-exoskeleton)               | SSVEP         | 12     |      
| [Freiburg Online ERP](https://zenodo.org/record/192684)            | ERP     |     13 | 
| [Inria ERN](https://www.kaggle.com/c/inria-bci-challenge)            | ErrP      |   26     |
| [Laresi Hyrbid]()            | Hybrid ERP/SSVEP      |    1    |
| [Physionet_MI](https://physionet.org/content/eegmmidb/1.0.0/)            | Motor Imagery      |    109    |
| [San Diego](ftp://sccn.ucsd.edu/pub/cca_ssvep)            | SSVEP      |   10     |
| [Tsinghua](http://bci.med.tsinghua.edu.cn/download.html)            | SSVEP     |   35     |       

### Deep Learning Models

|   Title       | Paradigm      | Architecture  |
| ------------- |:-------------:| -----:|
| [EEGTCNET](https://github.com/iis-eth-zurich/eeg-tcnet)       | Motor Imagery / ERP | ConvNet |
| [EEGNET](https://github.com/vlawhern/arl-eegmodels)       | Motor Imagery / ERP/Errp | ConvNet |
| [EEGNet SSVEP](https://github.com/vlawhern/arl-eegmodels)  | SSVEP         |   ConvNet |
| [DeepConvNet/ ShallowConvNet](https://github.com/TNTLFreiburg/braindecode) | Motor Imagery / ERP/Errp      |    ConvNet |
| [1DCSU](https://arxiv.org/abs/1805.04157)       | SSVEP | ConvNet |
| [PodNet](http://dro.dur.ac.uk/27626/)      | SSVEP | ConvNet |
| [KoreaU CNN](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0172578)       | SSVEP | ConvNet |
| [Xu_Jiang CNN](https://ieeexplore.ieee.org/document/8708243)       | SSVEP | ConvNet |
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

---

## Acknowledgment 

