# Robustness-aware Word Embedding Improves Certified Robustness to Adversarial Word Substitutions
This is the data and code for paper **[Robustness-Aware Word Embedding Improves Certified Robustness to Adversarial Word Substitutions (Findings of ACL 2023)](https://aclanthology.org/2023.findings-acl.42/)**.
## Environment
* Pytorch 1.11.0+cu113
* NLTK 3.7
* Keras 2.2.5
## Datasets
Download data dependencies by running the provided script:
```shell
./download_deps.sh
```
If you already have GloVe vectors on your system, it may be more convenient to comment out the part of download_deps.sh that downloads GloVe, and instead add a symlink to the directory containing the GloVe vectors at data/glove.
## How to run
We provide complete training scripts to reproduce the results in our paper. These scripts are in the `command` folder. For example, to reproduce the result of IMDB dataset with EIBC+IBP training method, simply run
```shell
bash command/eibc+ibp/eibc+ibp_imdb_textcnn.sh
```
In the `command` folder
* `command/eibc` provides scripts of EIBC+Normal training method.
* `command/eibc+ibp` provides scripts of EIBC+IBP training method.
* `command/ibp` provides scripts of our implementation of IBP method.
* `command/half_syns` provides scripts of unseen word substitutions experiment.
* `command/ga.sh` provides script of genetic attack.
## Implementation
Part of the codes in this repo are borrowed/modified from [1], [2].
## References:
[1] https://github.com/robinjia/certified-word-sub

[2] https://github.com/JHL-HUST/FTML
