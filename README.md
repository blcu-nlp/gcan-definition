# gcan-definition
基于门控化上下文感知网络的词语释义生成方法 
## Environment requirements and Data Preparation
+ Requirements
```
+ python 3.6
+ gensim (3.7.1)
+ numpy (1.16.2)
+ torch (0.4.0)
+ tqdm (4.31.1)
```

+ To get data for language model (LM) pretraining:
```
cd definition-example
mkdir data
cd data
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
unzip wikitext-103-v1.zip
```
+ To get data for Google word vectors use [official site](https://code.google.com/archive/p/word2vec/)

### Data Preparation
Contact the author of this [project](https://github.com/agadetsky/pytorch-definitions) to obtain the dataset.

将json文件放在data目录下

+ data format (json file)
```
data = [
  [
    ["word"],
    ["word1", "word2", ...],
    ["word1", "word2", ...]
  ],
  ...
]
So i-th element of the data:
data[i][0][0] - word being defined (string)
data[i][1] - definition (list of strings)
data[i][2] - context to understand word meaning (list of strings)
```
### Preprocess
+ Prepare the vocabulary and the word embedding.
```
cd scripts
bash preprocess.sh
```
### Baseline
+ S+G+CH 

The S+G+CH model is based on [Websail-NU/torch-defseq](https://github.com/Websail-NU/torch-defseq), and detailed instruction can be found there.
```
cd scripts
bash train_sgh.sh 0
```

+ S+IAdaptive

To install AdaGram software to implement S+IAdaptive Model

Specific content can be referred to [agadetsky/pytorch-definitions](https://github.com/agadetsky/pytorch-definitions)
```
bash train_ada.sh 0
```
+ S+IAttention

The S+IAttention model is based on [agadetsky/pytorch-definitions](https://github.com/agadetsky/pytorch-definitions), and detailed instruction can be found there.
```
bash train_iattention.sh 0
```

### Traing Proposed Model
+ Train the Gated Context-Aware Network
```
bash train_gca.sh 0
```
### Testing Model
+ Evaluate the PPL in test set
```
bash eval.sh
```
+ Generate the definitions in test set
```
bash generate.sh
```
You can compute bleu using python ./compute_bleu/bleu.py

### Reference:
https://github.com/salesforce/awd-lstm-lm

https://github.com/Websail-NU/torch-defseq

https://github.com/agadetsky/pytorch-definitions

Contact: z962630523@gmail.com
