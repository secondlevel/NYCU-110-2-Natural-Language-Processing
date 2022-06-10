# NYCU-110-2-Natural-Language-Processing

This project is to identify emotion through corpus, and it is related to [**kaggle competition**](https://www.kaggle.com/competitions/nycu-nlp110/overview).  

The model in this project was built by [**HuggingFace**](https://huggingface.co/) and Pytorch.  

<p float="center">
  <img src="https://user-images.githubusercontent.com/44439517/173000715-6985a7f0-3061-4b44-99f0-5c5e53a6b0cd.png" width="65%" height="65%" hspace="150"/>
</p>

## Reproducing Submission

Please do the following steps to reproduce the submission without retraining. You have two option to reproduce our method.

The first option will **not need to do** the model evaluation. The second option will **need to do** the model evaluation. 

- ### First Option
1. [Requirement](#Requirement)
2. [Repository Structure](#Repository-Structure)
3. [Ensemble](#Ensemble)

- ### Second Option
1. [Requirement](#Requirement)
2. [Repository Structure](#Repository-Structure)
3. [Testing](#Testing)
4. [Ensemble](#Ensemble)

## Hardware

In this project, the following machine was used to train the emotion classification model.

|                 | Operating System           | CPU                                      | GPU                         | 
|-----------------|----------------------------|------------------------------------------|-----------------------------|
| Machine 1       | Ubuntu 20.04.3 LTS         | Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz  | NVIDIA GeForce GTX TITAN X  |
| Machine 2       | Ubuntu 18.04.5 LTS         | Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz | NVIDIA GeForce GTX 1080     |
| Machine 3       | Ubuntu 20.04.3 LTS         | AMD Ryzen 5 5600X 6-Core Processor       | NVIDIA GeForce RTX 2080 Ti  |


## Requirement

In this project, the conda and pip toolkit was used to build the environment.

The following two **options** are provided for building the environment.
  
- ### First Option
```bash
conda env create -f environment.yml
```

- ### Second Option 
```bash
conda create --name nlp python=3.8
conda activate nlp
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
conda install matplotlib pandas scikit-learn -y
pip install tqdm
pip install transformers
pip install sentencepiece
pip install emoji
pip install nltk
```

## Repository Structure

The ERNIE, RoBERTa and XLNet directories can be downloaded from the following link. Please put them under the corresponding directory.  
https://drive.google.com/drive/folders/1RL8fe4Q6cFrMA9M2vysXNAHUE2C1wIm_?usp=sharing

```bash
├─ data
│  ├─ best_data
│     ├─ fixed_group_test.csv
│     ├─ fixed_group_train.csv
│     └─ fixed_group_valid.csv
│  ├─ 1(utterance+prompt)
│     ├─ fixed_group_test.csv
│     ├─ fixed_group_train.csv
│     └─ fixed_group_valid.csv
│  ├─ ...
│  ├─ ...
│  ├─ ...
│  └─ 1+2+3+4+5+6+7+8+9(utterance+prompt)
│     ├─ fixed_group_test.csv
│     ├─ fixed_group_train.csv
│     └─ fixed_group_valid.csv
├─ train.py
├─ test1.py
├─ test2.py
├─ ensemble1.py
├─ ensemble2.py
├─ parameters.yaml
├─ nlpdatasets.py
├─ total_model.py
├─ fixed_test.csv
├─ fixed_train.csv
├─ fixed_valid.csv
├─ ensemble2
│  └─ RoBERTa+ERNIE+XLNet
│     └─ submission.csv
├─ experiment
│  ├─ ERNIE                           <- You can find in the google cloud link above, please decompress under the experiment directory.
│     ├─ figure
│        ├─ accuracy.png
│        └─ loss.png 
│     ├─ best_model_state.bin
│     ├─ parameters.yaml
│     ├─ record.csv
│     ├─ submission1.csv
│     └─ submission2.csv
│  ├─ RoBERTa                         <- You can find in the google cloud link above, please decompress under the experiment directory.
│     ├─ figure
│        ├─ accuracy.png
│        └─ loss.png 
│     ├─ best_model_state.bin
│     ├─ parameters.yaml
│     ├─ record.csv
│     ├─ submission1.csv
│     └─ submission2.csv
│  └─ XLNet                           <- You can find in the google cloud link above, please decompress under the experiment directory.
│     ├─ figure
│        ├─ accuracy.png
│        └─ loss.png 
│     ├─ best_model_state.bin
│     ├─ parameters.yaml
│     ├─ record.csv
│     ├─ submission1.csv
│     └─ submission2.csv
└─ README.md
```

## Procedure
<p float="center">
  <img src="https://user-images.githubusercontent.com/44439517/172042422-8580fa5a-8bf4-47bc-b41e-ca24d4714d67.png" width="100%" height="100%" hspace="50"/>
</p>

## Data preprocess
In this project, we use the following data preprocess method to deal with the data, and we find that the fifth data preprocess method is the best method in this project. After doing the data preprocess, the sentences were separated into different tokens.


|NO. | Method                           | Before                                                                         | After                                    | 
|----------------------------------|----------------------------------|--------------------------------------------------------------------------------|------------------------------------------|
|1| replace \_comma\_ to ,             | There was a lot of people_comma_ but it only felt like us in the world.        | There was a lot of people, but it only felt like us in the world.  | 
|2| replace / to or                  | /Did you get him a teacher?                              | or Did you get him a teacher?                            | 
|3| replace & to and                 | I believe it's because you miss family & friends         | I believe it's because you miss family and friends       | 
|4| remove emoji                 | I just got new neighbors and they are so loud.,I know there probably isnt much you can do. :/         | I just got new neighbors and they are so loud.,I know there probably isnt much you can do.       | 
|5| restore he's to he is                 | Yeah_comma_ fortunately he's very small so he doesn't have as many joint problems as the bigger dogs I thnik at least.        | Yeah_comma_ fortunately he is very small so he does not have as many joint problems as the bigger dogs I thnik at least.       | 
|6| remove punctuation                 | The love towards my wife is become more and it tends to uncountable now!         | The love towards my wife is become more and it tends to uncountable now       | 
|7| replace integer to #number                 | It's really sleek and fun to drive,I got the new 2018 Honda Accord LX.         | It's really sleek and fun to drive,I got the new #number Honda Accord LX.       | 
|8| remove stopword                 | Was invited to a friends house after work.         | invited to friends house after work.       | 
|9| lemmatize                 | football is a family of team sports that involve, to varying degrees, kicking a ball to score a goal.         | football be a family   of team sport that involve, to vary degree, kick a ball to score a goal. | 

## Model Architecture
In this project, we used the following three pretrained models for transfer learning, which include [**RoBERTa-Large**](https://arxiv.org/pdf/1907.11692.pdf)[1], [**ERNIE 2.0-Large**](https://arxiv.org/pdf/1907.12412.pdf)[2] and [**XLNet**](https://arxiv.org/pdf/1906.08237.pdf)[3].

The framework of these three classification models is as follows **(only for demonstrate)**: the linear layer would be added to the model for transfer learning. 

- ### RoBERTa-Large
```python
class RoBERTa(nn.Module):

    def __init__(self, n_classes):
        super(RoBERTa, self).__init__()
        self.model = RobertaModel.from_pretrained("roberta-large", hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2)
        self.out = nn.Linear(self.model.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        last_hidden_state, pooled_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        return self.out(pooled_output)
```

- ### ERNIE 2.0-Large
```python
class ERNIE(nn.Module):

    def __init__(self, n_classes):
        super(ERNIE, self).__init__()
        self.model = AutoModel.from_pretrained("nghuyong/ernie-2.0-large-en", hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2)
        self.out = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, pooled_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        return self.out(pooled_output)
```

- ### XLNet
```python
class XLNet(nn.Module):

    def __init__(self, n_classes):
        super(XLNet, self).__init__()
        self.model = XLNetModel.from_pretrained("xlnet-large-cased")
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        return self.out(pooled_output[0][:, -1, :])
```

## Training

To train the model, please follow these steps.

### 1. Setting arguments through parameters.yaml file.

You can set the training arguments that you prefer through the parameters.yml file.  

```yaml
model: roberta-large # bert-base, ernie-base, roberta-base, xlnet-base, bert-large, ernie-large, roberta-large, xlnet-large, YOSO
USING_DATA: best_data # 1(utterance+prompt), 2(utterance+prompt)...
EPOCHS: 20 
BATCH_SIZE: 8 # 1 2 4 8(large model) 32(recommend) 64(recommend) 
LR: 2e-6 # 2e-3 2e-5(recommend) 2e-6(large model)
MAX_LEN: 160 # 100 128 160 256 512
FREEZE: [] #[], [embeddings], [encoder], [pooler], [embeddings, encoder], [encoder, pooler], [embeddings, encoder, pooler]...
DROPOUT_RATE: None #None or values
HIDDEN_DROPOUT_PROB: 0.2 #None or values
ATTENTION_PROBS_DROPOUT_PROB: 0.2 #None or values
```

### 2. Input Commend

You don't need to add any argument behind the train.py.

```bash
python train.py
```

### 3. The position of experiment result

```bash
├─ data
├─ ...
├─ ...
├─ experiment
│  ├─ exp1                <- You can find the experiment result in this position.(The directory can be exp1 exp2 exp3, etc. According to the highest directory number).
│     ├─ figure
│        ├─ accuracy.png
│        └─ loss.png 
│     ├─ best_model_state.bin
│     ├─ parameters.yaml
│     ├─ confusion_matrix.csv
│     └─ record.csv
│  ├─ ERNIE
│  ├─ RoBERTa
│  └─ XLNet
└─ README.md
```

## Testing

To generate the submission, please follow these steps.

### 1. Input Commend

These directory should have **parameters.yaml** and **best_model_state.bin** file.  

The argument behind the --directory is the directory under the experiment. The argument behind weight is the weight that needs to multiplicate with predicted probability.  

**Before you do the ensemble method, you should do all commends below.**

The weight of RoBERTa should be 0.4. 
```bash
python test2.py --directory RoBERTa --weight 0.4
```

The weight of ERNIE should be 0.35.
```bash
python test2.py --directory ERNIE --weight 0.35
```

The weight of XLNet should be 0.25.
```bash
python test2.py --directory XLNet --weight 0.25
```

### 2. The position of the testing result

The submission2.csv will be stored under the path of experiment/specified_directory. 

```bash
├─ data
├─ ...
├─ experiment
│  ├─ ERNIE
│  ├─ RoBERTa
│     ├─ figure
│        ├─ accuracy.png
│        └─ loss.png 
│     ├─ best_model_state.bin     <- You should have this file under this directory.
│     ├─ parameters.yaml          <- You should have this file under this directory.
│     ├─ record.csv
│     └─ submission2.csv          <- You can find the testing result in this position. 
│  └─ XLNet
└─ README.md
```

## Ensemble

To integrate different submission, please follow the steps below.

### 1. Input Commend

The argument behind the ensemble2.py is the directory under the experiment. These directory should have the submission2.csv file.

```bash
python ensemble2.py RoBERTa ERNIE XLNet
```

### 2. The position of the ensemble result

The ensemble submission2.csv will be stored under the path of experiment/ensemble2/RoBERTa+ERNIE+XLNet. 

```bash
├─ data
├─ ...
├─ experiment
│  ├─ ERNIE
│     ├─ ...
│     ├─ ...
│     └─ submission2.csv       <- You should have this file under this directory.
│  ├─ RoBERTa
│     ├─ ...
│     ├─ ...
│     └─ submission2.csv       <- You should have this file under this directory.
│  └─ XLNet
│     ├─ ...
│     ├─ ...
│     └─ submission2.csv       <- You should have this file under this directory.
├─ ensemble2                   <- The program will create ensemble2 directory.
│  └─ RoBERTa+ERNIE+XLNet      <- The program will create RoBERTa+ERNIE+XLNet directory.
│     └─ submission.csv        <- You can find the ensemble result in this position. 
└─ README.md
```

## Experiment Result

In this project, we use four experiments to verify our method has the best performance. The value of the accuracy which is shown below is the average of the multiple accuracy values.

### 1. Data Column 

We found that the combination of Utterance and Prompt is suitable for all of the models we use in this project.

|          | Utterance | Prompt | Utterance+Prompt     |
|----------|:-----------:|:--------:|:----------------------:|
| Accuracy | 0.6126    | 0.6137 | **0.6649**           | 

### 2. Data Preprocess Method 

We found that NO.5 is suitable for Roberta and ERNIE, but the combination of NO.3 and NO.5 is suitable for XLNet. Only the method NO. 3 and method NO. 5 is better than nothing to do.

|          | NO. 1   | NO. 2  | NO. 3       | NO. 4  | NO. 5      | NO. 6  | NO. 7  | NO. 8  | NO. 9  | NO. 10 |
|----------|:---------:|:--------:|:-------------:|:--------:|:------------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| Accuracy | 0.6606  | 0.6595 | **0.6635**  | 0.6592 | **0.6649** | 0.6542 | 0.6552 | 0.6238 | 0.6527 | 0.6624 |

### 3. Maximum number of tokens in one sentence

We found that 160 is suitable for RoBERTa and EERNIE, but 256 is more suitable for XLNet.

|          | 100    | 128    | 160    | 256     | 512    |
|:--------:|:------:|:------:|:------:|:-------:|:------:|
| Accuracy | 0.6439 | 0.6458 | 0.6477 | 0.65308 | 0.6480 |

### 4. Pretrained Model

We found that large model is better than base model, and the top three models in this project is RoBERTa-large, ERNIE-2.0-large and XLNet-large. 

|          | BERT-BASE | RoBERTa-BASE | ERNIE-BASE | XLNet-BASE | BERT-LARGE | RoBERTa-LARGE | ERNIE-LARGE | XLNet-LARGE | YOSO   |
|----------|:-----------:|:--------------:|:------------:|:------------:|:------------:|:---------------:|:-------------:|:-------------:|:--------:|
| Accuracy | 0.5227    | 0.6018       | 0.6181     | 0.6025     | 0.6166     |**0.6649**        | **0.6397**      | **0.6379**      | 0.5567 |

### 5. Dropout

We found that dropout are suitable for ERNIE and XLNet. In contrast, dropout is not suitable for RoBERTa. The hidden dropout and attention dropout are suitable for RoBERTa and ERNIE.

|          | None | dropout |  hidden_dropout_prob | attention_probs_dropout_prob | hidden_dropout_prob and attention_probs_dropout_prob    |
|----------|:--------:|:-----------:|:-----------------------:|:--------------------------------:|:--------------------------------------------------:|
| Accuracy | 0.6445 | 0.6477    | 0.6471                | 0.6451                         | **0.6524**                                       |

### 6. Ensemble

In this section, we use ensemble method with the top 3 models. 

|          | RoBERTa-LARGE | ERNIE-LARGE | XLNet-LARGE | Ensemble |
|----------|:---------------:|:-------------:|:-------------:|:-------------------:|
| F1-score | 0.64239        |    0.62535   |      0.61961       |        0.65633           |


## Reference
[1]	Y. Liu et al., “RoBERTa: A Robustly Optimized BERT Pretraining Approach,” arXiv, arXiv:1907.11692, Jul. 2019. doi: 10.48550/arXiv.1907.11692.  

[2]	Y. Sun et al., “ERNIE 2.0: A Continual Pre-training Framework for Language Understanding,” arXiv, arXiv:1907.12412, Nov. 2019. doi: 10.48550/arXiv.1907.12412.  

[3]	Z. Yang, Z. Dai, Y. Yang, J. Carbonell, R. Salakhutdinov, and Q. V. Le, “XLNet: Generalized Autoregressive Pretraining for Language Understanding,” arXiv, arXiv:1906.08237, Jan. 2020. doi: 10.48550/arXiv.1906.08237.
