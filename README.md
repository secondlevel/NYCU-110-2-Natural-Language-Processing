# NYCU-110-2-Natural-Language-Processing

This project is to identify emotion through corpus, and it is related to [**kaggle competition**](https://www.kaggle.com/competitions/nycu-nlp110/overview). The model in this project was built by [**HuggingFace**](https://huggingface.co/).  
HuggingFace toolkit is developed by **Pytorch**.

In addition, please refer to the following report link for detailed report and description of the experimental results.

<p float="center">
  <img src="https://user-images.githubusercontent.com/44439517/172013625-018a5f33-8b3d-4a97-8322-1103062d46c8.png" width="65%" height="65%" hspace="150"/>
</p>

## Hardware

In this project, the following machine was used to train the emotion classification model.

|                 | Operating System           | CPU                                      | GPU                         | 
|-----------------|----------------------------|------------------------------------------|-----------------------------|
| First machine   | Ubuntu 20.04.3 LTS         | Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz  | NVIDIA GeForce GTX TITAN X  |
| Second machine  | Ubuntu 18.04.5 LTS         | Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz | NVIDIA GeForce GTX 1080     |
| Third machine   | Ubuntu 20.04.3 LTS         | AMD Ryzen 5 5600X 6-Core Processor       | NVIDIA GeForce RTX 2080 Ti  |


## Requirement

In this project, we use conda and pip toolkit to build the environment.

The following two **options** are provided for building the environment.
  
- #### First Option
```bash
conda env create -f environment.yml
```

- #### Second Option 
```bash
conda nlp --name nlp python=3.8
conda activate nlp
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
conda install matplotlib pandas scikit-learn -y
pip install tqdm
pip install transformer
pip install sentencepiece
pip install emoji
pip install nltk
```

## Repository Structure

```
├─ ALL_model.py
├─ experiment
│  ├─ ERNIE
│     ├─ figure
│        ├─ accuracy.png
│        └─ loss.png 
│     ├─ best_model_state.bin
│     ├─ parameters.yaml
│     ├─ record.csv
│     └─ submission.csv
│  ├─ RoBERTa
│     ├─ figure
│        ├─ accuracy.png
│        └─ loss.png 
│     ├─ best_model_state.bin
│     ├─ parameters.yaml
│     ├─ record.csv
│     └─ submission.csv
│  └─ XLNet
│     ├─ figure
│        ├─ accuracy.png
│        └─ loss.png 
│     ├─ best_model_state.bin
│     ├─ parameters.yaml
│     ├─ record.csv
│     └─ submission.csv
├─ parameters.yaml
├─ nlpdatasets.py
├─ total_model.py
├─ train.py
├─ test.py
├─ fixed_test.csv
├─ fixed_train.csv
├─ fixed_valid.csv
└─ README.md
```

## Data preprocess

## Model Architecture
In this project, we used the following three pretrained models for transfer learning, which include [**RoBERTa-Large**](https://arxiv.org/pdf/1907.11692.pdf)[1], [**ERNIE 2.0-Large**](https://arxiv.org/pdf/1907.12412.pdf)[2] and [**XLNet**](https://arxiv.org/pdf/1906.08237.pdf)[3].

The framework of these three classification models is as follows **(only for demonstration)**: the linear layer would be added to model for transfer learning. 

- #### RoBERTa-Large
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

- #### ERNIE 2.0-Large
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

- #### XLNet
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


## Testing

## Ablation Study

## Reference
[1]	Y. Liu et al., “RoBERTa: A Robustly Optimized BERT Pretraining Approach,” arXiv, arXiv:1907.11692, Jul. 2019. doi: 10.48550/arXiv.1907.11692.  

[2]	Y. Sun et al., “ERNIE 2.0: A Continual Pre-training Framework for Language Understanding,” arXiv, arXiv:1907.12412, Nov. 2019. doi: 10.48550/arXiv.1907.12412.  

[3]	Z. Yang, Z. Dai, Y. Yang, J. Carbonell, R. Salakhutdinov, and Q. V. Le, “XLNet: Generalized Autoregressive Pretraining for Language Understanding,” arXiv, arXiv:1906.08237, Jan. 2020. doi: 10.48550/arXiv.1906.08237.
