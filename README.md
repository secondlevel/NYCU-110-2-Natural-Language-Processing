# NYCU-110-2-Natural-Language-Processing

This project is to identify emotion through corpus, and it is related to [**kaggle competition**](https://www.kaggle.com/competitions/nycu-nlp110/overview). The model in this project was built by [**HuggingFace**](https://huggingface.co/). HuggingFace toolkit is developed by Pytorch.

In addition, please refer to the following report link for detailed report and description of the experimental results.

<p float="center">
  <img src="https://user-images.githubusercontent.com/44439517/172013625-018a5f33-8b3d-4a97-8322-1103062d46c8.png" title="loss curve" width="65%" height="65%" hspace="150"/>
</p>

## Hardware

In this project, the following machine was used to train the emotion classification model.

|                 | Operating System           | CPU                                      | GPU                         | 
|-----------------|----------------------------|------------------------------------------|-----------------------------|
| First machine   | Ubuntu 20.04.3 LTS         | Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz  | NVIDIA GeForce GTX TITAN X  |
| Second machine  | Ubuntu 18.04.5 LTS         | Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz | NVIDIA GeForce GTX 1080     |
| Third machine   | Ubuntu 20.04.3 LTS         | AMD Ryzen 5 5600X 6-Core Processor       | NVIDIA GeForce RTX 2080 Ti  |


## Requirement

In this project, we use conda and pip toolkit to build the execution environment.

The following two **options** can be used to build an execution environment
  
- #### First Option
```bash=
conda env create -f environment.yml
```

- #### Second Option 
```bash=
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
