import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

import torch
import numpy as np
import pandas as pd
# import seaborn as sns

from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from nlpdatasets import GPReviewDataset
from tqdm import tqdm
import pandas as pd
import yaml
from total_model import *
from transformers import RobertaTokenizer, RobertaModel
import argparse


probability = []

parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=str, default='',
                    help='testing directory')
parser.add_argument('--weight', type=float, default='0.4',
                    help='model probability weight')

opt = parser.parse_args()

with open("experiment/"+opt.directory+"/parameters.yaml", 'r') as f:
    paramenter = yaml.full_load(f)

give_parameter(paramenter)

RANDOM_SEED = 42
EPOCHS = paramenter["EPOCHS"]
BATCH_SIZE = paramenter["BATCH_SIZE"]
MAX_LEN = paramenter["MAX_LEN"]
LR = float(paramenter["LR"])

model_name = ["bert-base", "ernie-base", "roberta-base", "xlnet-base",
              "bert-large", "ernie-large", "roberta-large", "xlnet-large", "YOSO"]

used_model = [bert_base, ernie_base, roberta_base, xlnet_base,
              bert_large, ernie_large, roberta_large, xlnet_large, YOSO]

used_token = [BertTokenizer, AutoTokenizer, RobertaTokenizer, XLNetTokenizer,
              BertTokenizer, AutoTokenizer, RobertaTokenizer, XLNetTokenizer, AutoTokenizer]

used_model_pretrained_name = ["bert-base-cased",
                              "nghuyong/ernie-2.0-en", "roberta-base", 'xlnet-base-cased', "bert-large-cased", "nghuyong/ernie-2.0-large-en", 'roberta-large', 'xlnet-large-cased', "uw-madison/yoso-4096"]

SentimentClassifier = dict(zip(model_name, used_model))
USING_TOKEN = dict(zip(model_name, used_token))
USING_PRETRAINED = dict(zip(model_name, used_model_pretrained_name))


def create_data_loader(df, tokenizer, max_len, batch_size):

    ds = GPReviewDataset(
        reviews=df.data.to_numpy(),
        targets=None,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=4)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df_test = pd.DataFrame(pd.read_csv(
    "data/"+paramenter["USING_DATA"]+"/fixed_group_test.csv"))
total_count = list(df_test["count"])

tokenizer = USING_TOKEN[paramenter["model"]].from_pretrained(
    USING_PRETRAINED[paramenter["model"]])

test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

df_train = pd.DataFrame(pd.read_csv(
    "data/"+paramenter["USING_DATA"]+"/fixed_group_train.csv"))
class_names = np.unique(np.array(df_train.label))

model = SentimentClassifier[paramenter["model"]](len(class_names))
model = model.to(device)
model.load_state_dict(torch.load(
    "experiment/"+opt.directory+"/best_model_state.bin"))


def eval_model(model, data_loader, device, n_examples):

    model = model.eval()
    all_prediction = []
    global probability

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            for x in (opt.weight*(outputs.cpu().numpy())):
                probability.append(x.tolist())

            _, preds = torch.max(outputs, dim=1)
            print(preds.shape)
            all_prediction += preds.tolist()

    return all_prediction


all_prediction = eval_model(
    model,
    test_data_loader,
    device,
    len(df_test)
)

repeat_predict = []

for idx, count in enumerate(total_count):
    for _ in range(int(count)):
        repeat_predict.append(probability[idx])

df = pd.DataFrame(columns=list(range(32)), data=repeat_predict)
df.to_csv("experiment/"+opt.directory+"/submission2.csv", index=False)
