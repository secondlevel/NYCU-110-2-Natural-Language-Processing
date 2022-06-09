import transformers
from transformers import AdamW, get_linear_schedule_with_warmup

import os
import torch
import yaml
import shutil
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

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
from transformers import BertModel, BertTokenizer
from transformers import AutoTokenizer, AutoModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import XLNetTokenizer, XLNetModel
from transformers import AutoTokenizer, YosoModel
# from draw_picture import draw_history
from total_model import give_parameter, bert_base, ernie_base, roberta_base, xlnet_base, bert_large, ernie_large, roberta_large, xlnet_large, YOSO

with open('parameters.yaml', 'r') as f:
    paramenter = yaml.full_load(f)

give_parameter(paramenter)

RANDOM_SEED = 42
EPOCHS = paramenter["EPOCHS"]
BATCH_SIZE = paramenter["BATCH_SIZE"]
MAX_LEN = paramenter["MAX_LEN"]
LR = float(paramenter["LR"])

y_pred_All_val_batch = []
y_true_All_val_batch = []

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


def create_directory(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)


def draw_history(dirname):

    df = pd.DataFrame(pd.read_csv(dirname + "/record.csv"))
    train_loss = list(df["train_loss"])
    validation_loss = list(df["validation_loss"])
    training_accuracy = list(df["training_accuracy"])
    validation_accuracy = list(df["validation_accuracy"])

    plt.plot(list(range(1, len(training_accuracy)+1)),
             training_accuracy, label='train accuracy')
    plt.plot(list(range(1, len(validation_accuracy)+1)),
             validation_accuracy, label='validation accuracy')
    plt.title('Training history')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 1])
    plt.savefig(dirname + "/figure/accuracy.png")

    plt.close()

    plt.plot(list(range(1, len(train_loss)+1)),
             train_loss, label='train loss')
    plt.plot(list(range(1, len(validation_loss)+1)),
             validation_loss, label='validation loss')
    plt.title('Training history')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim([0, 5])
    plt.savefig(dirname + "/figure/loss.png")

    plt.close()


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):

    model = model.train()
    losses = []
    correct_predictions = 0

    loop = tqdm(enumerate(data_loader), total=len(data_loader))
    for index, d in loop:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        loop.set_postfix(loss=loss.item(), accuracy=(torch.sum(
            preds == targets)/len(preds)).item())

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):

    model = model.eval()
    losses = []
    correct_predictions = 0
    global y_pred_All_val_batch
    global y_true_All_val_batch

    with torch.no_grad():
        loop = tqdm(enumerate(data_loader), total=len(data_loader))
        for index, d in loop:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs, dim=1)

            y_pred_All_val_batch += (preds.cpu().numpy().tolist())
            y_true_All_val_batch += (targets.cpu().numpy().tolist())

            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
            loop.set_postfix(loss=loss.item(), accuracy=(torch.sum(
                preds == targets)/len(preds)).item())

    return correct_predictions.double() / n_examples, np.mean(losses)


def create_data_loader(df, tokenizer, max_len, batch_size):

    ds = GPReviewDataset(
        reviews=df.data.to_numpy(),
        targets=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=4)


def plot_confusion_matrix_figure(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(round(cm[i, j], 1), fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def plot_confusion_matrix(y_pred, y_true, class_num):

    target_names = list(range(class_num))
    plt.figure()
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(8, 6)
    cnf_matrix = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix_figure(
        cnf_matrix, classes=target_names, normalize=True, title='confusion matrix')
    plt.savefig("experiment/exp"+str(experiment) +
                "/figure/confusion_matrix.jpg", dpi=100)
    # plt.show()
    plt.close()


def csv_confusion_matrix(y_pred, y_true, class_num):
    cnf_matrix = confusion_matrix(y_true, y_pred)
    df_matrix = pd.DataFrame(columns=list(range(class_num)), data=cnf_matrix)
    df_matrix.to_csv("experiment/exp"+str(experiment) +
                     "/confusion_matrix.csv", index=True)


if os.path.exists("experiment") == False:
    os.mkdir("experiment")


if len(os.listdir("experiment")) > 0:
    create_exp = False
    for file in os.listdir("experiment"):
        if "exp" in file:
            create_exp = True
    if create_exp:
        experiment = (sorted([int(x.replace("exp", ""))
                              for x in os.listdir("experiment") if "exp" in x])[-1])+1
    else:
        experiment = 1
else:
    experiment = 1

create_directory("experiment/exp"+str(experiment))
create_directory("experiment/exp"+str(experiment)+"/figure")

shutil.copy("parameters.yaml", "experiment/exp" +
            str(experiment)+"/parameters.yaml")

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df_train = pd.DataFrame(pd.read_csv(
    "data/"+paramenter["USING_DATA"]+"/fixed_group_train.csv"))
df_val = pd.DataFrame(pd.read_csv(
    "data/"+paramenter["USING_DATA"]+"/fixed_group_valid.csv"))
class_names = np.unique(np.array(df_train.label))

tokenizer = USING_TOKEN[paramenter["model"]].from_pretrained(
    USING_PRETRAINED[paramenter["model"]])

train_data_loader = create_data_loader(
    df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)

data = next(iter(train_data_loader))

print(data.keys())
print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['targets'].shape)

model = SentimentClassifier[paramenter["model"]](len(class_names))
model = model.to(device)

input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)

print(input_ids.shape)  # batch size x seq length
print(attention_mask.shape)

optimizer = AdamW(model.parameters(), lr=LR, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)

history = defaultdict(list)
best_accuracy = 0

df_record = pd.DataFrame(columns=["train_loss",
                                  "validation_loss", "training_accuracy", "validation_accuracy"])
df_record.to_csv("experiment/exp"+str(experiment)+"/record.csv", index=False)

for epoch in range(EPOCHS):

    df_record = pd.read_csv("experiment/exp"+str(experiment)+"/record.csv")
    new_record_data = list(df_record.values)
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(df_train)
    )

    print(f'Train loss {train_loss} accuracy {train_acc}')
    val_acc, val_loss = eval_model(
        model,
        val_data_loader,
        loss_fn,
        device,
        len(df_val)
    )

    print(f'Val loss {val_loss} accuracy {val_acc}')

    if val_acc > best_accuracy:
        torch.save(model.state_dict(), "experiment/exp" +
                   str(experiment)+"/best_model_state.bin")
        csv_confusion_matrix(y_pred_All_val_batch,
                             y_true_All_val_batch, len(class_names))
        best_accuracy = val_acc

    new_record_data.append(
        [train_loss, val_loss, train_acc.item(), val_acc.item()])
    df_record = pd.DataFrame(columns=df_record.columns, data=new_record_data)
    df_record.to_csv("experiment/exp"+str(experiment) +
                     "/record.csv", index=False)
    draw_history("experiment/exp"+str(experiment))v
