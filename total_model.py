from transformers import BertModel, BertTokenizer
from transformers import AutoTokenizer, AutoModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import XLNetTokenizer, XLNetModel
from transformers import AutoTokenizer, YosoModel

from torch import nn
import yaml

data = None


def give_parameter(information):
    global data
    data = information


class bert_base(nn.Module):

    def __init__(self, n_classes):

        super(bert_base, self).__init__()

        if (data["HIDDEN_DROPOUT_PROB"] == "None") and (data["ATTENTION_PROBS_DROPOUT_PROB"] == "None"):
            self.model = BertModel.from_pretrained("bert-base-cased")

        if (data["HIDDEN_DROPOUT_PROB"] != "None") and (data["ATTENTION_PROBS_DROPOUT_PROB"] != "None"):
            self.model = BertModel.from_pretrained("bert-base-cased", hidden_dropout_prob=float(data["HIDDEN_DROPOUT_PROB"]),
                                                   attention_probs_dropout_prob=float(data["ATTENTION_PROBS_DROPOUT_PROB"]))

        elif (data["HIDDEN_DROPOUT_PROB"] != "None"):
            self.model = BertModel.from_pretrained(
                "bert-base-cased", hidden_dropout_prob=float(data["HIDDEN_DROPOUT_PROB"]))

        elif (data["ATTENTION_PROBS_DROPOUT_PROB"] != "None"):
            self.model = BertModel.from_pretrained(
                "bert-base-cased", attention_probs_dropout_prob=float(data["ATTENTION_PROBS_DROPOUT_PROB"]))

        if data["DROPOUT_RATE"] != "None":
            self.drop = nn.Dropout(p=float(data["DROPOUT_RATE"]))

        for name, child in self.model.named_children():
            if name in data["FREEZE"]:
                print(name, "is freeze")
                for param in child.parameters():
                    param.requires_grad = False
            else:
                print(name, "is unfreeze")
                for param in child.parameters():
                    param.requires_grad = True

        self.out = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, pooled_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        if data["DROPOUT_RATE"] != "None":
            pooled_output = self.drop(pooled_output)
        return self.out(pooled_output)


class ernie_base(nn.Module):

    def __init__(self, n_classes):

        super(ernie_base, self).__init__()

        if (data["HIDDEN_DROPOUT_PROB"] == "None") and (data["ATTENTION_PROBS_DROPOUT_PROB"] == "None"):
            self.model = AutoModel.from_pretrained("nghuyong/ernie-2.0-en")

        if (data["HIDDEN_DROPOUT_PROB"] != "None") and (data["ATTENTION_PROBS_DROPOUT_PROB"] != "None"):
            self.model = AutoModel.from_pretrained("nghuyong/ernie-2.0-en", hidden_dropout_prob=float(data["HIDDEN_DROPOUT_PROB"]),
                                                   attention_probs_dropout_prob=float(data["ATTENTION_PROBS_DROPOUT_PROB"]))

        elif (data["HIDDEN_DROPOUT_PROB"] != "None"):
            self.model = AutoModel.from_pretrained(
                "nghuyong/ernie-2.0-en", hidden_dropout_prob=float(data["HIDDEN_DROPOUT_PROB"]))

        elif (data["ATTENTION_PROBS_DROPOUT_PROB"] != "None"):
            self.model = AutoModel.from_pretrained(
                "nghuyong/ernie-2.0-en", attention_probs_dropout_prob=float(data["ATTENTION_PROBS_DROPOUT_PROB"]))

        if data["DROPOUT_RATE"] != "None":
            self.drop = nn.Dropout(p=float(data["DROPOUT_RATE"]))

        for name, child in self.model.named_children():
            if name in data["FREEZE"]:
                print(name, "is freeze")
                for param in child.parameters():
                    param.requires_grad = False
            else:
                print(name, "is unfreeze")
                for param in child.parameters():
                    param.requires_grad = True

        self.out = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, pooled_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        if data["DROPOUT_RATE"] != "None":
            pooled_output = self.drop(pooled_output)

        return self.out(pooled_output)


class roberta_base(nn.Module):

    def __init__(self, n_classes):

        super(roberta_base, self).__init__()

        if (data["HIDDEN_DROPOUT_PROB"] == "None") and (data["ATTENTION_PROBS_DROPOUT_PROB"] == "None"):
            self.model = RobertaModel.from_pretrained("roberta-base")

        if (data["HIDDEN_DROPOUT_PROB"] != "None") and (data["ATTENTION_PROBS_DROPOUT_PROB"] != "None"):
            self.model = RobertaModel.from_pretrained("roberta-base", hidden_dropout_prob=float(data["HIDDEN_DROPOUT_PROB"]),
                                                      attention_probs_dropout_prob=float(data["ATTENTION_PROBS_DROPOUT_PROB"]))

        elif (data["HIDDEN_DROPOUT_PROB"] != "None"):
            self.model = RobertaModel.from_pretrained(
                "roberta-base", hidden_dropout_prob=float(data["HIDDEN_DROPOUT_PROB"]))

        elif (data["ATTENTION_PROBS_DROPOUT_PROB"] != "None"):
            self.model = RobertaModel.from_pretrained(
                "roberta-base", attention_probs_dropout_prob=float(data["ATTENTION_PROBS_DROPOUT_PROB"]))

        if data["DROPOUT_RATE"] != "None":
            self.drop = nn.Dropout(p=float(data["DROPOUT_RATE"]))

        for name, child in self.model.named_children():
            if name in data["FREEZE"]:
                print(name, "is freeze")
                for param in child.parameters():
                    param.requires_grad = False
            else:
                print(name, "is unfreeze")
                for param in child.parameters():
                    param.requires_grad = True

        self.out = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, pooled_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        if data["DROPOUT_RATE"] != "None":
            pooled_output = self.drop(pooled_output)

        return self.out(pooled_output)


class xlnet_base(nn.Module):

    def __init__(self, n_classes):

        super(xlnet_base, self).__init__()

        if (data["HIDDEN_DROPOUT_PROB"] == "None") and (data["ATTENTION_PROBS_DROPOUT_PROB"] == "None"):
            self.model = XLNetModel.from_pretrained('xlnet-base-cased')

        if (data["HIDDEN_DROPOUT_PROB"] != "None") and (data["ATTENTION_PROBS_DROPOUT_PROB"] != "None"):
            self.model = XLNetModel.from_pretrained('xlnet-base-cased')

        elif (data["HIDDEN_DROPOUT_PROB"] != "None"):
            self.model = XLNetModel.from_pretrained(
                'xlnet-base-cased', hidden_dropout_prob=float(data["HIDDEN_DROPOUT_PROB"]))

        elif (data["ATTENTION_PROBS_DROPOUT_PROB"] != "None"):
            self.model = XLNetModel.from_pretrained(
                'xlnet-base-cased', attention_probs_dropout_prob=float(data["ATTENTION_PROBS_DROPOUT_PROB"]))

        if data["DROPOUT_RATE"] != "None":
            self.drop = nn.Dropout(p=float(data["DROPOUT_RATE"]))

        for name, child in self.model.named_children():
            if name in data["FREEZE"]:
                print(name, "is freeze")
                for param in child.parameters():
                    param.requires_grad = False
            else:
                print(name, "is unfreeze")
                for param in child.parameters():
                    param.requires_grad = True

        self.out = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        if data["DROPOUT_RATE"] != "None":
            pooled_output = self.drop(pooled_output[0][:, -1, :])

            return self.out(pooled_output)

        else:
            return self.out(pooled_output[0][:, -1, :])


class bert_large(nn.Module):

    def __init__(self, n_classes):

        super(bert_large, self).__init__()

        if (data["HIDDEN_DROPOUT_PROB"] == "None") and (data["ATTENTION_PROBS_DROPOUT_PROB"] == "None"):
            self.model = BertModel.from_pretrained("bert-large-cased")

        if (data["HIDDEN_DROPOUT_PROB"] != "None") and (data["ATTENTION_PROBS_DROPOUT_PROB"] != "None"):
            self.model = BertModel.from_pretrained("bert-large-cased", hidden_dropout_prob=float(data["HIDDEN_DROPOUT_PROB"]),
                                                   attention_probs_dropout_prob=float(data["ATTENTION_PROBS_DROPOUT_PROB"]))

        elif (data["HIDDEN_DROPOUT_PROB"] != "None"):
            self.model = BertModel.from_pretrained(
                "bert-large-cased", hidden_dropout_prob=float(data["HIDDEN_DROPOUT_PROB"]))

        elif (data["ATTENTION_PROBS_DROPOUT_PROB"] != "None"):
            self.model = BertModel.from_pretrained(
                "bert-large-cased", attention_probs_dropout_prob=float(data["ATTENTION_PROBS_DROPOUT_PROB"]))

        if data["DROPOUT_RATE"] != "None":
            self.drop = nn.Dropout(p=float(data["DROPOUT_RATE"]))

        for name, child in self.model.named_children():
            if name in data["FREEZE"]:
                print(name, "is freeze")
                for param in child.parameters():
                    param.requires_grad = False
            else:
                print(name, "is unfreeze")
                for param in child.parameters():
                    param.requires_grad = True

        self.out = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, pooled_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        if data["DROPOUT_RATE"] != "None":
            pooled_output = self.drop(pooled_output)
        return self.out(pooled_output)


class ernie_large(nn.Module):

    def __init__(self, n_classes):

        super(ernie_large, self).__init__()

        if (data["HIDDEN_DROPOUT_PROB"] == "None") and (data["ATTENTION_PROBS_DROPOUT_PROB"] == "None"):
            self.model = AutoModel.from_pretrained(
                "nghuyong/ernie-2.0-large-en")

        if (data["HIDDEN_DROPOUT_PROB"] != "None") and (data["ATTENTION_PROBS_DROPOUT_PROB"] != "None"):
            self.model = AutoModel.from_pretrained("nghuyong/ernie-2.0-large-en", hidden_dropout_prob=float(data["HIDDEN_DROPOUT_PROB"]),
                                                   attention_probs_dropout_prob=float(data["ATTENTION_PROBS_DROPOUT_PROB"]))

        elif (data["HIDDEN_DROPOUT_PROB"] != "None"):
            self.model = AutoModel.from_pretrained(
                "nghuyong/ernie-2.0-large-en", hidden_dropout_prob=float(data["HIDDEN_DROPOUT_PROB"]))

        elif (data["ATTENTION_PROBS_DROPOUT_PROB"] != "None"):
            self.model = AutoModel.from_pretrained(
                "nghuyong/ernie-2.0-large-en", attention_probs_dropout_prob=float(data["ATTENTION_PROBS_DROPOUT_PROB"]))

        if data["DROPOUT_RATE"] != "None":
            self.drop = nn.Dropout(p=float(data["DROPOUT_RATE"]))

        for name, child in self.model.named_children():
            if name in data["FREEZE"]:
                print(name, "is freeze")
                for param in child.parameters():
                    param.requires_grad = False
            else:
                print(name, "is unfreeze")
                for param in child.parameters():
                    param.requires_grad = True

        self.out = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, pooled_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        if data["DROPOUT_RATE"] != "None":
            pooled_output = self.drop(pooled_output)

        return self.out(pooled_output)


class roberta_large(nn.Module):

    def __init__(self, n_classes):

        super(roberta_large, self).__init__()

        if (data["HIDDEN_DROPOUT_PROB"] == "None") and (data["ATTENTION_PROBS_DROPOUT_PROB"] == "None"):
            self.model = RobertaModel.from_pretrained('roberta-large')

        if (data["HIDDEN_DROPOUT_PROB"] != "None") and (data["ATTENTION_PROBS_DROPOUT_PROB"] != "None"):
            self.model = RobertaModel.from_pretrained('roberta-large', hidden_dropout_prob=float(data["HIDDEN_DROPOUT_PROB"]),
                                                      attention_probs_dropout_prob=float(data["ATTENTION_PROBS_DROPOUT_PROB"]))

        elif (data["HIDDEN_DROPOUT_PROB"] != "None"):
            self.model = RobertaModel.from_pretrained(
                'roberta-large', hidden_dropout_prob=float(data["HIDDEN_DROPOUT_PROB"]))

        elif (data["ATTENTION_PROBS_DROPOUT_PROB"] != "None"):
            self.model = RobertaModel.from_pretrained(
                'roberta-large', attention_probs_dropout_prob=float(data["ATTENTION_PROBS_DROPOUT_PROB"]))

        if data["DROPOUT_RATE"] != "None":
            self.drop = nn.Dropout(p=float(data["DROPOUT_RATE"]))

        for name, child in self.model.named_children():
            if name in data["FREEZE"]:
                print(name, "is freeze")
                for param in child.parameters():
                    param.requires_grad = False
            else:
                print(name, "is unfreeze")
                for param in child.parameters():
                    param.requires_grad = True

        self.out = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, pooled_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        if data["DROPOUT_RATE"] != "None":
            pooled_output = self.drop(pooled_output)

        return self.out(pooled_output)


class xlnet_large(nn.Module):

    def __init__(self, n_classes):

        super(xlnet_large, self).__init__()

        if (data["HIDDEN_DROPOUT_PROB"] == "None") and (data["ATTENTION_PROBS_DROPOUT_PROB"] == "None"):
            self.model = XLNetModel.from_pretrained('xlnet-large-cased')

        elif (data["HIDDEN_DROPOUT_PROB"] != "None"):
            self.model = XLNetModel.from_pretrained(
                'xlnet-large-cased', hidden_dropout_prob=float(data["HIDDEN_DROPOUT_PROB"]))

        elif (data["ATTENTION_PROBS_DROPOUT_PROB"] != "None"):
            self.model = XLNetModel.from_pretrained(
                'xlnet-large-cased', attention_probs_dropout_prob=float(data["ATTENTION_PROBS_DROPOUT_PROB"]))

        if data["DROPOUT_RATE"] != "None":
            self.drop = nn.Dropout(p=float(data["DROPOUT_RATE"]))

        for name, child in self.model.named_children():
            if name in data["FREEZE"]:
                print(name, "is freeze")
                for param in child.parameters():
                    param.requires_grad = False
            else:
                print(name, "is unfreeze")
                for param in child.parameters():
                    param.requires_grad = True

        self.out = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        if data["DROPOUT_RATE"] != "None":
            pooled_output = self.drop(pooled_output[0][:, -1, :])

            return self.out(pooled_output)

        else:
            return self.out(pooled_output[0][:, -1, :])


class YOSO(nn.Module):

    def __init__(self, n_classes):

        super(YOSO, self).__init__()

        if (data["HIDDEN_DROPOUT_PROB"] == "None") and (data["ATTENTION_PROBS_DROPOUT_PROB"] == "None"):
            self.model = YosoModel.from_pretrained("uw-madison/yoso-4096")

        if (data["HIDDEN_DROPOUT_PROB"] != "None") and (data["ATTENTION_PROBS_DROPOUT_PROB"] != "None"):
            self.model = YosoModel.from_pretrained("uw-madison/yoso-4096", hidden_dropout_prob=float(data["HIDDEN_DROPOUT_PROB"]),
                                                   attention_probs_dropout_prob=float(data["ATTENTION_PROBS_DROPOUT_PROB"]))

        elif (data["HIDDEN_DROPOUT_PROB"] != "None"):
            self.model = YosoModel.from_pretrained(
                "uw-madison/yoso-4096", hidden_dropout_prob=float(data["HIDDEN_DROPOUT_PROB"]))

        elif (data["ATTENTION_PROBS_DROPOUT_PROB"] != "None"):
            self.model = YosoModel.from_pretrained(
                "uw-madison/yoso-4096", attention_probs_dropout_prob=float(data["ATTENTION_PROBS_DROPOUT_PROB"]))

        if data["DROPOUT_RATE"] != "None":
            self.drop = nn.Dropout(p=float(data["DROPOUT_RATE"]))

        for name, child in self.model.named_children():
            if name in data["FREEZE"]:
                print(name, "is freeze")
                for param in child.parameters():
                    param.requires_grad = False
            else:
                print(name, "is unfreeze")
                for param in child.parameters():
                    param.requires_grad = True

        self.out = nn.Linear(self.model.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        if data["DROPOUT_RATE"] != "None":
            pooled_output = self.drop(pooled_output)

        return self.out(pooled_output[0])[:, -1, :]
