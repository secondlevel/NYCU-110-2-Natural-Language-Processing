import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import emoji
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
import re
import os
import shutil
from itertools import combinations
from string import punctuation

# nltk.download('stopwords')


def create_directory(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def remove_emoji(process):
    face_list = [":D", ":)", ":(", ":/", ":P", "<3", ";)"]
    for face in face_list:
        process = process.replace(face, "")
    process_string = emoji.demojize(process)
    return process_string


def restore_subject(process):
    pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)
    pat_s = re.compile("(?<=[a-zA-Z])\'s")
    pat_s2 = re.compile("(?<=s)\'s?")
    pat_not = re.compile("(?<=[a-zA-Z])\'t")
    pat_would = re.compile("(?<=[a-zA-Z])\'d")
    pat_will = re.compile("(?<=[a-zA-Z])\'ll")
    pat_am = re.compile("(?<=[I|i])\'m")
    pat_are = re.compile("(?<=[a-zA-Z])\'re")
    pat_ve = re.compile("(?<=[a-zA-Z])\'ve")

    process = pat_is.sub(r"\1 is", process)
    process = pat_s.sub("", process)
    process = pat_s2.sub("", process)
    process = pat_not.sub(" not", process)
    process = pat_would.sub(" would", process)
    process = pat_will.sub(" will", process)
    process = pat_am.sub(" am", process)
    process = pat_are.sub(" are", process)
    process = pat_ve.sub(" have", process)
    process = process.replace('\'', ' ')
    process_string = re.sub("\d+(?:st|nd|rd|th)", "#order", process)
    return process_string


def remove_punctuation(process):
    process_string = re.sub(r'[^a-zA-Z0-9]', ' ', process)
    return process_string


def replace_number(process):
    process_string = ""
    tokens = nltk.word_tokenize(process)
    tagged_sent = nltk.pos_tag(tokens)
    for tag in tagged_sent:
        if tag[0].isdigit():
            process_string += "#number "
        else:
            process_string += (tag[0]+" ")
    return process_string


def remove_stopword(process):
    process_string = ""
    nltk_stopwords = nltk.corpus.stopwords.words('english')
    tokens = nltk.word_tokenize(process)
    tagged_sent = nltk.pos_tag(tokens)
    for tag in tagged_sent:
        if (tag[0] not in nltk_stopwords):
            process_string += (tag[0]+" ")
    return process_string


def lemmatize(process):
    process_string = ""
    wnl = WordNetLemmatizer()
    tokens = nltk.word_tokenize(process)
    process = ""
    tagged_sent = nltk.pos_tag(tokens)
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        process_string += (wnl.lemmatize(tag[0], pos=wordnet_pos) + " ")
    return process_string


def string_process(content, used_list):

    process = content

    # replace _comma_ to ,
    if "1" in used_list:
        process = process.replace("_comma_", ",")

    # replace / to or
    if "2" in used_list:
        process = process.replace('/', ' or ')

    # replace & to and
    if "3" in used_list:
        process = process.replace('&', ' and ')

    # remove emoji
    if "4" in used_list:
        process = remove_emoji(process)

    # he's -> he is
    if "5" in used_list:
        process = restore_subject(process)

    # removing punctuation
    if "6" in used_list:
        process = remove_punctuation(process)

    # replace int to #number
    if "7" in used_list:
        process = replace_number(process)

    # remove stopword
    if "8" in used_list:
        process = remove_stopword(process)

    # lemmatize
    if "9" in used_list:
        process = lemmatize(process)

    if len(process) > 0 and process[-1] != '.':
        process += '.'

    return process


def group_by(df, mode, conversion, used_list, dirname):

    total_data = []
    total_label = []
    total_count = []
    conv_id = list(df.conv_id)
    prompt = list(df.prompt)
    utterance = list(df.utterance)
    if mode != "test":
        label = list(df.label)

    listKeys = list(set(conv_id))
    listKeys.sort(key=conv_id.index)
    total_dict = dict.fromkeys(listKeys, "")

    if mode == "train" or mode == "valid":
        conv_label = dict(zip(conv_id, label))

        for idx, id in enumerate(conv_id):
            if "utterance" == conversion:
                total_dict[id] += string_process(utterance[idx], used_list)
                total_dict[id] += " "
            if "prompt" == conversion:
                total_dict[id] += string_process(prompt[idx], used_list)
                total_dict[id] += " "
            if "utterance+prompt" == conversion:
                total_dict[id] += string_process(prompt[idx], used_list)
                total_dict[id] += " "
                total_dict[id] += string_process(utterance[idx], used_list)
                total_dict[id] += " "

        for id in list(total_dict.keys()):
            total_data.append(total_dict[id])
            total_label.append(conv_label[id])

        new_df = pd.DataFrame(columns=["data", "label"])
        new_df["data"] = total_data
        new_df["label"] = total_label

    elif mode == "test":
        uniconv_id, count = np.unique(np.array(conv_id), return_counts=True)
        conv_count = dict(zip(uniconv_id, count))
        for idx, id in enumerate(conv_id):
            if "utterance" == conversion:
                total_dict[id] += string_process(utterance[idx], used_list)
                total_dict[id] += " "
            if "prompt" == conversion:
                total_dict[id] += string_process(prompt[idx], used_list)
                total_dict[id] += " "
            if "utterance+prompt" == conversion:
                total_dict[id] += string_process(prompt[idx], used_list)
                total_dict[id] += " "
                total_dict[id] += string_process(utterance[idx], used_list)
                total_dict[id] += " "
        for id in list(total_dict.keys()):
            total_data.append(total_dict[id])
            total_count.append(conv_count[id])

        new_df = pd.DataFrame(columns=["data", "count"])
        new_df["data"] = total_data
        new_df["count"] = total_count

    if mode == "train":
        new_df.to_csv(dirname+"/fixed_group_train.csv", index=False)
        print(dirname+"/fixed_group_train.csv is processed.")

    elif mode == "valid":
        new_df.to_csv(dirname+"/fixed_group_valid.csv", index=False)
        print(dirname+"/fixed_group_valid.csv is processed.")

    elif mode == "test":
        new_df.to_csv(dirname+"/fixed_group_test.csv", index=False)
        print(dirname+"/fixed_group_test.csv is processed.")


if __name__ == "__main__":
    train_df = pd.DataFrame(pd.read_csv("fixed_train.csv"))
    valid_df = pd.DataFrame(pd.read_csv("fixed_valid.csv"))
    test_df = pd.DataFrame(pd.read_csv("fixed_test.csv"))
    conversion = ["utterance", "prompt", "utterance+prompt"]

    if os.path.exists("data") == False:
        os.mkdir("data")

    for conv in conversion:
        dirname = "+".join([
            "1", "2", "3", "4", "5", "6", "7", "8", "9"])+"("+conv+")"
        create_directory("data/"+dirname)
        group_by(train_df, "train", conv, [
            "1", "2", "3", "4", "5", "6", "7", "8", "9"], "data/"+dirname)
        group_by(valid_df, "valid", conv, [
            "1", "2", "3", "4", "5", "6", "7", "8", "9"], "data/"+dirname)
        group_by(test_df, "test", conv, [
            "1", "2", "3", "4", "5", "6", "7", "8", "9"], "data/"+dirname)

    combination1 = list(combinations("123456789", 8))

    for comb in combination1:
        comb = sorted(list(comb))
        dirname = "+".join(comb)+"(utterance+prompt)"
        dirname = "data/"+dirname
        create_directory(dirname)
        group_by(train_df, "train", "utterance+prompt", comb, dirname)
        group_by(valid_df, "valid", "utterance+prompt", comb, dirname)
        group_by(test_df, "test", "utterance+prompt", comb, dirname)

    for conv in conversion:
        dirname = "+".join(["3", "5"])+"("+conv+")"
        create_directory("data/"+dirname)
        group_by(train_df, "train", conv, ["3", "5"], "data/"+dirname)
        group_by(valid_df, "valid", conv, ["3", "5"], "data/"+dirname)
        group_by(test_df, "test", conv, ["3", "5"], "data/"+dirname)

    for conv in conversion:
        dirname = "+".join(["3"])+"("+conv+")"
        create_directory("data/"+dirname)
        group_by(train_df, "train", conv, ["3"], "data/"+dirname)
        group_by(valid_df, "valid", conv, ["3"], "data/"+dirname)
        group_by(test_df, "test", conv, ["3"], "data/"+dirname)

    for conv in conversion:
        dirname = "+".join(["5"])+"("+conv+")"
        create_directory("data/"+dirname)
        group_by(train_df, "train", conv, ["5"], "data/"+dirname)
        group_by(valid_df, "valid", conv, ["5"], "data/"+dirname)
        group_by(test_df, "test", conv, ["5"], "data/"+dirname)

    for conv in conversion:
        dirname = "+".join(["1", "2", "3", "4", "5", "6"])+"("+conv+")"
        create_directory("data/"+dirname)
        group_by(train_df, "train", conv, [
                 "1", "2", "3", "4", "5", "6"], "data/"+dirname)
        group_by(valid_df, "valid", conv, [
                 "1", "2", "3", "4", "5", "6"], "data/"+dirname)
        group_by(test_df, "test", conv, [
                 "1", "2", "3", "4", "5", "6"], "data/"+dirname)

    for number in range(1, 11):
        dirname = "+".join([str(number)])+"(utterance+prompt)"
        dirname = "data/"+dirname
        create_directory(dirname)
        group_by(train_df, "train", "utterance+prompt",
                 [str(number)], dirname)
        group_by(valid_df, "valid", "utterance+prompt",
                 [str(number)], dirname)
        group_by(test_df, "test", "utterance+prompt",
                 [str(number)], dirname)
