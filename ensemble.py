from typing import Counter
import pandas as pd
import numpy as np
import sys
import os
from collections import Counter
import sys
import shutil
import os

total_data = []

# csvfile = list(os.listdir(os.path.dirname(os.path.realpath(__file__))))
if os.path.exists("ensemble") == False:
    os.mkdir("ensemble")

number = []

for file in sys.argv:
    if file != "ensemble.py":
        number.append(file)
        df = pd.DataFrame(pd.read_csv("experiment/"+file+"/submission.csv"))
        total_data.append(np.array(df["pred"]))

total_data = np.array(total_data).T
final_data = [Counter(x).most_common(1)[0][0] for x in total_data]

df = pd.DataFrame(columns=["pred"])
df["pred"] = final_data
if os.path.exists("ensemble/"+("+".join(number))) == True:
    shutil.rmtree("ensemble/"+("+".join(number)))
os.mkdir("ensemble/"+("+".join(number)))
df.to_csv("ensemble/"+("+".join(number))+"/submission.csv")
