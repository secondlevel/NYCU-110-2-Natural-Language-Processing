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

if os.path.exists("ensemble1") == False:
    os.mkdir("ensemble1")

number = []

for file in sys.argv:
    if file != "ensemble1.py":
        number.append(file)
        df = pd.DataFrame(pd.read_csv("experiment/"+file+"/submission1.csv"))
        total_data.append(np.array(df["pred"]))

total_data = np.array(total_data).T
final_data = [Counter(x).most_common(1)[0][0] for x in total_data]

df = pd.DataFrame(columns=["pred"])
df["pred"] = final_data
if os.path.exists("ensemble1/"+("+".join(number))) == True:
    shutil.rmtree("ensemble1/"+("+".join(number)))
os.mkdir("ensemble1/"+("+".join(number)))
df.to_csv("ensemble1/"+("+".join(number))+"/submission.csv")
