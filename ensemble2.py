from operator import index
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
dirname = "ensemble2"
if os.path.exists(dirname) == False:
    os.mkdir(dirname)

number = []


for file in sys.argv:
    if file != "ensemble2.py":
        number.append(file)
        total_data.append(pd.read_csv(
            "experiment/"+file+"/submission2.csv").values)

total_data = np.array(total_data)
total_data = np.sum(total_data, axis=0)

final_data = [np.argmax(x) for x in total_data]

if os.path.exists(dirname+"/"+("+".join(number))) == True:
    shutil.rmtree(dirname+"/"+("+".join(number)))
os.mkdir(dirname+"/"+("+".join(number)))
df = pd.DataFrame(columns=["pred"])
df["pred"] = final_data
df.to_csv(dirname+"/"+("+".join(number))+"/submission.csv")
