import pandas as pd
import numpy as np
import os
import random
import re

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
file_name = 'deneme'
data_path = os.path.join(THIS_DIR, file_name + ".csv")

df = pd.read_csv(data_path)

index_zeros = df[(df['1. Migraine'] == 0) & (df['2. Tension-type headache (TTH)'] == 0) & (df['3. Trigeminal autonomic cephalalgias (TACs)'] == 0) & (df['4. Other primary headache disorders'] == 0)].index

dropped_df = df[df.index.isin(index_zeros)]

df.drop(index_zeros , inplace=True)

dropped_df.to_csv(file_name + "_dropped_rows.csv", index=False)
df.to_csv(file_name + "_nonzero.csv", index=False)