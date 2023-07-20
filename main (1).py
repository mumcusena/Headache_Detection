import pandas as pd
import os, json
from time import sleep

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(THIS_DIR, 'raw_data.csv')
df = pd.read_csv(data_path)
conv_path = os.path.join(THIS_DIR, 'conv_d.json')

with open(conv_path, 'r') as f:
    conv_d = json.load(f)

data_d = {value : 0 for key, value in conv_d.items()}
empty_d = {}
for i in range(len(df)):
    for key, value in conv_d.items():
        if str(df[key][i]) == 'nan':
            data_d[value] += 1
            if df['no'][i] not in empty_d.keys():
                empty_d[df['no'][i]] = []
            empty_d[df['no'][i]].append(value)

with open('empty_d.txt', 'w') as f:
    f.write(str(empty_d))
# with open('empty_d.json', 'w') as f:
#     json.dump(empty_d, f, indent=4, ensure_ascii=False)

# print(data_d)
