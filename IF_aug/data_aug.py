import pandas as pd
import numpy as np
import os
import random
import re

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(THIS_DIR, 'dataS_10_IF_aug_aug.csv')
dataS_00_IF = pd.read_csv(data_path)

#dataS_10_IF.drop(columns=dataS_10_IF.columns[0], axis=1, inplace=True)


df = dataS_00_IF.loc[(dataS_00_IF['1. Migraine'] == 0) & 
                        (dataS_00_IF['2. Tension-type headache (TTH)'] == 0) & 
                        (dataS_00_IF['3. Trigeminal autonomic cephalalgias (TACs)'] == 0) & 
                        (dataS_00_IF['4. Other primary headache disorders'] == 1)]


np.random.seed(1) # seed value for reproducability
for i in range(500):
    
    random_rows = df.sample(n=5)
    random_row = random_rows.iloc[np.random.randint(0, len(random_rows))]
    
    new_row = pd.DataFrame(columns=dataS_00_IF.columns)
    
    pattern = '^trigger_food_\d+$'
    for column in dataS_00_IF.columns:
        if re.match(pattern, column):
            new_row[column] = random_row[column]
        elif column == 'no':
            pass
        elif column in ['Age' , 'Headache frequency of last month', 'Mean severity of the headache']:
            new_row[column] = [random_rows[column].mean()]
        else:
            new_row[column] = random_row[column]            

    dataS_00_IF = pd.concat([dataS_00_IF, new_row], ignore_index=True)

os.chdir(THIS_DIR)
dataS_00_IF.to_csv('dataS_10_IF_aug_aug.csv', index=False)

