import pandas as pd
import csv
import os, json
from time import sleep

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(THIS_DIR, 'data.csv')

df = pd.read_csv(data_path)

conv_path = os.path.join(THIS_DIR, 'conv_d.json')
with open(conv_path, 'r') as f:
    conv_d = json.load(f)

for column in df.columns:
    if column not in conv_d:
        df.drop(column, axis='columns', inplace=True)

empty_data = pd.DataFrame(columns=df.columns)          
# Traverse rows of dataframe and the elements, drop that row if there is an empty element
for index, row in df.iterrows():
    for i in range(len(row)):
        if pd.isnull(row[i]):
            empty_data.loc[len(empty_data)] = row
            df.drop(index, inplace=True)
            break
print(empty_data.shape)
empty_data.to_csv("empty_rows.csv") # CSV after task A

df_5 = pd.DataFrame(columns=df.columns)
for ind, row in df.iterrows():
    # F column - index 5
    try:
        var = int(row[5])
    except:
        df_5.loc[len(df_5.index)] = row
        df.drop(ind, inplace=True)     
     

df_5.to_csv("data_f_column_not_int.csv")
df.to_csv("data_after_A_B.csv")




# column 37 - food triggers
# Define the list of food classes
food_classes = range(1, 18)  # Assuming food classes range from 1 to 17

def is_triggered(row, food_class):
    print(row.split(','))
    return 1 if str(food_class) in row.split(',') else 0

# Iterate over each food class and create a new column with 1 if triggered, 0 otherwise
for food_class in food_classes:
    column_name = f'trigger_food_{food_class}'  # Name the new column
    df.insert(37+food_class, column_name, 0)

for index,row in df.iterrows():
    values = row[37].split(',')
    values = [int(i) for i in values]
    for value in values:
        if value == 0:
            continue
        df.at[index, f"trigger_food_{value}"] = str(1)

print(df["trigger_food_7"])
# Drop the original food triggers column if needed
df = df.drop('If some spesific foods trigger your headache please specify.', axis=1)

df.to_csv("deneme.csv")