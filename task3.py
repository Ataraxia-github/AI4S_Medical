import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_columns', 10000)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', 10000)
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor, Ridge, RidgeCV
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime, time
from model import TimestampPredictionModel


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load data
data = pd.read_csv("1-sepshock_MIMIC_allfeatures.csv")
# Screen out infectious patients H-L=1
data = data[data['H-L'] != 0]
data1 = pd.read_csv("sepshock_eicu_allfeatures.csv")
delete = list(set(data.columns) ^ set(data1.columns))
s_time = data['sur_time']
age = data['admission_age']
gcs = data['gcs_min']

l = ['age', 'gcs', 'patientunitstayid', 'H-L']
for i in l:
    delete.remove(i)

train_data = data.drop(delete, axis=1)
train_data = train_data.drop(['rate_dopamine', 'rate_dobutamine', 'rate_epinephrine', 'rate_norepinephrine', 'specimen'], axis=1)
train_data['age'] = age
col1, col2 = 'age', 'ethnicity'
train_data[col1], train_data[col2] = train_data[col2].copy(), train_data[col1].copy()
train_data = train_data.rename(columns={'age': 'ethnicity', 'ethnicity': 'age'})
train_data['gcs'] = gcs

# 填充缺失值
w = train_data.columns.tolist()
p = ['gender', 'H-L', 'ethnicity', 'charlson_comorbidity_index', 'antibiotic', 'age', 'gcs']
train_data['gcs'] = train_data['gcs'].fillna(train_data['gcs'].mean())
for r in p:
    w.remove(r)
for t in w:
    train_data[t] = train_data[t].fillna(data[t].median())

dic = {'F': 0, "M": 1}
dic1 = {'BLACK/AFRICAN AMERICAN': 0, 'UNKNOWN': 1, 'WHITE': 2, 'UNABLE TO OBTAIN': 3, 'OTHER': 4,
 'ASIAN': 5, 'HISPANIC/LATINO': 6, 'AMERICAN INDIAN/ALASKA NATIVE': 7}
dic2 = {}
op = train_data['antibiotic'].unique()
i = 0

for u in op:
    dic2.update({u: i})
    i += 1

train_data['antibiotic'] = train_data['antibiotic'].replace(dic2)
train_data['gender'] = train_data['gender'].replace(dic)
train_data['ethnicity'] = train_data['ethnicity'].replace(dic1)
train_data_x = train_data
train_data_y = s_time
train_data_x['ethnicity']=train_data_x['ethnicity'].astype(float)
print(train_data_x.info())


test1_data = pd.read_csv('sepshock_label.csv')
new_test_data = pd.merge(data1, test1_data, on="patientunitstayid", how='right')
test_data = new_test_data.drop(['rate_dopamine', 'rate_dobutamine', 'rate_epinephrine', 'rate_norepinephrine',
                                'patientunitstayid', 'unitdischargeoffset'], axis=1)

colum_name = test_data.columns.tolist()
l = ['gender', 'age', 'charlson_comorbidity_index']
for u in l:
    colum_name.remove(u)
test_data.loc[test_data['ethnicity'] == 'Other/Unknown', 'ethnicity'] = 1
test_data = test_data.drop('specimen', axis=1)
dic = {'Female': 0, "Male": 1}
dic1 = {'African American': 0, 'Caucasian': 2,
 'Asian': 5, 'Hispanic': 6, 'Native American': 7}
dic2 = {}
op = test_data['antibiotic'].unique()
i = 0
for u in op:
    dic2.update({u: i})
    i += 1
test_data['antibiotic'] = test_data['antibiotic'].replace(dic2)
test_data['gender'] = test_data['gender'].replace(dic)
test_data['ethnicity'] = test_data['ethnicity'].replace(dic1)

w = test_data.columns.tolist()
p = ['gender', 'age', 'charlson_comorbidity_index', 'gcs']
test_data['gcs'] = test_data['gcs'].fillna(test_data['gcs'].mean())
test_data.loc[test_data['age'] == '> 89', 'age'] = 90
for r in p:
    w.remove(r)
for t in w:
    test_data[t] = test_data[t].fillna(test_data[t].median())

yu = test_data['ethnicity']
test_data = test_data.drop('ethnicity', axis=1)
test_data['ethnicity'] = yu
yu1 = test_data['gcs']
test_data = test_data.drop('gcs', axis=1)
test_data['gcs'] = yu1
test_data['age'] = test_data['age'].astype(float)
print(test_data.info())

train_features = torch.tensor(train_data_x.values, dtype=torch.float32)
train_labels = torch.tensor(train_data_y.values, dtype=torch.float32)
test_features = torch.tensor(test_data.values, dtype=torch.float32)

# test_data_y = torch.tensor(test_data_y_np, dtype=torch.float32)
# test_data_y = test_data_y.unsqueeze(1)

train_labels = train_labels.unsqueeze(1)
train_dataset = TensorDataset(train_features, train_labels)
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = TimestampPredictionModel(input_size=84)
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    total_loss = 0.0

    for batch_features, batch_labels in train_loader:

        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')


with torch.no_grad():
    test_outputs = model(test_features)

    rounded_sur_time = torch.round(test_outputs * 10) / 10.0
    #test_data_y_handler = torch.round(test_data_y * 10) / 10.0
    rounded_sur_time = rounded_sur_time.numpy()
    #test_data_y_handler=test_data_y_handler.numpy()
    df = pd.DataFrame(rounded_sur_time)
    df.columns = ['sur_time']
    columns_to_output = ['sur_time']
    selected_columns = df[columns_to_output]
    selected_columns.to_csv('output_sur_time.csv', index=False, header=True)
    # with open('output.txt', 'w') as f:
    #     for value in df['prediction']:
    #         print(value)
    #         f.write(str(value) + '\n')#chu'chu
    # df= pd.DataFrame(rounded_predictions)
    # df.columns=['prediction']
    # df.loc[(df['prediction'] > 0) & (df['prediction'] <= 0.3), 'prediction'] = 0
    # df.loc[(df['prediction'] > 0.3) & (df['prediction'] <= 0.6), 'prediction'] = 1
    # df.loc[(df['prediction'] > 0.6) & (df['prediction'] < 1), 'prediction'] = 2
    # df.loc[(df['prediction'] > 1) & (df['prediction'] < 2), 'prediction'] = 3
    # df.loc[(df['prediction'] > 2) & (df['prediction'] < 3), 'prediction'] = 4
    # df.loc[(df['prediction'] > 3) & (df['prediction'] < 4), 'prediction'] = 5
    # df.loc[(df['prediction'] > 4) & (df['prediction'] < 5), 'prediction'] = 6
    # df.loc[(df['prediction'] > 5) & (df['prediction'] < 6), 'prediction'] = 7
    # df.loc[(df['prediction'] > 6) & (df['prediction'] < 7), 'prediction'] = 8
    # df.loc[(df['val_y'] > 0) & (df['val_y'] <= 0.3), 'val_y'] = 0
    # df.loc[(df['val_y'] > 0.3) & (df['val_y'] <= 0.6), 'val_y'] = 1
    # df.loc[(df['val_y'] > 0.6) & (df['val_y'] < 1), 'val_y'] = 2
    # df.loc[(df['val_y'] > 1) & (df['val_y'] < 2), 'val_y'] = 3
    # df.loc[(df['val_y'] > 2) & (df['val_y'] < 3), 'val_y'] = 4
    # df.loc[(df['val_y'] > 3) & (df['val_y'] < 4), 'val_y'] = 5
    # df.loc[(df['val_y'] > 4) & (df['val_y'] < 5), 'val_y'] = 6
    # df.loc[(df['val_y'] > 5) & (df['val_y'] < 6), 'val_y'] = 7
    # df.loc[(df['val_y'] > 6) & (df['val_y'] < 7), 'val_y'] = 8

    # correct_predictions = (df['prediction'] == df['val_y'])
    # True_sum = correct_predictions.sum()
    #
    # accuracy = True_sum/ len(correct_predictions)
    # print("Accuracy:", accuracy)

