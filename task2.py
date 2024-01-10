import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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


# Load data
data = pd.read_csv("1-sepshock_MIMIC_allfeatures.csv")
data1 = pd.read_csv("sepshock_eicu_allfeatures.csv")
delete = list(set(data.columns) ^ set(data1.columns))
timestamp = data['suspected_infection_time']
# Wait, sep_shock_t will be deleted, the label has changed, it is not sep_shock_t
sofa_time = data['sep_shock_t']
list=[]

# Time conversion
for i in range(len(timestamp)):
    dt_object = datetime.strptime(timestamp[i], "%d/%m/%Y %H:%M:%S")
    supect_infection_time = int(dt_object.timestamp())
    t = (sofa_time[i]-supect_infection_time)/86400            # Difference days
    list.append(t)

data['difference'] = list                                     # Add difference column

# Remove rows with negative difference
data.loc[data['difference'] < 0, 'difference'] = 0
data = data[data['difference'] != 0]
data.loc[data['difference'] > 7, 'difference'] = 0            # Delete rows with a difference of 0
data = data[data['difference'] != 0]                          # Delete rows with a difference of 0
age = data['admission_age']                                   # Get age
gcs = data['gcs_min']
difference_Y = data['difference']
data = data.drop('difference', axis=1)

l=['age', 'gcs', 'patientunitstayid', 'H-L']
for i in l:
    delete.remove(i)

train_data = data.drop(delete, axis=1)
train_data = train_data.drop(['rate_dopamine', 'rate_dobutamine', 'rate_epinephrine', 'rate_norepinephrine', 'specimen'], axis=1)
train_data['age'] = age

# Assume that the column names to be exchanged are age and ehnincity respectively.
col1, col2 = 'age', 'ethnicity'
train_data[col1], train_data[col2] = train_data[col2].copy(), train_data[col1].copy()
train_data=train_data.rename(columns={'age': 'ethnicity', 'ethnicity': 'age'})
train_data['gcs'] = gcs

# Fill in missing values
w = train_data.columns.tolist()
p = ['gender', 'H-L', 'ethnicity', 'charlson_comorbidity_index', 'antibiotic', 'age', 'gcs']
train_data['gcs'] = train_data['gcs'].fillna(train_data['gcs'].mean())
for r in p:
    w.remove(r)
for t in w:
    # Fill with median value
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

# Divide training set and validation set
train_data_x = train_data[:8000]
train_data_y = difference_Y[:8000]

test_data_x = train_data[8000:9864]
test_data_y = difference_Y[8000:9864]

# Prepare data into PyTorch tensors
# Assume train_features is the feature tensor of the training data and train_labels is the label tensor of the timestamp
train_features = torch.tensor(train_data_x.values, dtype=torch.float32)
train_labels = torch.tensor(train_data_y.values, dtype=torch.float32)
test_data_x = torch.tensor(test_data_x.values, dtype=torch.float32)
# Convert val_data_y from pandas Series to NumPy array
test_data_y_np = test_data_y.to_numpy()

# Then convert the NumPy array to a PyTorch tensor
test_data_y = torch.tensor(test_data_y_np, dtype=torch.float32)
test_data_y = test_data_y.unsqueeze(1)


# When preparing training data, change the shape of train_labels to [batch_size, 1]
train_labels = train_labels.unsqueeze(1)

#Create training data set and data loader
train_dataset = TensorDataset(train_features, train_labels)
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = TimestampPredictionModel(input_size=84)
# Define loss function and optimizer
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
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
    test_outputs = model(test_data_x)
    test_loss = criterion(test_outputs, test_data_y)

    print("Validation Loss:", test_loss.item())

    # Truncate the predicted value to one decimal place
    rounded_predictions = torch.round(test_outputs * 10) / 10.0
    #test_data_y_handler = torch.round(test_data_y * 10) / 10.0
    rounded_predictions=rounded_predictions.numpy()
    print(rounded_predictions)
    #test_data_y_handler=test_data_y_handler.numpy()
    df = pd.DataFrame(rounded_predictions)
    df.columns = ['prediction']
    columns_to_output = ['prediction']
    selected_columns = df[columns_to_output]
    # Output the values of multiple columns to a file
    selected_columns.to_csv('output.csv', index=False, header=True)
    # with open('output.txt', 'w') as f:
    #     for value in df['prediction']:
    #         print(value)
    #         f.write(str(value) + '\n')#chu'chu
    # df= pd.DataFrame(rounded_predictions)
    # df.columns=['prediction']

    # tag
    # df.loc[(df['prediction'] > 0) & (df['prediction'] <= 0.3), 'prediction'] = 0
    # df.loc[(df['prediction'] > 0.3) & (df['prediction'] <= 0.6),'prediction'] = 1
    # df.loc[(df['prediction'] > 0.6) & (df['prediction'] < 1),'prediction'] = 2
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

    # Compare the labeled predicted values with the labels of the validation set
    # correct_predictions = (df['prediction'] == df['val_y'])
    # True_sum=correct_predictions.sum()

    # Calculate accuracy
    # accuracy =True_sum/ len(correct_predictions)
    # print("Accuracy:", accuracy)