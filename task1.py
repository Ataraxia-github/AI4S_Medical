import torch
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve, roc_auc_score, make_scorer, f1_score, auc, precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

pd.set_option('display.max_columns', 10000)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', 10000)

warnings.filterwarnings("ignore")

# Load data
data = pd.read_csv("./MIMIC-IV_train/1-sepshock_MIMIC_allfeatures.csv")
data1 = pd.read_csv("./EICU_test/sepshock_eicu_allfeatures.csv")
age = data['admission_age']                       # attach age
gcs = data['gcs_min']                             # attach gcs value

delete = list(set(data.columns) ^ set(data1.columns))
l = ['age', 'gcs', 'patientunitstayid', 'H-L']

for i in l:
    delete.remove(i)

# Handle data
train_data = data.drop(delete, axis=1)
train_data = train_data.drop(['rate_dopamine', 'rate_dobutamine', 'rate_epinephrine', 'rate_norepinephrine', 'specimen'], axis=1)
train_data['age'] = age
col1, col2 = 'age', 'ethnicity'
train_data[col1], train_data[col2] = train_data[col2].copy(), train_data[col1].copy()
train_data = train_data.rename(columns={'age': 'ethnicity', 'ethnicity': 'age'})    # Edit column names
train_data['gcs'] = gcs

# Fill missing values
w = train_data.columns.tolist()
p = ['gender', 'H-L', 'ethnicity', 'charlson_comorbidity_index', 'antibiotic', 'age', 'gcs']
# Use average
train_data['gcs'] = train_data['gcs'].fillna(train_data['gcs'].mean())

for r in p:
    w.remove(r)
for t in w:
    # Use median value to fill
    train_data[t] = train_data[t].fillna(data[t].median())

dic = {'F': 0, "M": 1}
dic1 = {'BLACK/AFRICAN AMERICAN': 0, 'UNKNOWN': 1, 'WHITE': 2, 'UNABLE TO OBTAIN': 3, 'OTHER': 4, 'ASIAN': 5, 'HISPANIC/LATINO': 6, 'AMERICAN INDIAN/ALASKA NATIVE': 7}
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
train_data_x = train_data[:10000]
train_data_y = train_data_x['H-L']
train_data_x = train_data_x.drop('H-L', axis=1)

val_data_x = train_data[10000:11948]
val_data_y = val_data_x['H-L']
val_data_x = val_data_x.drop('H-L', axis=1)

# Divide training batches
train_data_x = torch.tensor(train_data_x.values, dtype=torch.float32)
train_data_y = torch.tensor(train_data_y.values, dtype=torch.long)
train_data_new = TensorDataset(train_data_x.clone().detach(), train_data_y.clone().detach())
train_loader = DataLoader(dataset=train_data_new, batch_size=500, shuffle=True)
# print(train_loader)

# Test set processing
test1_data = pd.read_csv('./EICU_test/sepshock_label.csv')
new_test_data = pd.merge(data1, test1_data, on="patientunitstayid", how='right')
test_Y_data = new_test_data['H-L']
test_data = new_test_data.drop(['rate_dopamine', 'rate_dobutamine', 'rate_epinephrine', 'rate_norepinephrine', 'patientunitstayid', 'unitdischargeoffset', 'H-L'], axis=1)

# Missing value handling
colum_name = test_data.columns.tolist()
l = ['gender', 'age', 'charlson_comorbidity_index']

for u in l:
    colum_name.remove(u)

test_data.loc[test_data['ethnicity'] == 'Other/Unknown', 'ethnicity'] = 1
test_data = test_data.drop('specimen', axis=1)
dic = {'Female': 0, "Male": 1}
dic1 = {'African American': 0, 'Caucasian': 2, 'Asian': 5, 'Hispanic': 6, 'Native American': 7}
dic2 = {}

op = test_data['antibiotic'].unique()
i = 0
for u in op:
    dic2.update({u: i})
    i += 1

test_data['antibiotic'] = test_data['antibiotic'].replace(dic2)
test_data['gender'] = test_data['gender'].replace(dic)
test_data['ethnicity'] = test_data['ethnicity'].replace(dic1)

# Missing value filling
w = test_data.columns.tolist()
p = ['gender', 'age', 'charlson_comorbidity_index', 'gcs']
# average value of gcs to fill
test_data['gcs'] = test_data['gcs'].fillna(test_data['gcs'].mean())
# Those over 89 years old will be treated as 90 years old
test_data.loc[test_data['age'] == '> 89', 'age'] = 90

for r in p:
    w.remove(r)
for t in w:
    test_data[t] = test_data[t].fillna(test_data[t].median())

#  Make one-to-one correspondence between the feature names of the test set and the training set.
yu = test_data['ethnicity']
test_data = test_data.drop('ethnicity', axis=1)
test_data['ethnicity'] = yu
yu1 = test_data['gcs']
test_data = test_data.drop('gcs', axis=1)
test_data['gcs'] = yu1

alg1 = DecisionTreeClassifier()                     # Use decision trees
alg2 = SVC(probability=True, random_state=15)       # Support vector machine
alg3 = RandomForestClassifier()                     # Random forest
alg4 = KNeighborsClassifier(n_jobs=-1)              # KNN

parameters1 = {'max_depth': range(1, 10), 'min_samples_split': range(2, 10), 'random_state': [23]}  # DecisionTreeClassifier
parameters2 = {"C": range(1, 20), "gamma": [0.05, 0.1, 0.15, 0.2, 0.25]}                            # SVC
parameters3 = {'n_estimators': range(10, 200, 10)}                                                  # RandomForestClassifie
parameters4 = {'n_neighbors': range(2, 10), 'leaf_size': range(10, 80, 20)}                         # KNeighborsClassifier


# Initialize an empty array to hold the true labels and predicted probabilities for all batches
all_labels = []
all_probs = []

# Train
for batch_size, (features, labels) in enumerate(train_loader):
    scorer = make_scorer(roc_auc_score)  # 评分标准
    grid = GridSearchCV(estimator=alg1, param_grid=parameters1, scoring=scorer, cv=None)  # 找到最好的参数
    grid.fit(features, labels)
    a = grid.score(features, labels)

# Predict
# Assuming it is a binary classification task, the probability of extracting the positive category
test_probs = grid.predict_proba(test_data)[:, 1]
predicted_labels = grid.predict(test_data)

# Calculate the true class rate and false positive class rate of the test set
# Calculate precision and recall
# Calculate the overall ROC curve and AUROC value
# Calculate AUPRC value
# Calculate precision, recall and F1 Score under different thresholds
fpr, tpr, thresholds = roc_curve(test_Y_data, test_probs)
precision, recall, _ = precision_recall_curve(test_Y_data, test_probs)
roc_auc = auc(fpr, tpr)
auprc = auc(recall, precision)
f1 = f1_score(test_Y_data, predicted_labels)

# Calculate accuracy
# Assume that 0.5 is used as the threshold for binary classification
y_pred = (test_probs > 0.5).astype(int)
accuracy = accuracy_score(test_Y_data, y_pred)

print("Train Accuracy：", a)
print("Val Accuracy：", grid.score(val_data_x, val_data_y))
print("Test Accuracy：", grid.score(test_data, test_Y_data))
print("F1 Score:", f1)

# Draw AUROC curve
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()

# Draw AUPRC curve
# plt.figure()
# plt.plot(recall, precision, color='darkorange', lw=2, label='AUPRC curve (area = %0.2f)' % auprc)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend(loc="lower left")
# plt.show()

# Draw performance evaluation graph
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2, label='AUPRC curve (AUC = %0.2f)' % auprc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Performance Evaluation')
plt.legend(loc="lower left")

# Add accuracy curve
plt.twinx()
plt.plot([0, 1], [accuracy, accuracy], color='green', lw=2, label='Accuracy = %0.2f' % accuracy)
plt.ylim([0.0, 1.05])
plt.ylabel('Accuracy')

plt.legend(loc="lower right")
plt.show()
