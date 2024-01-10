'''

    You may need to use this file separately to process data

'''

import pandas as pd
pd.set_option('display.max_columns', 10000)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', 10000)


def Datautils(data_path):
    # Read data file
    data = pd.read_csv(data_path)
    # Organize training data and filter useless features
    # Fill in the missing values of the feature with the median value
    train_data = data.drop(['subject_id', 'hadm_id', 'stay_id', 'sofa_time', 'sur_time', 'los_hospital', 'los_icu', 'dead', 'H-L'])
    w = train_data.columns.tolist()
    for i in w:
        train_data[i] = train_data[i].fillna(train_data[i].median())

    # Map processing text values and replace them with numerical values
    dic1 = {'F': 0, "M": 1}
    dic2 = {}
    dic3 = {}
    op2 = train_data['ethnicity'].unique()
    op3 = train_data['antibiotic'].unique()
    i = 0

    # Process dictionaries separately
    for o in op2:
        dic2.update({o: i})
        i += 1

    # Reset i to 0 to re-count when processing op2
    i = 0
    for o in op3:
        dic3.update({o: i})
        i += 1

    train_data['gender']=train_data['gender'].replace(dic1)
    train_data['antibiotic']=train_data['antibiotic'].replace(dic2)
    train_data['ethnicity']=train_data['ethnicity'].replace(dic3)

    train_data_x = train_data[0:10000].drop('sep_shock_t', axis=1)
    val_data_x = train_data[10000:11948].drop('sep_shock_t', axis=1)
    train_data_y = train_data[0:10000]['sep_shock_t']
    val_data_y = train_data[10000:11948]['sep_shock_t']

    train_data_x['suspected_infection_time'] = train_data_x['suspected_infection_time'].str.replace(r'/\d{4}\s', ' ')


    '''
    
    # Process tags (timestamps)
    
    
    '''

    return train_data_x, train_data_y, val_data_x, val_data_y
