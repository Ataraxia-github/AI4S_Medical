## Project Introduction
This project aims to use machine learning methods to analyze electronic medical health record data. Specifically, we are committed to using characteristic data within 24 hours after patients are admitted to the ICU to determine whether they may suffer from septic shock and predict the risk of septic shock. onset time and patient survival time. Therefore, this project contains three subtasks, requiring classification prediction and regression prediction. We hope that through this work, we will provide strong support for artificial intelligence methods to improve medical decision-making with the help of electronic medical data, and provide better medical care and treatment for patients.

Subtask 1: Classify septic shock patients and determine whether the patient is a septic shock patient based on the input patient characteristics;

Subtask 2: Predict the onset time of septic shock patients. Enter the data of the patient 24 hours after admission to the ICU to predict the onset time of septic shock;

Subtask 3: Predict the survival time of patients with septic shock. Enter the data of patients with septic shock to predict the patient’s survival time.

## Project Highlights
In task one, we used a strict feature screening method to select 84 types of feature data that are closely related to the output results, and adopted a diversity strategy in model selection. We were not limited to a single model, but It combines a variety of classic machine learning algorithms for comprehensive and in-depth consideration.

In task two, considering the complexity of the data and the uncertainty of the labeled samples, we adopted a method of calculating the difference as the basis for label generation. Specifically, we subtracted the known onset time of septic shock from the time when the patient was suspected of infection to obtain a time difference, and used this time difference as a label for model training and prediction output. This difference not only improves the interpretability of the results, allowing readers to intuitively understand how long a patient may have become ill after a suspected infection, but also enhances the consistency of the data. By replacing the original long character timestamp format data with time difference values, we further improve the reusability and optimizability of medical data.

In task three, we combine the methods of task one and task two and directly use existing labels as positive and negative samples for model training. At the same time, we maintained the overall stability of the data cleaning process and model, highlighting the model's versatility and resource utilization efficiency. This comprehensive strategy aims to make full use of existing resources, ensure the shareability of models across different tasks, and provide strong support for further research and applications.

## Dependencies
```python
python = 3.8	# python edition
```

```python
>>> import torch	# pytorch edition
>>> print(torch.__version__) 
1.13.1
```

Necessary libraries：``` pandas、sklearn、numpy```

## Directory Structure
├ ─ ─ README.md           // Chinese help document

├ ─ ─ README_en.md           // English help document

├ ─ ─ EICU_data
      |— — 1-sepshock_MIMIC_allfeatures
      |— — sep_shock_admtime

├ ─ ─ EICU_project    			
|— — task1.py
|— — task2.py
|— — task3.py
|— — model.py
|— — util_data.py


## Usage
This project is the winning entry of the 2023 "Greater Bay Area Cup" Guangdong, Hong Kong and Macao AI for Science technology competition, and has applied for China's national software copyright and related authorization protection. We embrace open source but refuse direct theft. The core code has been slightly modified and deleted. You can add and expand the process of data preprocessing and model training by yourself. The ideas of this project are for reference only!
