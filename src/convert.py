import pandas as pd

# Column names from UCI dataset description
columns = [
    "age",  
    "sex",  # 1 = male, 0 = female
    "cp",   # chest_pain_type
    "restBP",  # resting_blood_pressure
    "chol",  # serum_cholesterol
    "fbs",  # fasting_blood_sugar
    "restECG",  # resting_ecg
    "max_HR",  # max_heart_rate
    "exang",  # exercise_induced_angina
    "oldpeak",  # st_depression_exercise
    "slope",  # slope_peak_exercise
    "ca",  # number_of_major_vessels
    "thal",  # thalassemia
    "target"  # heart_disease (1=yes, 0=no)
]

# Load the data file
cleveland = pd.read_csv("data/raw/cleveland.data", names=columns)
hungarian = pd.read_csv("data/raw/hungarian.data", names=columns)
switzerland = pd.read_csv("data/raw/switzerland.data", names=columns)

#merge all into one
all_data=pd.concat([cleveland,hungarian,switzerland],ignore_index=True)

print("shape of merged data", all_data.shape)
all_data.to_csv("data/interim/heart_disease_combined.csv", index=False)
