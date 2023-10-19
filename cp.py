import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the dataset
patientdf = pd.read_csv('LungCancerPrediction\\cancer patient data sets.csv')

# Remove unnecessary columns
patientdf = patientdf.iloc[:, 2:]

# Remove columns with low correlation to the target
columns_to_remove = [
    'Fatigue', 'Weight Loss', 'Shortness of Breath', 'Wheezing',
    'Swallowing Difficulty', 'Clubbing of Finger Nails',
    'Frequent Cold', 'Dry Cough', 'Snoring'
]

patientdf.drop(columns=columns_to_remove, axis=1, inplace=True)

# Encode the 'Level' column to numeric values
le = LabelEncoder()
patientdf["Level"] = le.fit_transform(patientdf["Level"])

# Scale numeric features to the range [0, 1]
scaler = MinMaxScaler()
patientdf.iloc[:, :-1] = scaler.fit_transform(patientdf.iloc[:, :-1])
patientdf = patientdf.astype(int)

# Define predictor and target columns
col = list(patientdf.columns)
predictor = col[:-1]
target = col[-1]

# Split the dataset into training and testing sets
train, test = train_test_split(patientdf, test_size=0.2, random_state=0)

# Train a Logistic Regression model
model = LogisticRegression(random_state=0, n_jobs=-1)
model.fit(train[predictor], train[target])

# Evaluate the Logistic Regression model on test data
print("For Test: ")
test_pred = model.predict(test[predictor])
print("Accuracy: ", accuracy_score(test[target], test_pred))

print("..............................................................")

# Function to make a prediction
def predict_cancer(age, gender, symptoms):
    # Create a DataFrame with user input
    data = pd.DataFrame({
        'Age' : [age],
        'Gender': 0 if gender == "male" or gender == "Male" else 1,
        'Air Pollution' : [age],
        'Alcohol Use'  : [symptoms[0]],
        'Dust Allergy' : [symptoms[1]],
        'Occupational Hazards' : [symptoms[2]],
        'Genetic Risk' : [symptoms[3]],
        'Chronic Lung Disease' : [symptoms[4]],
        'Balanced Diet' : [symptoms[5]],
        'Obesity' : [symptoms[6]],
        'Smoking' : [symptoms[7]],
        'Passive Smoker' : [symptoms[8]],
        'Chest Pain' : [symptoms[9]],
        'Coughing of Blood' : [symptoms[10]],
    })

    # Scale numeric features to the range [0, 1]
    data.iloc[:, 2:-1] = scaler.transform(data.iloc[:, 2:-1])
    
    # Make a prediction
    prediction = model.predict(data)

    return "Cancer" if prediction[0] == 1 else "No Cancer"

# Get user input for prediction
age = float(input("Enter Age: "))
gender = input("Enter Gender (Male/Female): ")
symptoms = []
for symptom in patientdf.columns[2:-1]:
    symptom_value = input(f"Does the patient have {symptom}? (Y/N): ").strip().lower()
    symptoms.append(1 if symptom_value == 'y' else 0)

# Make a prediction
result = predict_cancer(age, gender, symptoms)
print(f"The prediction is: {result}")