import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score
from imblearn.over_sampling import SMOTE
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Load the dataset
data = pd.read_csv('medical_blood_test_dataset.csv')

# Drop irrelevant columns
data = data.drop(['Patient_ID', 'Name', 'Region'], axis=1)

# Define numerical and categorical columns based on the dataset
numerical_cols = ['Age', 'BMI', 'Hemoglobin', 'WBC', 'RBC', 'Platelets', 'Glucose_Fasting', 'HbA1c', 
                  'Cholesterol_Total', 'Triglycerides', 'LDL', 'HDL', 'ALT', 'AST', 'Bilirubin_Total', 
                  'Creatinine', 'Urea', 'TSH', 'T3', 'T4', 'CRP']

# Replace negative values with NaN for biologically implausible values
for col in numerical_cols:
    data.loc[data[col] < 0, col] = np.nan

# Handle missing values for numerical columns
num_imputer = SimpleImputer(strategy='median')
data[numerical_cols] = num_imputer.fit_transform(data[numerical_cols])

# Define categorical columns
categorical_cols = ['Gender', 'Smoker', 'Alcohol_Intake', 'Exercise_Freq', 'COVID_Antigen_Pos', 
                    'HIV_Positive', 'Hepatitis_B_Pos', 'Hepatitis_C_Pos', 'Malaria_Positive', 'Tuberculosis_Pos']

# Handle missing values for categorical columns
cat_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])

# Encode categorical variables
# Handle 'Gender' with 'Other' by mapping to a numeric value (e.g., 2 for 'Other')
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1, 'Other': 2})
data['Smoker'] = data['Smoker'].map({'No': 0, 'Yes': 1})
data['Alcohol_Intake'] = data['Alcohol_Intake'].map({'None': 0, 'Low': 1, 'Moderate': 2, 'High': 3})

# Map binary categorical columns
binary_cols = ['COVID_Antigen_Pos', 'HIV_Positive', 'Hepatitis_B_Pos', 'Hepatitis_C_Pos', 'Malaria_Positive', 'Tuberculosis_Pos']
for col in binary_cols:
    data[col] = data[col].map({'No': 0, 'Yes': 1})

# One-hot encode Exercise_Freq
data = pd.get_dummies(data, columns=['Exercise_Freq'], prefix='Exercise')

# Define features and target
X = data.drop('Disease', axis=1)
y = data['Disease']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numerical_cols])
X_train_scaled = pd.DataFrame(X_train_scaled, columns=numerical_cols, index=X_train.index)
X_train_scaled = pd.concat([X_train_scaled, X_train.drop(columns=numerical_cols)], axis=1)

X_test_scaled = scaler.transform(X_test[numerical_cols])
X_test_scaled = pd.DataFrame(X_test_scaled, columns=numerical_cols, index=X_test.index)
X_test_scaled = pd.concat([X_test_scaled, X_test.drop(columns=numerical_cols)], axis=1)

# Adjusted SMOTE: Partial oversampling to avoid overfitting
majority_class_count = y_train.value_counts().max()
sampling_strategy = {}
for cls, count in y_train.value_counts().items():
    if count >= majority_class_count // 2:
        # Keep classes with counts >= majority_class_count // 2 unchanged
        sampling_strategy[cls] = count
    else:
        # Oversample minority classes to at least 100 or up to majority_class_count // 2
        target = max(100, min(count * 2, majority_class_count // 2))
        sampling_strategy[cls] = target

# Print class distribution and sampling strategy for debugging
print("Class distribution in y_train:\n", y_train.value_counts())
print("\nSampling strategy:\n", sampling_strategy)

# Adjust k_neighbors for SMOTE based on the smallest class size
min_class_size = y_train.value_counts().min()
k_neighbors = max(1, min_class_size - 1)  # Ensure at least 1 neighbor, but not more than available
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=k_neighbors)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# Train RandomForestClassifier with balanced class weights
model = RandomForestClassifier(random_state=42, class_weight='balanced', max_depth=10)
model.fit(X_train_smote, y_train_smote)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation Metrics
print("\nWeighted F1-score:", f1_score(y_test, y_pred, average='weighted'))
print("Weighted Recall:", recall_score(y_test, y_pred, average='weighted'))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)
print("\nFeature Importance (Top 10):\n", feature_importance.head(10))

# Visualize Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('Top 10 Most Important Features')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Cross-Validation
cv_scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring='f1_weighted')
print("\nCross-validation F1-scores:", cv_scores)
print("Mean CV F1-score:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())

# Adjust Prediction Thresholds for Minority Classes
probabilities = model.predict_proba(X_test_scaled)
minority_classes = ['HIV/AIDS', 'Hepatitis C', 'Tuberculosis', 'Malaria', 'Hepatitis B']
minority_indices = [list(model.classes_).index(cls) for cls in minority_classes if cls in model.classes_]
threshold = 0.3  # Lower threshold for minority classes
y_pred_adjusted = np.argmax(probabilities, axis=1)
for i, prob in enumerate(probabilities):
    for idx in minority_indices:
        if prob[idx] > threshold:
            y_pred_adjusted[i] = idx
            break
y_pred_adjusted = [model.classes_[idx] for idx in y_pred_adjusted]
print("\nWeighted F1-score (with adjusted thresholds):", f1_score(y_test, y_pred_adjusted, average='weighted'))
print("Weighted Recall (with adjusted thresholds):", recall_score(y_test, y_pred_adjusted, average='weighted'))

# Test on new data (example input)
# Ensure Exercise_Freq matches the original dataset's categories
new_data = pd.DataFrame({
    'Age': [45], 'Gender': [0], 'BMI': [27.5], 'Hemoglobin': [14.2], 'WBC': [6.5], 'RBC': [4.8], 
    'Platelets': [250], 'Glucose_Fasting': [95], 'HbA1c': [5.5], 'Cholesterol_Total': [190], 
    'Triglycerides': [130], 'LDL': [110], 'HDL': [50], 'ALT': [25], 'AST': [20], 
    'Bilirubin_Total': [0.8], 'Creatinine': [0.9], 'Urea': [20], 'TSH': [2.5], 
    'T3': [1.2], 'T4': [8.0], 'CRP': [2.0], 'Smoker': [0], 'Alcohol_Intake': [1], 
    'COVID_Antigen_Pos': [0], 'HIV_Positive': [0], 'Hepatitis_B_Pos': [0], 'Hepatitis_C_Pos': [0], 
    'Malaria_Positive': [0], 'Tuberculosis_Pos': [0], 'Exercise_Freq': ['Never']  # Use raw category
})

# One-hot encode Exercise_Freq in new_data consistently with training
new_data = pd.get_dummies(new_data, columns=['Exercise_Freq'], prefix='Exercise')

# Align new_data columns with X_train_scaled
missing_cols = set(X_train_scaled.columns) - set(new_data.columns)
for col in missing_cols:
    new_data[col] = 0
new_data = new_data[X_train_scaled.columns]  # Reorder to match X_train_scaled

# Scale numerical features
new_data_scaled = scaler.transform(new_data[numerical_cols])
new_data_scaled = pd.DataFrame(new_data_scaled, columns=numerical_cols, index=new_data.index)
new_data_scaled = pd.concat([new_data_scaled, new_data.drop(columns=numerical_cols)], axis=1)

# Predict
new_pred = model.predict(new_data_scaled)
new_pred_proba = model.predict_proba(new_data_scaled)
print("\nPrediction for new_data:", new_pred[0])
print("Prediction probabilities:", dict(zip(model.classes_, new_pred_proba[0])))

# Update predict_disease function similarly
def predict_disease(new_data):
    # One-hot encode Exercise_Freq
    new_data = pd.get_dummies(new_data, columns=['Exercise_Freq'], prefix='Exercise')
    # Align columns
    missing_cols = set(X_train_scaled.columns) - set(new_data.columns)
    for col in missing_cols:
        new_data[col] = 0
    new_data = new_data[X_train_scaled.columns]
    # Scale numerical features
    new_data_scaled = scaler.transform(new_data[numerical_cols])
    new_data_scaled = pd.DataFrame(new_data_scaled, columns=numerical_cols, index=new_data.index)
    new_data_scaled = pd.concat([new_data_scaled, new_data.drop(columns=numerical_cols)], axis=1)
    # Predict
    prediction = model.predict(new_data_scaled)[0]
    probabilities = model.predict_proba(new_data_scaled)[0]
    prob_dict = dict(zip(model.classes_, probabilities))
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    print(f"\nPredicted Disease: {prediction}")
    print(f"Confidence: {prob_dict[prediction]*100:.2f}%")
    print("Other possibilities:")
    for disease, prob in sorted_probs[1:5]:
        if prob > 0.005:  # Show only if probability > 0.5%
            print(f"{disease}: {prob*100:.2f}%")

# Test the function
predict_disease(pd.DataFrame({
    'Age': [45], 'Gender': [0], 'BMI': [27.5], 'Hemoglobin': [14.2], 'WBC': [6.5], 'RBC': [4.8], 
    'Platelets': [250], 'Glucose_Fasting': [95], 'HbA1c': [5.5], 'Cholesterol_Total': [190], 
    'Triglycerides': [130], 'LDL': [110], 'HDL': [50], 'ALT': [25], 'AST': [20], 
    'Bilirubin_Total': [0.8], 'Creatinine': [0.9], 'Urea': [20], 'TSH': [2.5], 
    'T3': [1.2], 'T4': [8.0], 'CRP': [2.0], 'Smoker': [0], 'Alcohol_Intake': [1], 
    'COVID_Antigen_Pos': [0], 'HIV_Positive': [0], 'Hepatitis_B_Pos': [0], 'Hepatitis_C_Pos': [0], 
    'Malaria_Positive': [0], 'Tuberculosis_Pos': [0], 'Exercise_Freq': ['Never']
}))
