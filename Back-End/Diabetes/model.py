import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import BorderlineSMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# Section 1: Loading and Cleaning Data
# Loading the diabetes dataset
df = pd.read_csv('diabetes.csv')

# Removing less relevant columns
columns_to_drop = ['CholCheck', 'Smoker', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 
                   'AnyHealthcare', 'NoDocbcCost', 'MentHlth', 'Sex']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Dropping rows with invalid target values (if any)
df = df[df['Diabetes_binary'].isin([0.0, 1.0])].copy()

# Section 2: Preprocessing Numeric Columns
# Defining numeric columns
numeric_columns = ['BMI', 'GenHlth', 'PhysHlth', 'Age', 'Education', 'Income']

# Converting to numeric and handling outliers
num_imputer = SimpleImputer(strategy='median')
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    # Capping outliers using IQR (tighter threshold)
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(lower=Q1 - 1.25 * IQR, upper=Q3 + 1.25 * IQR)
df[numeric_columns] = num_imputer.fit_transform(df[numeric_columns])

# Section 3: Preprocessing Binary Columns
# Defining binary columns
binary_columns = ['HighBP', 'HighChol', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'DiffWalk']

# Converting to 0/1 and imputing missing values
bin_imputer = SimpleImputer(strategy='most_frequent')
for col in binary_columns:
    df[col] = df[col].apply(lambda x: 1 if str(x).lower() in ['1', '1.0', 'yes', 'true'] 
                           else 0 if str(x).lower() in ['0', '0.0', 'no', 'false'] 
                           else np.nan)
df[binary_columns] = bin_imputer.fit_transform(df[binary_columns])

# Section 4: Feature Engineering
# Creating BMI_category
df['BMI_category'] = pd.cut(df['BMI'], 
                          bins=[0, 18.5, 25, 30, 100], 
                          labels=[0, 1, 2, 3], 
                          include_lowest=True)
df['BMI_category'] = df['BMI_category'].cat.codes
df['BMI_category'] = df['BMI_category'].replace(-1, 1)

# Creating a health condition index
df['Health_index'] = df[['HighBP', 'HighChol', 'Stroke', 'HeartDiseaseorAttack']].sum(axis=1)

# Adding interaction terms
df['BMI_Age_interaction'] = df['BMI'] * df['Age']
df['HighBP_HighChol_interaction'] = df['HighBP'] * df['HighChol']

# Adding polynomial features
df['BMI_squared'] = df['BMI'] ** 2
df['Age_squared'] = df['Age'] ** 2

# Checking for NaNs
if df.isna().any().any():
    print("NaNs found in columns:", df.columns[df.isna().any()].tolist())
    df = df.fillna(df.median(numeric_only=True))

# Section 5: Preparing Features and Target
# Encoding target
label_encoder = LabelEncoder()
df['Diabetes_binary'] = label_encoder.fit_transform(df['Diabetes_binary'])

# Splitting features and target
X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']

# Standardizing features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Final NaN check
if X.isna().any().any():
    print("NaNs in X after preprocessing:", X.columns[X.isna().any()].tolist())

# Section 6: Training the Model
# Applying Borderline-SMOTE to handle class imbalance
smote = BorderlineSMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, 
                                                    test_size=0.2, random_state=42)

# Initializing XGBoost Classifier
xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')

# Expanded hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
grid_search = GridSearchCV(xgb_model, param_grid, cv=10, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Cross-validation score
cv_scores = cross_val_score(best_model, X_train, y_train, cv=10, scoring='accuracy')
print("Cross-Validation Accuracy: {:.2f} ± {:.2f}".format(cv_scores.mean(), cv_scores.std()))

# Evaluating on test set
y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Section 7: Saving the Model and Artifacts
joblib.dump(best_model, 'diabetes_model.pkl')
joblib.dump(label_encoder, 'label_encoder_diabetes.pkl')
joblib.dump(scaler, 'scaler_diabetes.pkl')
joblib.dump(X_train.columns, 'columns_diabetes.pkl')
print("Model and artifacts saved successfully.")

# Section 8: Function for Loading and Predicting
def load_and_predict(new_data):
    model = joblib.load('diabetes_model.pkl')
    label_encoder = joblib.load('label_encoder_diabetes.pkl')
    scaler = joblib.load('scaler_diabetes.pkl')
    saved_columns = joblib.load('columns_diabetes.pkl')

    # Feature engineering for new data
    new_data['BMI_category'] = pd.cut(new_data['BMI'], 
                                     bins=[0, 18.5, 25, 30, 100], 
                                     labels=[0, 1, 2, 3], 
                                     include_lowest=True)
    new_data['BMI_category'] = new_data['BMI_category'].cat.codes
    new_data['BMI_category'] = new_data['BMI_category'].replace(-1, 1)
    new_data['Health_index'] = new_data[['HighBP', 'HighChol', 'Stroke', 'HeartDiseaseorAttack']].sum(axis=1)
    new_data['BMI_Age_interaction'] = new_data['BMI'] * new_data['Age']
    new_data['HighBP_HighChol_interaction'] = new_data['HighBP'] * new_data['HighChol']
    new_data['BMI_squared'] = new_data['BMI'] ** 2
    new_data['Age_squared'] = new_data['Age'] ** 2

    # Ensuring same columns
    new_data = new_data.reindex(columns=saved_columns, fill_value=0)

    # Scaling features
    new_data = scaler.transform(new_data)

    # Making predictions
    predictions = model.predict(new_data)
    predicted_labels = label_encoder.inverse_transform(predictions)
    # Convert to human-readable labels
    readable_labels = ['No Diabetes' if label == 0 else 'Diabetes' for label in predicted_labels]

    return readable_labels

# Section 9: Example Prediction
new_data = pd.DataFrame({
    'HighBP': [1], 'HighChol': [1], 'BMI': [30.0], 'Stroke': [0], 
    'HeartDiseaseorAttack': [0], 'PhysActivity': [1], 'GenHlth': [3.0], 
    'PhysHlth': [5.0], 'DiffWalk': [0], 'Age': [50], 'Education': [4.0], 
    'Income': [6.0]
})

predictions = load_and_predict(new_data)
print("\nPrediction for new data:", predictions)