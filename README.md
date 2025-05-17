Multi-Model Pregnancy Health Prediction System
Overview
The Multi-Model Pregnancy Health Prediction System is a web application built with Streamlit that integrates three machine learning models to predict pregnancy-related health risks. The system aims to assist healthcare professionals and patients by providing early risk assessments for Gestational Diabetes, Maternal Health Risks, and Preeclampsia, enabling timely interventions.
Aim
The primary goal of this project is to develop a user-friendly tool that leverages machine learning to predict the following pregnancy-related risks:

Gestational Diabetes: Identifies the likelihood of developing gestational diabetes mellitus (GDM).
Maternal Health Risk: Assesses overall maternal health risk levels (low, mid, high).
Preeclampsia: Predicts the risk of preeclampsia, a serious pregnancy complication.

By providing these predictions, the system aims to support early diagnosis and management of pregnancy complications, improving maternal and fetal outcomes.
Features

Gestational Diabetes Prediction:
Predicts whether a patient has GDM (GDM or Non GDM).
Uses features like Age, BMI, OGTT, and family history.


Maternal Health Risk Prediction:
Classifies maternal health risk into low risk, mid risk, or high risk.
Incorporates features such as Age, Blood Pressure, and Blood Sugar levels.


Preeclampsia Risk Prediction:
Predicts preeclampsia risk as low, mid, or high.
Utilizes features including Gestational Age, BMI, and Blood Pressure.


Interactive Interface:
Built with Streamlit, featuring three tabs for each prediction model.
User-friendly forms for inputting patient data.
Displays predictions with interpretive statements for actionable insights.



Directory Structure
C:.
+---Back-End
|   +---Gestational_Diabetic
|   |       columns_gdm.pkl              # GDM model columns list
|   |       gdm_model.pkl                # Trained GDM model
|   |       Gestational_Diabetic.csv     # Dataset for GDM model training
|   |       label_encoder_gdm.pkl        # Label encoder for GDM target
|   |       model.py                     # Script to train GDM model
|   |       scaler_gdm.pkl               # Scaler for GDM features
|   |
|   +---Maternal_Health
|   |       columns_maternal.pkl         # Maternal Health model columns list
|   |       label_encoder_maternal.pkl   # Label encoder for Maternal Health target
|   |       Maternal Health Risk Data Set.csv  # Dataset for Maternal Health model training
|   |       maternal_risk_model.pkl      # Trained Maternal Health model
|   |       model.py                     # Script to train Maternal Health model
|   |       scaler_maternal.pkl          # Scaler for Maternal Health features
|   |
|   +---Preeclampsia
|   |   |   model.ipynb                  # Jupyter notebook to train Preeclampsia model
|   |   |   model.py                     # Python script version of Preeclampsia model training
|   |   |   Preeclampsia.csv             # Dataset for Preeclampsia model training
|   |   |
|   |   \---models
|   |           columns_improved.pkl     # Preeclampsia model columns list
|   |           label_encoder_improved.pkl  # Label encoder for Preeclampsia target
|   |           preeclampsia_model_improved.pkl  # Trained Preeclampsia model
|   |           scaler.pkl               # Scaler for Preeclampsia features
|   |
|   \---streamlit
|           app.py                       # Main Streamlit application script
|
\---Documents
        Project-Synopsis.pdf             # Project synopsis document

Prerequisites

Python 3.8+: Ensure Python is installed on your system.
Required Libraries:
streamlit: For the web interface.
pandas, numpy: For data manipulation.
scikit-learn: For machine learning utilities (scaling, label encoding).
xgboost: For the XGBoost models used in training.
imblearn: For handling imbalanced datasets (SMOTE).
joblib: For loading model artifacts.


Trained Model Artifacts:
Ensure all .pkl files (models, scalers, label encoders, columns) are present in their respective directories as shown above.



Setup Instructions

Clone or Download the Project:

If using a repository, clone it:git clone <repository-url>


Otherwise, ensure all files are in the directory structure shown above.


Install Dependencies:

Navigate to the project root directory (Back-End/).
Install the required Python libraries:pip install streamlit pandas numpy scikit-learn xgboost imblearn joblib




Verify Model Artifacts:

Ensure the following artifacts are present:
Back-End/Gestational_Diabetic/: gdm_model.pkl, scaler_gdm.pkl, label_encoder_gdm.pkl, columns_gdm.pkl.
Back-End/Maternal_Health/: maternal_risk_model.pkl, scaler_maternal.pkl, label_encoder_maternal.pkl, columns_maternal.pkl.
Back-End/Preeclampsia/models/: preeclampsia_model_improved.pkl, scaler.pkl, label_encoder_improved.pkl, columns_improved.pkl.


If any artifacts are missing, train the models using the respective scripts:
GDM: Run Back-End/Gestational_Diabetic/model.py.
Maternal Health: Run Back-End/Maternal_Health/model.py.
Preeclampsia: Run Back-End/Preeclampsia/model.ipynb or model.py.




Navigate to the Streamlit Directory:
cd Back-End/streamlit


Run the Application:
streamlit run app.py


This will launch the app in your default browser at http://localhost:8501.



Usage

Launch the App:

After running streamlit run app.py, the app will open in your browser.


Select a Tab:

The app has three tabs:
Gestational Diabetes Prediction
Maternal Health Risk Prediction
Preeclampsia Risk Prediction




Enter Patient Details:

Each tab contains a form with input fields specific to the model.
Example inputs:
Gestational Diabetes:
Age: 30
Number of Pregnancies: 2
BMI: 28.0
HDL: 50.0
Family History: No (0)
Large Child or Birth Defect: No (0)
PCOS: No (0)
Systolic BP: 120.0
Diastolic BP: 80.0
OGTT: 130.0
Sedentary Lifestyle: No (0)
Prediabetes: No (0)


Maternal Health:
Age: 25
Systolic BP: 120.0
Diastolic BP: 80.0
Blood Sugar: 7.0
Heart Rate: 70


Preeclampsia:
Gravida: 3
Parity: 1
Gestational Age (Weeks): 22.2
Age: 25
BMI: 21.0
Diabetes: No (0)
Hypertension History: No (0)
Systolic BP: 110.0
Diastolic BP: 70.0
Hemoglobin: 9.3
Fetal Weight: 0.501
Protein Uria: No (0)
Amniotic Fluid Levels: 10.0






Submit and View Results:

Click the "Predict" button in the respective tab.
The app will display the prediction and an interpretive statement.
Example outputs:
Gestational Diabetes: "Gestational Diabetes Prediction: Non GDM"
Interpretation: "This indicates no immediate concern for gestational diabetes, but continue monitoring."


Maternal Health: "Maternal Health Risk: low risk"
Interpretation: "This indicates a low risk level, but regular monitoring is recommended."


Preeclampsia: "Preeclampsia Risk Prediction: low"
Interpretation: "This indicates a low risk level, but regular monitoring is recommended."







Model Details

Gestational Diabetes Prediction:

Features: Age, No of Pregnancy, BMI, HDL, Family History, Large Child or Birth Defect, PCOS, Systolic BP, Diastolic BP, OGTT, Sedentary Lifestyle, Prediabetes.
Feature Engineering: BP_ratio, Age_category, High_OGTT, BMI_category, OGTT_BMI_interaction, Age_OGTT_interaction.
Model: XGBoost (binary classification).
Accuracy: ~80% (based on training script output).


Maternal Health Risk Prediction:

Features: Age, Systolic BP, Diastolic BP, Blood Sugar (BS), Heart Rate.
Feature Engineering: BP_ratio, Age_category, High_BS, SystolicBP_BS_interaction, Age_BS_interaction.
Model: XGBoost (multiclass classification).
Accuracy: ~80% (based on training script output).


Preeclampsia Risk Prediction:

Features: Gravida, Parity, Gestational Age (Weeks), Age, BMI, Diabetes, Hypertension History, Systolic BP, Diastolic BP, Hemoglobin (HB), Fetal Weight, Protein Uria, Amniotic Fluid Levels.
Feature Engineering: BP_ratio, BMI_category.
Model: XGBoost (multiclass classification).
Accuracy: ~81% (as per model.ipynb output).



Troubleshooting

FileNotFoundError for Model Artifacts:

Error: "Gestational Diabetes model artifacts not found..."
Solution:
Ensure .pkl files are in the correct directories:
Back-End/Gestational_Diabetic/
Back-End/Maternal_Health/
Back-End/Preeclampsia/models/


Re-run the training scripts if artifacts are missing:
GDM: Back-End/Gestational_Diabetic/model.py
Maternal Health: Back-End/Maternal_Health/model.py
Preeclampsia: Back-End/Preeclampsia/model.ipynb






Gestational Diabetes Model Fails:

Issue: The GDM model may fail due to an issue with Class_Label in Gestational_Diabetic.csv.
Solution:
Open Back-End/Gestational_Diabetic/Gestational_Diabetic.csv.
Ensure the Class_Label column contains valid values (GDM or Non GDM).
Re-run Back-End/Gestational_Diabetic/model.py to regenerate artifacts.




Incorrect Predictions:

Issue: Predictions don’t match expected outcomes.
Solution:
Verify input values are within realistic ranges (e.g., Age 18-50 for GDM).
Check if the model artifacts were trained correctly by reviewing training script outputs.




Streamlit App Doesn’t Launch:

Issue: streamlit run app.py fails.
Solution:
Ensure you’re in the Back-End/streamlit/ directory.
Check for missing dependencies: pip install -r requirements.txt (create one if needed).





Contributing
Contributions are welcome! To contribute:

Fork the repository (if applicable).
Create a new branch for your feature or bug fix:git checkout -b feature-name


Make your changes and test thoroughly.
Submit a pull request with a detailed description of your changes.

License
[Add your license here, e.g., MIT License, Apache License, etc.]
Acknowledgments

Datasets:
Gestational Diabetes: Gestational_Diabetic.csv
Maternal Health: Maternal Health Risk Data Set.csv
Preeclampsia: Preeclampsia.csv


Tools:
Streamlit for the web interface.
XGBoost for model training.
Scikit-learn and Imblearn for data preprocessing.



