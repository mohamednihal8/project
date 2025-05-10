from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load symptom-based data
desc_data = pd.read_csv('dataset/symptom_Description.csv')
precaution_data = pd.read_csv('dataset/symptom_precaution.csv')
symptom_columns = joblib.load('models/disease_symptom_columns.pkl')
disease_model = joblib.load('models/disease_rf_model.pkl')
disease_le = joblib.load('models/disease_label_encoder.pkl')

# Load other models
heart_model = joblib.load('models/heart_model.pkl')
heart_scaler = joblib.load('models/heart_scaler.pkl')
lung_model = joblib.load('models/lung_cancer_model.pkl')
lung_preprocessor = joblib.load('models/lung_preprocessor.pkl')
thyroid_model = joblib.load('models/thyroid_rf_model.pkl')
thyroid_preprocessor = joblib.load('models/thyroid_preprocessor.pkl')
diabetes_model = joblib.load('models/diabetes_model.pkl')
diabetes_scaler = joblib.load('models/diabetes_scaler.pkl')

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        return jsonify({'status': 'success', 'message': 'Thank you for your message!'})
    return render_template('contact.html')

@app.route('/symptom', methods=['GET', 'POST'])
def symptom():
    if request.method == 'POST':
        symptoms = request.form['symptoms'].split(',')
        symptoms = [s.strip() for s in symptoms]
        
        symptom_vector = pd.DataFrame(np.zeros((1, len(symptom_columns))), columns=symptom_columns)
        for symptom in symptoms:
            if symptom in symptom_columns:
                symptom_vector[symptom] = 1
        
        prediction = disease_model.predict(symptom_vector)[0]
        disease = disease_le.inverse_transform([prediction])[0]
        
        desc_row = desc_data[desc_data['Disease'] == disease]
        description = desc_row['Description'].iloc[0] if not desc_row.empty else "Description not available"
        prec_row = precaution_data[precaution_data['Disease'] == disease]
        precautions = prec_row[[col for col in prec_row.columns if col.startswith('Precaution_')]].values[0] if not prec_row.empty else []
        precautions = [p for p in precautions if pd.notna(p)]
        
        return jsonify({
            'disease': disease,
            'description': description,
            'precautions': precautions
        })
    return render_template('symptom.html')

@app.route('/heart', methods=['GET', 'POST'])
def heart():
    if request.method == 'POST':
        data = [
            float(request.form['age']),
            1 if request.form['sex'] == 'M' else 0,
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal'])
        ]
        df = pd.DataFrame([data], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                                           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
        scaled_data = heart_scaler.transform(df)
        prediction = heart_model.predict(scaled_data)[0]
        result = "Heart Disease Present" if prediction == 1 else "No Heart Disease"
        return jsonify({'result': result})
    return render_template('heart.html')

@app.route('/lung', methods=['GET', 'POST'])
def lung():
    if request.method == 'POST':
        data = [
            request.form['gender'],
            int(request.form['age']),
            int(request.form['smoking']),
            int(request.form['yellow_fingers']),
            int(request.form['anxiety']),
            int(request.form['peer_pressure']),
            int(request.form['chronic_disease']),
            int(request.form['fatigue']),
            int(request.form['allergy']),
            int(request.form['wheezing']),
            int(request.form['alcohol']),
            int(request.form['coughing']),
            int(request.form['shortness_breath']),
            int(request.form['swallowing_difficulty']),
            int(request.form['chest_pain'])
        ]
        # Use exact column names with trailing spaces to match preprocessor
        feature_names = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
                         'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING', 
                         'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']
        df = pd.DataFrame([data], columns=feature_names)
        # Transform binary features (1=No, 2=Yes to 0=No, 1=Yes)
        binary_cols = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 
                       'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 
                       'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']
        df[binary_cols] = df[binary_cols].replace({1: 0, 2: 1})
        processed_data = lung_preprocessor.transform(df)
        prediction = lung_model.predict(processed_data)[0]
        result = "Lung Cancer Present" if prediction == 1 else "No Lung Cancer"
        return jsonify({'result': result})
    return render_template('lung.html')

@app.route('/thyroid', methods=['GET', 'POST'])
def thyroid():
    if request.method == 'POST':
        data = [
            int(request.form['age']),
            request.form['gender'],
            request.form['smoking'],
            request.form['hx_smoking'],
            request.form['hx_radiothreapy'],
            request.form['thyroid_function'],
            request.form['physical_examination'],
            request.form['adenopathy'],
            request.form['pathology'],
            request.form['focality'],
            request.form['risk'],
            request.form['t'],
            request.form['n'],
            request.form['m'],
            request.form['stage'],
            request.form['response']
        ]
        feature_names = ['Age', 'Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy', 
                         'Thyroid Function', 'Physical Examination', 'Adenopathy', 
                         'Pathology', 'Focality', 'Risk', 'T', 'N', 'M', 'Stage', 'Response']
        df = pd.DataFrame([data], columns=feature_names)
        df[['Smoking', 'Hx Smoking', 'Hx Radiothreapy']] = df[['Smoking', 'Hx Smoking', 'Hx Radiothreapy']].replace({'Yes': 1, 'No': 0})
        processed_data = thyroid_preprocessor.transform(df)
        prediction = thyroid_model.predict(processed_data)[0]
        result = "Recurrence Likely" if prediction == 1 else "No Recurrence"
        return jsonify({'result': result})
    return render_template('thyroid.html')

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        data = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['blood_pressure']),
            float(request.form['skin_thickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['dpf']),
            float(request.form['age'])
        ]
        df = pd.DataFrame([data], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        scaled_data = diabetes_scaler.transform(df)
        prediction = diabetes_model.predict(scaled_data)[0]
        result = "Diabetic" if prediction == 1 else "Not Diabetic"
        return jsonify({'result': result})
    return render_template('diabetes.html')

if __name__ == '__main__':
    app.run(debug=True)