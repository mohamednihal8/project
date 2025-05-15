import streamlit as st
import pandas as pd
import joblib
import ast

# Load models
@st.cache_resource
modelrf = joblib.load('models\rf_disease_model.pkl')
scalerrf = joblib.load('models\scaler.pkl')

# Define numerical columns
numerical_cols = ['Age', 'BMI', 'Hemoglobin', 'WBC', 'RBC', 'Platelets', 'Glucose_Fasting', 
                 'HbA1c', 'Cholesterol_Total', 'Triglycerides', 'LDL', 'HDL', 'ALT', 'AST', 
                 'Bilirubin_Total', 'Creatinine', 'Urea', 'TSH', 'T3', 'T4', 'CRP']

# Define feature columns to match training data
X_train_columns = numerical_cols + ['Gender', 'Smoker', 'Alcohol_Intake', 'COVID_Antigen_Pos', 
                                   'HIV_Positive', 'Hepatitis_B_Pos', 'Hepatitis_C_Pos', 
                                   'Malaria_Positive', 'Tuberculosis_Pos', 
                                   'Exercise_Daily', 'Exercise_Rare', 'Exercise_Weekly']

def predict_disease(new_data, modelrf, scalerrf):
    # Map input categories to training categories
    exercise_mapping = {
        'Never': 'Rare',
        'Occasionally': 'Weekly',
        'Regularly': 'Daily',
        'Rarely': 'Rare'  # Added to handle text input example
    }
    new_data['Exercise_Freq'] = new_data['Exercise_Freq'].map(exercise_mapping)
    
    # One-hot encode Exercise_Freq
    new_data = pd.get_dummies(new_data, columns=['Exercise_Freq'], prefix='Exercise')
    
    # Align columns with training data
    missing_cols = set(X_train_columns) - set(new_data.columns)
    for col in missing_cols:
        new_data[col] = 0
    new_data = new_data[X_train_columns]
    
    # Scale numerical features
    new_data_scaled = scalerrf.transform(new_data[numerical_cols])
    new_data_scaled = pd.DataFrame(new_data_scaled, columns=numerical_cols, index=new_data.index)
    new_data_scaled = pd.concat([new_data_scaled, new_data.drop(columns=numerical_cols)], axis=1)
    
    # Predict
    prediction = modelrf.predict(new_data_scaled)[0]
    probabilities = modelrf.predict_proba(new_data_scaled)[0]
    prob_dict = dict(zip(modelrf.classes_, probabilities))
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    
    return prediction, prob_dict, sorted_probs

def parse_text_input(text_input):
    try:
        # Remove square brackets from values and convert to dictionary
        text_input = text_input.replace('[', '').replace(']', '')
        # Convert string to dictionary
        input_dict = ast.literal_eval('{' + text_input + '}')
        # Ensure single values are in lists
        input_dict = {k: [v] for k, v in input_dict.items()}
        return pd.DataFrame(input_dict)
    except Exception as e:
        st.error(f"Error parsing text input: {str(e)}. Please ensure the format matches the example.")
        return None

def main():
    st.set_page_config(page_title="Predictive Disease Diagnosis", layout="centered")
    st.title("Predictive Disease Diagnosis")
    st.markdown("Enter patient details to predict potential diseases. Use the form or paste text input at the bottom. All fields are required for form input.")

    modelrf, scalerrf = load_models()
    if modelrf is None or scalerrf is None:
        return

    with st.form("patient_form"):
        # General Information
        with st.expander("General Information", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=0, max_value=120, value=70, help="Patient's age in years")
                gender = st.selectbox("Gender", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0], help="Select patient's gender")
                bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=26.0, step=0.1, help="Body Mass Index")
            with col2:
                smoker = st.selectbox("Smoker", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0], help="Does the patient smoke?")
                alcohol = st.selectbox("Alcohol Intake", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0], help="Does the patient consume alcohol?")
                exercise = st.selectbox("Exercise Frequency", options=["Never", "Occasionally", "Regularly"], help="How often does the patient exercise?")

        # Blood Tests
        with st.expander("Blood Tests"):
            col1, col2 = st.columns(2)
            with col1:
                hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=20.0, value=14.0, step=0.1, help="Hemoglobin level")
                wbc = st.number_input("WBC (x10³/μL)", min_value=0.0, max_value=50.0, value=6.0, step=0.1, help="White Blood Cell count")
                rbc = st.number_input("RBC (x10⁶/μL)", min_value=0.0, max_value=10.0, value=4.9, step=0.1, help="Red Blood Cell count")
                platelets = st.number_input("Platelets (x10³/μL)", min_value=0, max_value=1000, value=260, help="Platelet count")
            with col2:
                alt = st.number_input("ALT (U/L)", min_value=0, max_value=100, value=22, help="Alanine Aminotransferase")
                ast = st.number_input("AST (U/L)", min_value=0, max_value=100, value=20, help="Aspartate Aminotransferase")
                bilirubin = st.number_input("Bilirubin Total (mg/dL)", min_value=0.0, max_value=5.0, value=0.8, step=0.1, help="Total Bilirubin")
                crp = st.number_input("CRP (mg/L)", min_value=0.0, max_value=50.0, value=3.0, step=0.1, help="C-reactive Protein")

        # Metabolic Panel
        with st.expander("Metabolic Panel"):
            col1, col2 = st.columns(2)
            with col1:
                glucose = st.number_input("Glucose Fasting (mg/dL)", min_value=0, max_value=500, value=90, help="Fasting glucose level")
                hba1c = st.number_input("HbA1c (%)", min_value=0.0, max_value=20.0, value=5.4, step=0.1, help="Glycated hemoglobin")
                cholesterol = st.number_input("Cholesterol Total (mg/dL)", min_value=0, max_value=500, value=180, help="Total cholesterol")
                triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=0, max_value=1000, value=120, help="Triglyceride level")
            with col2:
                ldl = st.number_input("LDL (mg/dL)", min_value=0, max_value=300, value=105, help="Low-density lipoprotein")
                hdl = st.number_input("HDL (mg/dL)", min_value=0, max_value=100, value=55, help="High-density lipoprotein")
                creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, help="Creatinine level")
                urea = st.number_input("Urea (mg/dL)", min_value=0, max_value=100, value=21, help="Urea level")

        # Thyroid Function
        with st.expander("Thyroid Function"):
            col1, col2 = st.columns(2)
            with col1:
                tsh = st.number_input("TSH (mIU/L)", min_value=0.0, max_value=10.0, value=2.7, step=0.1, help="Thyroid-Stimulating Hormone")
                t3 = st.number_input("T3 (ng/mL)", min_value=0.0, max_value=5.0, value=1.2, step=0.1, help="Triiodothyronine")
            with col2:
                t4 = st.number_input("T4 (μg/dL)", min_value=0.0, max_value=20.0, value=8.2, step=0.1, help="Thyroxine")

        # Infectious Diseases
        with st.expander("Infectious Diseases"):
            col1, col2 = st.columns(2)
            with col1:
                covid = st.selectbox("COVID Antigen Positive", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], help="COVID-19 antigen test result")
                hiv = st.selectbox("HIV Positive", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], help="HIV test result")
                hep_b = st.selectbox("Hepatitis B Positive", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], help="Hepatitis B test result")
            with col2:
                hep_c = st.selectbox("Hepatitis C Positive", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], help="Hepatitis C test result")
                malaria = st.selectbox("Malaria Positive", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], help="Malaria test result")
                tb = st.selectbox("Tuberculosis Positive", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], help="Tuberculosis test result")

        # Text Input for Dictionary
        with st.expander("Text Input (Alternative)", expanded=False):
            st.markdown("Paste patient data in the format below (omit curly braces):")
            st.code("""
            'Age': [35], 'Gender': [0], 'BMI': [35.5], 'Hemoglobin': [13.0], 'WBC': [9.5], 'RBC': [4.7], 
            'Platelets': [220], 'Glucose_Fasting': [110], 'HbA1c': [6.0], 'Cholesterol_Total': [210], 
            'Triglycerides': [200], 'LDL': [130], 'HDL': [35], 'ALT': [35], 'AST': [30], 
            'Bilirubin_Total': [0.9], 'Creatinine': [1.2], 'Urea': [28], 'TSH': [3.5], 
            'T3': [0.9], 'T4': [7.0], 'CRP': [15.0], 'Smoker': [0], 'Alcohol_Intake': [1], 
            'COVID_Antigen_Pos': [0], 'HIV_Positive': [0], 'Hepatitis_B_Pos': [0], 'Hepatitis_C_Pos': [0], 
            'Malaria_Positive': [0], 'Tuberculosis_Pos': [1], 'Exercise_Freq': ['Rarely']
            """)
            text_input = st.text_area("Enter patient data (dictionary format without curly braces):", height=200)

        submitted = st.form_submit_button("Predict Disease")

    if submitted:
        try:
            if text_input.strip():
                # Process text input
                input_data = parse_text_input(text_input)
                if input_data is None:
                    return
            else:
                # Validate form inputs
                if any(v is None for v in [age, bmi, hemoglobin, wbc, rbc, platelets, glucose, hba1c, cholesterol, 
                                        triglycerides, ldl, hdl, alt, ast, bilirubin, creatinine, urea, tsh, t3, t4, crp]):
                    st.error("Please fill in all numerical fields with valid values.")
                    return

                # Create DataFrame from form inputs
                input_data = pd.DataFrame({
                    'Age': [age], 'Gender': [gender[1]], 'BMI': [bmi], 'Hemoglobin': [hemoglobin], 
                    'WBC': [wbc], 'RBC': [rbc], 'Platelets': [platelets], 'Glucose_Fasting': [glucose], 
                    'HbA1c': [hba1c], 'Cholesterol_Total': [cholesterol], 'Triglycerides': [triglycerides], 
                    'LDL': [ldl], 'HDL': [hdl], 'ALT': [alt], 'AST': [ast], 'Bilirubin_Total': [bilirubin], 
                    'Creatinine': [creatinine], 'Urea': [urea], 'TSH': [tsh], 'T3': [t3], 'T4': [t4], 
                    'CRP': [crp], 'Smoker': [smoker[1]], 'Alcohol_Intake': [alcohol[1]], 
                    'COVID_Antigen_Pos': [covid[1]], 'HIV_Positive': [hiv[1]], 'Hepatitis_B_Pos': [hep_b[1]], 
                    'Hepatitis_C_Pos': [hep_c[1]], 'Malaria_Positive': [malaria[1]], 'Tuberculosis_Pos': [tb[1]], 
                    'Exercise_Freq': [exercise]
                })

            # Predict
            prediction, prob_dict, sorted_probs = predict_disease(input_data, modelrf, scalerrf)
            
            # Display results
            st.subheader("Prediction Results")
            st.success(f"**Predicted Disease:** {prediction}")
            st.info(f"**Confidence:** {prob_dict[prediction]*100:.2f}%")
            
            st.subheader("Other Possible Diagnoses")
            for disease, prob in sorted_probs[1:5]:
                if prob > 0.005:  # Show only if probability > 0.5%
                    st.write(f"{disease}: {prob*100:.2f}%")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

if __name__ == "__main__":
    main()
