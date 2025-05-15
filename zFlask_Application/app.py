from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
from flask_sqlalchemy import SQLAlchemy
from config import Config
from models import db, User, Prediction
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
import bcrypt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
import os

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

# Ensure instance directory exists
metadir = os.path.join(app.config['BASE_DIR'], 'instance')
if not os.path.exists(metadir):
    os.makedirs(metadir)

# Load ML models
model = joblib.load('ml_models/rf_disease_model.pkl')
scaler = joblib.load('ml_models/scaler.pkl')

# Numerical and categorical columns
numerical_cols = ['Age', 'BMI', 'Hemoglobin', 'WBC', 'RBC', 'Platelets', 'Glucose_Fasting', 'HbA1c',
                  'Cholesterol_Total', 'Triglycerides', 'LDL', 'HDL', 'ALT', 'AST', 'Bilirubin_Total',
                  'Creatinine', 'Urea', 'TSH', 'T3', 'T4', 'CRP']
feature_cols = numerical_cols + ['Gender', 'Smoker', 'Alcohol_Intake', 'COVID_Antigen_Pos', 'HIV_Positive',
                                'Hepatitis_B_Pos', 'Hepatitis_C_Pos', 'Malaria_Positive', 'Tuberculosis_Pos',
                                'Exercise_Never', 'Exercise_Occasionally', 'Exercise_Rarely', 'Exercise_Regularly']

def create_pdf(prediction_data):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, "Predictive Diagnostics Report")
    c.drawString(100, 730, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(100, 710, f"Predicted Disease: {prediction_data['predicted_disease']}")
    c.drawString(100, 690, f"Confidence: {prediction_data['confidence']*100:.2f}%")
    c.drawString(100, 670, "Other Possibilities:")
    y = 650
    for disease, prob in prediction_data['other_probs']:
        c.drawString(120, y, f"{disease}: {prob*100:.2f}%")
        y -= 20
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password'].encode('utf-8')
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.checkpw(password, user.password.encode('utf-8')):
            session['user_id'] = user.id
            session['is_admin'] = user.is_admin
            flash('Login successful!', 'success')
            if user.is_admin:
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('user_dashboard'))
        flash('Invalid credentials!', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password'].encode('utf-8')
        hashed_password = bcrypt.hashpw(password, bcrypt.gensalt()).decode('utf-8')
        user = User(name=name, username=username, email=email, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Account created successfully!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('is_admin', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        flash('Please login to access this page!', 'danger')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        data = {
            'Age': float(request.form['Age']),
            'Gender': int(request.form['Gender']),
            'BMI': float(request.form['BMI']),
            'Hemoglobin': float(request.form['Hemoglobin']),
            'WBC': float(request.form['WBC']),
            'RBC': float(request.form['RBC']),
            'Platelets': float(request.form['Platelets']),
            'Glucose_Fasting': float(request.form['Glucose_Fasting']),
            'HbA1c': float(request.form['HbA1c']),
            'Cholesterol_Total': float(request.form['Cholesterol_Total']),
            'Triglycerides': float(request.form['Triglycerides']),
            'LDL': float(request.form['LDL']),
            'HDL': float(request.form['HDL']),
            'ALT': float(request.form['ALT']),
            'AST': float(request.form['AST']),
            'Bilirubin_Total': float(request.form['Bilirubin_Total']),
            'Creatinine': float(request.form['Creatinine']),
            'Urea': float(request.form['Urea']),
            'TSH': float(request.form['TSH']),
            'T3': float(request.form['T3']),
            'T4': float(request.form['T4']),
            'CRP': float(request.form['CRP']),
            'Smoker': int(request.form['Smoker']),
            'Alcohol_Intake': int(request.form['Alcohol_Intake']),
            'COVID_Antigen_Pos': int(request.form['COVID_Antigen_Pos']),
            'HIV_Positive': int(request.form['HIV_Positive']),
            'Hepatitis_B_Pos': int(request.form['Hepatitis_B_Pos']),
            'Hepatitis_C_Pos': int(request.form['Hepatitis_C_Pos']),
            'Malaria_Positive': int(request.form['Malaria_Positive']),
            'Tuberculosis_Pos': int(request.form['Tuberculosis_Pos']),
            'Exercise_Freq': request.form['Exercise_Freq']
        }
        new_data = pd.DataFrame([data])
        new_data = pd.get_dummies(new_data, columns=['Exercise_Freq'], prefix='Exercise')
        missing_cols = set(feature_cols) - set(new_data.columns)
        for col in missing_cols:
            new_data[col] = 0
        new_data = new_data[feature_cols]
        new_data_scaled = scaler.transform(new_data[numerical_cols])
        new_data_scaled = pd.DataFrame(new_data_scaled, columns=numerical_cols, index=new_data.index)
        new_data_scaled = pd.concat([new_data_scaled, new_data.drop(columns=numerical_cols)], axis=1)
        prediction = model.predict(new_data_scaled)[0]
        probabilities = model.predict_proba(new_data_scaled)[0]
        prob_dict = dict(zip(model.classes_, probabilities))
        confidence = prob_dict[prediction]
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[1:5]
        other_probs = [(disease, prob) for disease, prob in sorted_probs if prob > 0.005]
        
        prediction_entry = Prediction(
            user_id=session['user_id'], **data, predicted_disease=prediction, confidence=confidence
        )
        db.session.add(prediction_entry)
        db.session.commit()
        
        if 'download_pdf' in request.form:
            pdf_buffer = create_pdf({
                'predicted_disease': prediction,
                'confidence': confidence,
                'other_probs': other_probs
            })
            return send_file(
                pdf_buffer,
                as_attachment=True,
                download_name='prediction_report.pdf',
                mimetype='application/pdf'
            )
        
        return render_template('predict.html', prediction=prediction, confidence=confidence*100,
                             other_probs=other_probs)
    return render_template('predict.html')

@app.route('/user_dashboard')
def user_dashboard():
    if 'user_id' not in session:
        flash('Please login to access this page!', 'danger')
        return redirect(url_for('login'))
    predictions = Prediction.query.filter_by(user_id=session['user_id']).all()
    return render_template('user_dashboard.html', predictions=predictions)

@app.route('/admin_dashboard')
def admin_dashboard():
    if 'user_id' not in session or not session['is_admin']:
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('login'))
    users = User.query.all()
    predictions = Prediction.query.all()
    return render_template('admin_dashboard.html', users=users, predictions=predictions)

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)