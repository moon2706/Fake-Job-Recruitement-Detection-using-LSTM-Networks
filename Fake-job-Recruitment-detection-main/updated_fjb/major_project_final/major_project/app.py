import os
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import nltk
from flask import Flask, request, render_template, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.under_sampling import RandomUnderSampler
from nltk.corpus import stopwords
import pickle
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

nltk.download("stopwords")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Load and preprocess the dataset
def load_data():
    data = pd.read_csv("fake_job_posting.csv")
    data.fillna(' ', inplace=True)
    stop_words = set(stopwords.words("english"))
    data['text'] = (data['title'] + ' ' + data['location'] + ' ' + data['company_profile'] + ' ' + 
                    data['description'] + ' ' + data['requirements'] + ' ' + data['benefits'] + ' ' + data['industry'])
    data.drop(['job_id', 'title', 'location', 'department', 'company_profile', 'description', 
               'requirements', 'benefits', 'required_experience', 'required_education', 'industry', 
               'function', 'employment_type'], axis=1, inplace=True)
    data['text'] = data['text'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    # Undersample the data
    X = data['text']
    y = data['fraudulent']
    under_sampler = RandomUnderSampler()
    X_res, y_res = under_sampler.fit_resample(X.values.reshape(-1, 1), y)
    data_resampled = pd.DataFrame({'text': X_res.flatten(), 'fraudulent': y_res})

    return data_resampled

# Train the model
def train_model():
    data = load_data()
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['fraudulent'], test_size=0.3, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(C=1))  # Adjusted C value
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = pipeline.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return pipeline

# Load the trained model
model = train_model()

# Function to check if uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Extract text from PDF and check for job-related keywords
def extract_text_from_pdf(pdf_path, keywords):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text += page.get_text()
    text = text.lower()
    keyword_count = sum(text.count(keyword) for keyword in keywords)
    return text if keyword_count >= 3 else None  # Lowered keyword threshold

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for PDF upload page
@app.route('/upload_pdf', methods=['GET', 'POST'])
def upload_pdf():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            job_keywords = ['job', 'position', 'hiring', 'vacancy', 'employment', 'salary', 'experience', 'recruitment', 'recruiting', 'career', 'opportunity']

            text = extract_text_from_pdf(filepath, job_keywords)
            if text:
                prediction = model.predict([text])[0]
                result = "FAKE" if prediction == 1 else "REAL"
            else:
                result = "UNRELATED CONTENT"
            return render_template('result.html', result=result)

    return render_template('upload_pdf.html')

# Route for text description input page
@app.route('/enter_description', methods=['GET', 'POST'])
def enter_description():
    if request.method == 'POST':
        description = request.form['description']
        prediction = model.predict([description.lower()])[0]
        result = "FAKE" if prediction == 1 else "REAL"
        return render_template('result.html', result=result)
    return render_template('enter_description.html')

# Route for job link input page



@app.route('/enter_link', methods=['GET', 'POST'])
def enter_link():
    if request.method == 'POST':
        job_link = request.form['job_link']
        try:
            response = requests.get(job_link)
            if response.status_code == 200:
                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Example: Get job description from LinkedIn
                description = soup.find('section', {'class': 'description'}).get_text()

                if description:
                    prediction = model.predict([description.lower()])[0]
                    result = "FAKE" if prediction == 1 else "REAL"
                else:
                    result = "INVALID JOB DESCRIPTION"
            else:
                result = "INVALID LINK OR UNREACHABLE PAGE"
        except Exception as e:
            result = "ERROR PROCESSING LINK"
        
        return render_template('result.html', result=result)

    return render_template('enter_link.html')




if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
