# Heart Shield AI

**Heart Shield** is an AI-powered web application that predicts an individual's risk of developing heart disease. Users can manually enter health data or upload medical documents (lab reports) for automatic data extraction using OCR. The system provides risk categorization: **Low**, **Moderate**, or **High**.


---


##  Features

-  **User Authentication** – Sign-up, login, email verification, password reset
-  **Manual Data Input** – Enter health metrics manually
-  **Document Upload** – Upload medical reports (PDF/images) with OCR auto-extraction
-  **ML Predictions** – Multiple models (Logistic Regression, Random Forest, SVM, Neural Networks)
-  **Dashboard** – Interactive results with risk visualization (color-coded badges)
-  **History Log** – View all past predictions
-  **User Profile** – Manage account details

---

##  Tech Stack

| Category | Technologies |
|----------|--------------|
| **Backend** | Python, Flask, SQLite, Flask-Mail |
| **Machine Learning** | Scikit-learn, TensorFlow/Keras, Pandas, NumPy |
| **OCR** | Tesseract, OpenCV, PyTesseract |
| **Frontend** | HTML, CSS, JavaScript, Bootstrap |
| **Version Control** | Git, GitHub |

---

##  Project Structure
```
HeartShield/
├── app/ # Web app (Flask routes, templates, static files)
├── data/ # Datasets & trained models (.pkl, .keras)
├── src/ # ML scripts (training, tuning, evaluation, OCR)
├── reports/ # Performance graphs & metrics
├── requirements.txt
└── run.py # Start the app
```

---

##  Quick Start

```bash
# Clone
git clone https://github.com/your-username/HeartShield.git
cd HeartShield

# Setup
python -m venv venv
venv\Scripts\activate      # or: source venv/bin/activate
pip install -r requirements.txt

# Run
python run.py
```
Note: Install Tesseract OCR for document upload feature.

---

## Usage

Web Application
- Register a new account
- Login to your dashboard
- Enter data in two ways:
  - Manual Input: Fill in health metrics (age, cholesterol, blood pressure, etc.)
  - Upload Report: Upload a medical document (OCR auto-fills the form)
- View Results – Get risk prediction (Low/Moderate/High) with confidence score
- Track History – All past predictions saved to your profile

---

## Model Performance

**Best Model:** Random Forest – **88.5% accuracy**

| Model | Accuracy |
|-------|----------|
| Random Forest | 88.5% |
| Logistic Regression | 87.2% |
| SVM | 86.8% |
| Neural Network | 85.9% |

**Risk Levels:** 🟢 Low (<30%) | 🟡 Moderate (30-70%) | 🔴 High (>70%)

---

## Future Scope

- Integration with electronic health records (EHR)
- Support for more medical document formats
- Multi-language OCR support
- Deployment on cloud (AWS/GCP)

---
### 📄 License

This project is for educational and non-commercial use only:)
