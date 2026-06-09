import sys, os
from flask import render_template, redirect, url_for, request, Blueprint, flash, current_app, session
from flask_login import login_user, logout_user, login_required, current_user
from .models import User, PredictionHistory, Document
from . import db
from .email import send_email
from werkzeug.utils import secure_filename
import json

# Correctly set the path to the project root to import the pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.prediction_pipeline import HeartDiseasePredictionPipeline
from src.ocr_utils import run_ocr_pipeline

def map_string_to_int(data):
    for key, value in data.items():
        try:
            data[key] = int(value)
        except (ValueError, TypeError):
            pass
    return data

# Create a Blueprint for the main application routes
main_bp = Blueprint('main', __name__)

# Initialize the prediction pipeline once to avoid reloading the model on every request
pipeline = HeartDiseasePredictionPipeline()

# The home page redirects to the login page.
@main_bp.route('/')
def index():
    return redirect(url_for('main.register'))

@main_bp.route('/home')
@login_required
def home():
    return render_template('home.html')

# The login page route.
@main_bp.route('/login', methods=['GET', 'POST'])
def login():
    # If the user is already logged in, redirect to the new home page.
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))
    # If the request method is POST, it means the user has submitted the login form.
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        # Find the user by email in the database.
        user = User.query.filter_by(email=email).first()
        # If the user does not exist or the password is incorrect, flash an error message.
        if user is None or not user.check_password(password):
            flash('Invalid email or password')
            return redirect(url_for('main.login'))
        # If the credentials are correct, log the user in.
        login_user(user)
        return redirect(url_for('main.home'))
    # If the request method is GET, render the login page.
    return render_template('login.html')

# The logout route.
@main_bp.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('main.login'))

# The registration page route.
@main_bp.route('/register', methods=['GET', 'POST'])
def register():
    # If the user is already logged in, redirect to the dashboard.
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    # If the request method is POST, it means the user has submitted the registration form.
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        full_name = request.form.get('full_name') # Get full_name from form
        # Check if the email address is already in use.
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email address already exists')
            return redirect(url_for('main.register'))
        # Create a new user and save it to the database.
        new_user = User(username=username, email=email, full_name=full_name) # Pass full_name
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        # Send a confirmation email to the new user.
        token = new_user.generate_confirmation_token()
        confirm_url = url_for('main.confirm_email', token=token, _external=True)
        html = render_template('confirm_email.html', confirm_url=confirm_url, user=new_user)
        subject = "Please confirm your email"
        send_email(new_user.email, subject, html)
        flash('A confirmation email has been sent to you by email.')
        return redirect(url_for('main.login'))
    # If the request method is GET, render the registration page.
    return render_template('register.html')

# The email confirmation route.
@main_bp.route('/confirm/<token>')
@login_required
def confirm_email(token):
    # If the user's email is already confirmed, flash a message.
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    # Confirm the user's email address.
    if current_user.confirm(token):
        db.session.commit()
        flash('You have confirmed your account. Thanks!')
    else:
        flash('The confirmation link is invalid or has expired.')
    return redirect(url_for('main.dashboard'))

# The route to send a confirmation email.
@main_bp.route('/send-confirmation')
@login_required
def send_confirmation():
    # Generate a new confirmation token and send a confirmation email.
    token = current_user.generate_confirmation_token()
    confirm_url = url_for('main.confirm_email', token=token, _external=True)
    html = render_template('confirm_email.html', confirm_url=confirm_url)
    subject = "Please confirm your email"
    send_email(current_user.email, subject, html)
    flash('A new confirmation email has been sent to you by email.')
    return redirect(url_for('main.dashboard'))

# The password reset request route.
@main_bp.route('/reset-password-request', methods=['GET', 'POST'])
def reset_password_request():
    # If the user is already logged in, redirect to the dashboard.
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    # If the request method is POST, it means the user has submitted the password reset request form.
    if request.method == 'POST':
        email = request.form['email']
        user = User.query.filter_by(email=email).first()
        if user:
            # Generate a password reset token and send a password reset email.
            token = user.generate_reset_token()
            reset_url = url_for('main.reset_password', token=token, _external=True)
            html = render_template('reset_password_email.html', reset_url=reset_url)
            subject = "Password reset request"
            send_email(user.email, subject, html)
        flash('An email with instructions to reset your password has been sent to you.')
        return redirect(url_for('main.login'))
    # If the request method is GET, render the password reset request page.
    return render_template('reset_password_request.html')

# The password reset route.
@main_bp.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    # If the user is already logged in, redirect to the dashboard.
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    # Verify the password reset token.
    user = User.verify_reset_token(token)
    if not user:
        return redirect(url_for('main.login'))
    # If the request method is POST, it means the user has submitted the new password.
    if request.method == 'POST':
        password = request.form['password']
        user.set_password(password)
        db.session.commit()
        flash('Your password has been reset.')
        return redirect(url_for('main.login'))
    # If the request method is GET, render the password reset page.
    return render_template('reset_password.html')

    # Render the results page with the prediction results.
    return render_template('results.html', success=True, result=result, model_accuracy=pipeline.overall_model_accuracy)

@main_bp.route('/results')
@login_required
def results():
    predictions = PredictionHistory.query.filter_by(user_id=current_user.id).order_by(PredictionHistory.timestamp.desc()).all()
    return render_template('results.html', predictions=predictions, model_accuracy=pipeline.overall_model_accuracy)

# The file upload route.
@main_bp.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    # If the request method is POST, it means the user has uploaded a file.
    if request.method == 'POST':
        # Check if a file was uploaded.
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # Save the file to the uploads folder.
            filename = secure_filename(file.filename)
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Create a new Document object and save it to the database.
            new_document = Document(filename=filename, filepath=filepath, user_id=current_user.id)
            db.session.add(new_document)
            db.session.commit()

            # Perform OCR on the uploaded file and get the numeric data.
            ocr_mapped_data = run_ocr_pipeline(filepath)
            
            # Validate the OCR data
            is_valid, errors = pipeline.validate_input(ocr_mapped_data)

            if not is_valid:
                # Store data and errors in session and redirect to manual input
                session['ocr_mapped_data'] = ocr_mapped_data
                session['validation_errors'] = errors
                flash('Some data was missing or invalid. Please review and complete the form.', 'warning')
                return redirect(url_for('main.manual_input'))

            # If valid, proceed with prediction
            result = pipeline.predict(ocr_mapped_data)

            if not result.get('success'):
                errors = result.get('errors') or [result.get('error', 'Unknown error')]
                return render_template('results.html', success=False, errors=errors, model_accuracy=pipeline.overall_model_accuracy)

            # Save the prediction to the user's prediction history.
            prediction_to_save = PredictionHistory(
                input_data=json.dumps(ocr_mapped_data),
                prediction=result['prediction']['has_heart_disease'],
                probability=result['prediction']['probability'],
                user_id=current_user.id
            )
            db.session.add(prediction_to_save)
            db.session.commit()

            # Personalized Recommendations Logic
            recommendations = []
            risk_level = result.get('risk_assessment', {}).get('risk_category', 'Medium Risk')

            if risk_level == 'High Risk':
                recommendations = [
                    "Urgent: Schedule an appointment with a cardiologist as soon as possible.",
                    "Your doctor may recommend an electrocardiogram (ECG) and a stress test.",
                    "Strictly follow any prescribed medications and lifestyle changes.",
                    "Avoid strenuous activities until you have consulted with a doctor."
                ]
            elif risk_level == 'Medium Risk':
                recommendations = [
                    "Schedule a check-up with your primary care physician to discuss these results.",
                    "Focus on a heart-healthy diet: reduce sodium, saturated fats, and processed foods.",
                    "Increase regular, moderate physical activity to at least 150 minutes per week.",
                    "Monitor your blood pressure and cholesterol levels regularly."
                ]
            else: # Low Risk
                recommendations = [
                    "Excellent results! Continue to maintain your healthy lifestyle.",
                    "Engage in regular physical activity and maintain a balanced diet.",
                    "Annual check-ups with your doctor are still recommended for preventive care.",
                    "Be mindful of your family's health history and discuss it with your doctor."
                ]

            return render_template('results.html', success=True, result=result, model_accuracy=pipeline.overall_model_accuracy, recommendations=recommendations, body_class='results-page')

    # If the request method is GET, render the upload page.
    return render_template('upload.html')

# The dashboard page route.
@main_bp.route('/dashboard')
@login_required
def dashboard():
    # Get the user's prediction history and uploaded documents from the database.
    predictions = PredictionHistory.query.filter_by(user_id=current_user.id).order_by(PredictionHistory.timestamp.desc()).all()
    documents = Document.query.filter_by(user_id=current_user.id).order_by(Document.upload_date.desc()).all()
    latest_prediction = PredictionHistory.query.filter_by(user_id=current_user.id).order_by(PredictionHistory.timestamp.desc()).first()
    
    # Calculate stats for the new overview section
    total_assessments = len(predictions)
    risk_counts = {'High Risk': 0, 'Medium Risk': 0, 'Low Risk': 0}
    if total_assessments > 0:
        for p in predictions:
            # Determine risk category based on stored probability
            if p.prediction and p.probability >= 0.7:
                risk_counts['High Risk'] += 1
            elif (p.prediction and p.probability < 0.7) or (not p.prediction and p.probability >= 0.3):
                risk_counts['Medium Risk'] += 1
            else: # Not p.prediction and p.probability < 0.3
                risk_counts['Low Risk'] += 1

    # Prepare data for the CSS pie chart
    pie_chart_data = {
        'percentages': {k: (v / total_assessments * 100) if total_assessments > 0 else 0 for k, v in risk_counts.items()},
        'counts': risk_counts
    }

    # Generate conic-gradient style string for the pie chart
    conic_gradient_parts = []
    current_angle = 0
    color_map = {
        'High Risk': '#dc3545',
        'Medium Risk': '#ffc107',
        'Low Risk': '#28a745'
    }

    for risk_category in ['High Risk', 'Medium Risk', 'Low Risk']:
        percentage = pie_chart_data['percentages'][risk_category]
        if percentage > 0:
            start_angle = current_angle
            end_angle = current_angle + percentage * 3.6
            conic_gradient_parts.append(f"{color_map[risk_category]} {start_angle}deg {end_angle}deg")
            current_angle = end_angle
    
    
    latest_input_data = {} # Initialize as empty dict
    if latest_prediction and latest_prediction.input_data:
        try:
            latest_input_data = json.loads(latest_prediction.input_data)
        except json.JSONDecodeError:
            # Handle cases where input_data might not be valid JSON (e.g., old entries)
            latest_input_data = {"error": "Invalid JSON data in history"}

    # Render the dashboard page with the user's data.
    return render_template('dashboard.html', 
                           predictions=predictions, 
                           documents=documents, 
                           latest_input_data=latest_input_data,
                           total_assessments=total_assessments,
                           latest_assessment_date=latest_prediction.timestamp if latest_prediction else None,
                           pie_chart_data=pie_chart_data,
                           model_accuracy=pipeline.overall_model_accuracy,
                           conic_gradient_parts=conic_gradient_parts)

# The manual input page route.
@main_bp.route('/manual_input', methods=['GET'])
@login_required
def manual_input():
    ocr_data = session.pop('ocr_mapped_data', {})
    errors = session.pop('validation_errors', [])
    
    return render_template('manual_input.html', ocr_data=ocr_data, errors=errors, feature_descriptions=pipeline.feature_descriptions, feature_ranges=pipeline.feature_ranges, feature_names=pipeline.feature_names)

@main_bp.route('/manual_predict', methods=['POST'])
@login_required
def manual_predict():
    patient_data = {}
    for key in pipeline.feature_names:
        # Convert form data to appropriate types
        value = request.form.get(key)
        if value is not None and value != '':
            try:
                # Attempt to convert to float, then int if it's a whole number
                float_val = float(value)
                if float_val.is_integer():
                    patient_data[key] = int(float_val)
                else:
                    patient_data[key] = float_val
            except ValueError:
                patient_data[key] = value # Keep as string if conversion fails (e.g., for categorical text if we ever allow it)
        else:
            patient_data[key] = None # Explicitly set None for empty fields

    # Validate the manually entered data
    is_valid, errors = pipeline.validate_input(patient_data)

    if not is_valid:
        # If validation fails, re-render the manual input page with errors and current data
        flash('Please correct the highlighted errors.', 'danger')
        return render_template('manual_input.html', ocr_data=patient_data, errors=errors, feature_descriptions=pipeline.feature_descriptions, feature_ranges=pipeline.feature_ranges, feature_names=pipeline.feature_names)

    # If valid, proceed with prediction
    result = pipeline.predict(patient_data)

    if not result.get('success'):
        errors = result.get('errors') or [result.get('error', 'Unknown error')]
        return render_template('results.html', success=False, errors=errors, model_accuracy=pipeline.overall_model_accuracy)

    # Save the prediction to the user's prediction history.
    prediction_to_save = PredictionHistory(
        input_data=json.dumps(patient_data),
        prediction=result['prediction']['has_heart_disease'],
        probability=result['prediction']['probability'],
        user_id=current_user.id
    )
    db.session.add(prediction_to_save)
    # Personalized Recommendations Logic
    recommendations = []
    risk_level = result.get('risk_assessment', {}).get('risk_category', 'Medium Risk')

    if risk_level == 'High Risk':
        recommendations = [
            "Urgent: Schedule an appointment with a cardiologist as soon as possible.",
            "Your doctor may recommend an electrocardiogram (ECG) and a stress test.",
            "Strictly follow any prescribed medications and lifestyle changes.",
            "Avoid strenuous activities until you have consulted with a doctor."
        ]
    elif risk_level == 'Medium Risk':
        recommendations = [
            "Schedule a check-up with your primary care physician to discuss these results.",
            "Focus on a heart-healthy diet: reduce sodium, saturated fats, and processed foods.",
            "Increase regular, moderate physical activity to at least 150 minutes per week.",
            "Monitor your blood pressure and cholesterol levels regularly."
        ]
    else: # Low Risk
        recommendations = [
            "Excellent results! Continue to maintain your healthy lifestyle.",
            "Engage in regular physical activity and maintain a balanced diet.",
            "Annual check-ups with your doctor are still recommended for preventive care.",
            "Be mindful of your family's health history and discuss it with your doctor."
        ]

    return render_template('results.html', success=True, result=result, model_accuracy=pipeline.overall_model_accuracy, recommendations=recommendations, body_class='results-page')

@main_bp.route('/profile')
@login_required
def profile():
    user = current_user
    uploads = Document.query.filter_by(user_id=current_user.id).order_by(Document.upload_date.desc()).all()
    predictions = PredictionHistory.query.filter_by(user_id=current_user.id).order_by(PredictionHistory.timestamp.desc()).all()
    return render_template('profile.html', user=user, uploads=uploads, predictions=predictions, model_accuracy=pipeline.overall_model_accuracy)

@main_bp.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    if request.method == 'POST':
        phone_number = request.form.get('phone_number')
        current_user.phone_number = phone_number
        db.session.commit()
        flash('Profile updated successfully!', 'success')
    return redirect(url_for('main.profile'))

@main_bp.route('/clear_document/<int:document_id>', methods=['POST'])
@login_required
def clear_document(document_id):
    document = Document.query.get_or_404(document_id)
    if document.user_id != current_user.id:
        flash('You do not have permission to delete this document.')
        return redirect(url_for('main.profile'))
    
    # Optionally delete the file from the server as well
    if os.path.exists(document.filepath):
        os.remove(document.filepath)

    db.session.delete(document)
    db.session.commit()
    flash('Document removed successfully.')
    return redirect(url_for('main.profile'))

@main_bp.route('/clear_history', methods=['POST'])
@login_required
def clear_history():
    PredictionHistory.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    flash('Your prediction history has been cleared.')
    return redirect(url_for('main.dashboard'))

@main_bp.route('/clear_documents', methods=['POST'])
@login_required
def clear_documents():
    Document.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    flash('Your uploaded documents history has been cleared.')
    return redirect(url_for('main.dashboard'))

# The prediction route for manual input.
@main_bp.route('/predict', methods=['POST'])
@login_required
def predict():
    # Get the patient data from the form.
    patient_data = {
        'age': request.form.get('age'),
        'sex': request.form.get('sex'),
        'cp': request.form.get('cp'),
        'restBP': request.form.get('restBP'),
        'chol': request.form.get('chol'),
        'fbs': request.form.get('fbs'),
        'restECG': request.form.get('restECG'),
        'max_HR': request.form.get('max_HR'),
        'exang': request.form.get('exang'),
        'oldpeak': request.form.get('oldpeak'),
        'slope': request.form.get('slope'),
        'ca': request.form.get('ca'),
        'thal': request.form.get('thal')
    }

    # Make a prediction using the pipeline.
    mapped_patient_data = map_string_to_int(patient_data)
    result = pipeline.predict(mapped_patient_data)

    # If the prediction is not successful, render the results page with the errors.
    if not result.get('success'):
        errors = result.get('errors') or [result.get('error', 'Unknown error')]
        return render_template('results.html', success=False, errors=errors, model_accuracy=pipeline.overall_model_accuracy)

    # Save the prediction to the user's prediction history.
    prediction_to_save = PredictionHistory(
        input_data=json.dumps(patient_data),
        prediction=result['prediction']['has_heart_disease'],
        probability=result['prediction']['probability'],
        user_id=current_user.id
    )
    db.session.add(prediction_to_save)
    db.session.commit()

    # Render the results page with the prediction results.
    return render_template('results.html', success=True, result=result, model_accuracy=pipeline.overall_model_accuracy)