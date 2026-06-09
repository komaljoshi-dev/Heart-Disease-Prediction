import pytesseract
from PIL import Image
import cv2
import re

def _calculate_age_group(age):
    """Calculates age group based on bins defined in src/features.py."""
    bins = [0, 30, 40, 50, 60, 100]
    labels = [0, 1, 2, 3, 4]
    for i in range(len(bins) - 1):
        if bins[i] <= age < bins[i+1]:
            return labels[i]
    return None # Handle age outside defined bins

def run_ocr_pipeline(image_path="reports/medical_report.png"):
    image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    extracted_text = pytesseract.image_to_string(thresh_image, lang='eng', config='--psm 6')
    print("\nExtracted Text\n")
    print(extracted_text)

    patterns = {
        'age': re.compile(r'Age\s*:\s*(\d+)', re.IGNORECASE),
        'sex': re.compile(r'Sex\s*:\s*(\w+)', re.IGNORECASE),
        'cp': re.compile(r'Chest pain\s*:\s*(.+)', re.IGNORECASE),
        'restBP': re.compile(r'Blood pressure\s*:\s*(\d+)', re.IGNORECASE),
        'chol': re.compile(r'Cholesterol\s*:\s*(\d+)', re.IGNORECASE),
        'fbs_raw': re.compile(r'Blood sugar\s*:\s*(\d+)', re.IGNORECASE), # Renamed to fbs_raw to apply logic later
        'restECG': re.compile(r'ECG\s*:\s*(.+)', re.IGNORECASE),
        'max_HR': re.compile(r'Max heart rate\s*:\s*(\d+)', re.IGNORECASE),
        'exang': re.compile(r'Exercise angina\s*:\s*(\w+)', re.IGNORECASE),
        'oldpeak': re.compile(r'ST depression\s*:\s*([\d.]+)', re.IGNORECASE),
        'slope': re.compile(r'ST slope\s*:\s*(.+)', re.IGNORECASE),
        'ca': re.compile(r'Number of major vessels\s*:\s*(\d+)', re.IGNORECASE), # Added ca pattern
        'thal': re.compile(r'Thalassemia\s*:\s*(.+)', re.IGNORECASE)
    }

    ocr_data = {}
    for key, pattern in patterns.items():
        match = pattern.search(extracted_text)
        ocr_data[key] = match.group(1).strip() if match else None

    print("\nocr data\n")
    print(ocr_data)

    # Mapping categorical values to numeric codes
    mapping = {
        'sex': {'male': 1, 'female': 0}, # Assuming 1=male, 0=female in CSV
        'cp': { # Align with CSV 1-4
            'typical angina': 1,
            'atypical angina': 2,
            'non anginal pain': 3,
            'asymptomatic': 4
        },
        'exang': {'yes': 1, 'no': 0}, # Consistent with CSV
        'slope': { # Align with CSV 1-3
            'upsloping': 1,
            'flat': 2,
            'downsloping': 3
        },
        'restECG': { # Align with CSV 0, 1, 2
            'normal': 0,
            'st-t wave abnormality': 1, # Assuming 1 for this in CSV
            'abnormal': 1, # Mapping 'abnormal' to ST-T wave abnormality
            'left ventricular hypertrophy': 2
        },
        'thal': { # Align with CSV 3, 6, 7
            'normal': 3,
            'fixed defect': 6,
            'reversible defect': 7
        }
    }

    # Convert values to numeric
    numeric_data = {}
    
    # Handle age and age_group
    age_val = None
    if ocr_data.get('age') is not None:
        try:
            age_val = float(ocr_data['age'])
            numeric_data['age'] = age_val
            numeric_data['age_group'] = _calculate_age_group(age_val)
        except ValueError:
            numeric_data['age'] = -1
            numeric_data['age_group'] = -1 # Indicate error or missing

    # Handle fbs (fasting blood sugar) with binary logic
    if ocr_data.get('fbs_raw') is not None:
        try:
            fbs_val = float(ocr_data['fbs_raw'])
            numeric_data['fbs'] = 1 if fbs_val > 120 else 0
        except ValueError:
            numeric_data['fbs'] = -1 # Indicate error or missing
    else:
        numeric_data['fbs'] = -1 # Indicate error or missing

    # Handle other fields
    for key, value in ocr_data.items():
        if key in ['age', 'fbs_raw']: # Already handled
            continue
        
        if value is None:
            if key == 'ca':
                numeric_data[key] = 0 # Default to 0 for ca if not found
            else:
                numeric_data[key] = -1 # Use -1 for other missing values
        elif key == 'slope': # Special handling for slope to accept digits or text
            if isinstance(value, str) and value.isdigit():
                numeric_data[key] = int(value)
            else:
                numeric_data[key] = mapping[key].get(value.lower(), -1)
        elif key in mapping:
            # Convert value to lowercase for case-insensitive matching
            numeric_data[key] = mapping[key].get(value.lower(), -1)
        else: # Direct numerical fields (restBP, chol, max_HR, oldpeak, ca)
            try:
                numeric_data[key] = float(value)
            except (ValueError, TypeError):
                numeric_data[key] = -1 # Indicate error or missing

    print("\nNumeric Data\n")
    print(numeric_data)
    return numeric_data

if __name__ == "__main__":
    run_ocr_pipeline("reports/medical_report.png")
