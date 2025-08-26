import pytesseract
from PIL import Image
import cv2
import re

def run_ocr_pipeline(image_path="reports/medical_report.png"):
    image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    extracted_text = pytesseract.image_to_string(thresh_image, lang='eng', config='--psm 6')
    print("\nExtracted Text\n")
    print(extracted_text)

    patterns = {
        'Age': re.compile(r'Age\s*:\s*(\d+)', re.IGNORECASE),
        'Sex': re.compile(r'Sex\s*:\s*(\w+)', re.IGNORECASE),
        'chestPain': re.compile(r'Chest pain\s*:\s*(.+)', re.IGNORECASE),
        'bloodPressure': re.compile(r'Blood pressure\s*:\s*(\d+)', re.IGNORECASE),
        'Cholesterol': re.compile(r'Cholesterol\s*:\s*(\d+)', re.IGNORECASE),
        'bloodSugar': re.compile(r'Blood sugar\s*:\s*(\d+)', re.IGNORECASE),
        'ECG': re.compile(r'ECG\s*:\s*(.+)', re.IGNORECASE),
        'maxHeartRate': re.compile(r'Max heart rate\s*:\s*(\d+)', re.IGNORECASE),
        'exerciseAngina': re.compile(r'Exercise angina\s*:\s*(\w+)', re.IGNORECASE),
        'stDepression': re.compile(r'ST depression\s*:\s*([\d.]+)', re.IGNORECASE),
        'stSlope': re.compile(r'ST slope\s*:\s*(\w+)', re.IGNORECASE),
        'Thal': re.compile(r'Thalassemia\s*:\s*(.+)', re.IGNORECASE)
    }

    ocr_data = {}
    for key, pattern in patterns.items():
        match = pattern.search(extracted_text)
        ocr_data[key] = match.group(1).strip() if match else None

    print("\nocr data\n")
    print(ocr_data)

    # Mapping categorical values to numeric codes
    mapping = {
        'Sex': {'Male': 1, 'Female': 0},
        'chestPain': {
            'Typical angina': 1,
            'Atypical angina': 2,
            'Non anginal pain': 3,  
            'Asymptomatic': 4
        },
        'exerciseAngina': {'Yes': 1, 'No': 0},
        'stSlope': {'Up': 1, 'Flat': 2, 'Down': 3},
        'ECG': {'Normal': 0, 'Abnormal': 1},
        'Thal': {'Normal': 1, 'Fixed defect': 2, 'Reversible defect': 3}
    }

    # Convert values to numeric
    numeric_data = {}
    int_fields = ['Age', 'bloodPressure', 'Cholesterol', 'bloodSugar', 'maxHeartRate']
    float_fields = ['stDepression']

    for key, value in ocr_data.items():
        if value is None:
            numeric_data[key] = -1
        elif key in mapping:
            numeric_data[key] = mapping[key].get(value, -1)
        else:
            if key in int_fields:
                numeric_data[key] = int(value)
            elif key in float_fields:
                numeric_data[key] = float(value)
            else:
                numeric_data[key] = float(value)

    print("\nNumeric Data\n")
    print(numeric_data)
    return numeric_data

if __name__ == "__main__":
    run_ocr_pipeline("reports/medical_report.png")
