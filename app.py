from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import os
from utils import preprocess_image_from_path, save_to_csv
from datetime import datetime
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'model/my_age_gender_model.keras'
CSV_PATH = 'data/actuals.csv'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('data', exist_ok=True)

model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")


# /predict

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    img_array = preprocess_image_from_path(file_path)
    if img_array is None:
        return jsonify({'error': 'Invalid image'}), 400

    # Model prediction
    gender_pred, age_pred = model.predict(img_array, verbose=0)

    pred_gender = "Male" if gender_pred[0][0] < 0.5 else "Female"
    pred_age = int(np.round(age_pred[0][0] * 100))

    save_to_csv({
        'filename': filename,
        'uploaded_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'pred_gender': pred_gender,
        'pred_age': pred_age,
        'actual_gender': None,
        'actual_age': None
    }, CSV_PATH)

    return jsonify({
        'filename': filename,
        'predicted_gender': pred_gender,
        'predicted_age': pred_age,
        'file_path': file_path
    })



# /update_actual

@app.route('/update_actual', methods=['POST'])
def update_actual():
    data = request.get_json()
    filename = data.get('filename')
    actual_age = data.get('actual_age')
    actual_gender = data.get('actual_gender')

    if not all([filename, actual_age, actual_gender]):
        return jsonify({'error': 'Missing fields'}), 400

    if not os.path.exists(CSV_PATH):
        return jsonify({'error': 'No predictions found'}), 400

    df = pd.read_csv(CSV_PATH)
    if filename not in df['filename'].values:
        return jsonify({'error': 'Filename not found'}), 400

    df.loc[df['filename'] == filename, 'actual_age'] = actual_age
    df.loc[df['filename'] == filename, 'actual_gender'] = actual_gender
    df.to_csv(CSV_PATH, index=False)

    return jsonify({'message': 'Actual values updated successfully'})



# /accuracy_data

@app.route('/accuracy_data', methods=['GET'])
def accuracy_data():
    if not os.path.exists(CSV_PATH):
        return jsonify({'message': 'No data yet'})

    df = pd.read_csv(CSV_PATH)

    # Convert to numeric 
    df['pred_age'] = pd.to_numeric(df['pred_age'], errors='coerce')
    df['actual_age'] = pd.to_numeric(df['actual_age'], errors='coerce')

    df = df.dropna(subset=['actual_age', 'actual_gender'])

    if len(df) == 0:
        return jsonify({'message': 'No actuals recorded yet'})

    # ±5 years age accuracy
    age_diff = np.abs(df['pred_age'] - df['actual_age'])
    age_accuracy = np.mean(age_diff <= 5)

    # Gender accuracy
    gender_acc = np.mean(
        df['pred_gender'].str.lower() ==
        df['actual_gender'].str.lower()
    )

    return jsonify({
        'age_accuracy': round(age_accuracy, 2),
        'gender_accuracy': round(gender_acc, 2),
        'total_samples': len(df)
    })



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

