import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import os

IMG_SIZE = (48, 48)

def preprocess_image_from_path(img_path):
    try:
        img = Image.open(img_path).convert('L')  # grayscale
        img = img.resize(IMG_SIZE)
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))  # (1,48,48,1)
        return img_array
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


def save_to_csv(data_dict, csv_path='data/actuals.csv'):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame([data_dict])
    df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
