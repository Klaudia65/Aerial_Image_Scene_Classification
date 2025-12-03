

"""
#verify missing values in dataset
def verify_missing_values(df):
    missing_values = df.isnull().sum()
    total_cells = df.size
    total_missing = missing_values.sum()
    print(f"Total missing values: {total_missing}")
    print(f"Percentage of missing values: {total_missing / total_cells * 100:.2%}")
    return missing_values[missing_values > 0]

train_data = pd.read_csv('dataset/train copy.csv')
test_data = pd.read_csv('dataset/test.csv')
validation_data = pd.read_csv('dataset/validation.csv')

#verify_missing_values(train_data)
#ces fichiers sont générés à partir de l'organisation des dossiers, il n'y a pas de missing values dans test et validation

Chaque image a une étiquette : Le nom du dossier parent de l'image (ex. airplane) est l'étiquette de la classe. Il n'est donc pas possible d'avoir une image sans étiquette.

Action requise : Aucune vérification de colonne Label manquante n'est nécessaire, car cette information est intégrée dans la structure du répertoire.
"""
import pandas as pd
import os
import cv2
import numpy as np


def extract_basic_features(image_path):
    """Compute the mean and standard deviation of each RGB channel and the ExG index."""
    try:
        # Read the image using OpenCV (default BGR)
        img = cv2.imread(image_path)
        if img is None:
            return None # Handle cases where the file is not found
            
        # Convert to float for calculations
        img_float = img.astype(np.float32)

        # OpenCV reads BGR; extract channels accordingly
        B = img_float[:, :, 0]
        G = img_float[:, :, 1]
        R = img_float[:, :, 2]
        
        # 1. Per-band statistics (normalized to [0, 255])
        features = {
            'mean_R': np.mean(R) / 255.0,
            'std_R': np.std(R) / 255.0,
            'mean_G': np.mean(G) / 255.0,
            'std_G': np.std(G) / 255.0,
            'mean_B': np.mean(B) / 255.0,
            'std_B': np.std(B) / 255.0,
        }
        
        # 2. Excess Green Index (ExG):
        # ExG = 2*G - R - B
        # This index helps highlight vegetation in RGB images.
        ExG = 2 * G - R - B
        features['mean_ExG'] = np.mean(ExG)
        
        return features

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None
    

images_folder = 'dataset/images/'
labels = os.listdir(images_folder)
print(labels)
images_data = []

for label in labels:
    folder_path = os.path.join(images_folder, label)
    if not os.path.exists(folder_path):
        print(f"folder {folder_path} not found, skipping...")
        continue

    print(f"label: {label} in process...")

    for image_name in os.listdir(folder_path): #puts the image name in a list
        image_path = os.path.join(folder_path,image_name)
        features = extract_basic_features(image_path)
        if features:
            row_data = {
                "label": label,
                "image_name": image_name,
            }
            row_data.update(features)
            images_data.append(row_data) #in the list add a dict with image name, label and its features

df_features = pd.DataFrame(images_data)
print(df_features.head())
df_features.to_csv('dataset/image_features.csv', index=False)

print("Feature extraction completed and saved to image_features.csv")

