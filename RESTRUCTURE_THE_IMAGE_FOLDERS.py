import pandas as pd
import os
import shutil

import pandas as pd

metadata_train_df = pd.read_csv('YOUR IMAGE FOLDER PATH')

metadata_train_df['benign_malignant'].fillna('benign', inplace=True)

metadata_train_df['benign_malignant'] = metadata_train_df['benign_malignant'].map({'benign': 0, 'malignant': 1})

metadata_train_df = metadata_train_df[['benign_malignant', 'isic_id']]  
metadata_train_df.to_csv('metadata_train_encoded.csv', index=False)


df = pd.read_csv('metadata_train_encoded.csv')

current_dir = r'YOUR_IMAGE_FOLDER_PATH'
malignant_dir = os.path.join(current_dir, 'malignant')
benign_dir = os.path.join(current_dir, 'benign')

os.makedirs(malignant_dir, exist_ok=True)
os.makedirs(benign_dir, exist_ok=True)

for index, row in df.iterrows():
  
    image_id = row['isic_id']
    diagnosis = row['benign_malignant']

    current_path = os.path.join(current_dir, f'{image_id}.JPG')
    new_path = os.path.join(malignant_dir if diagnosis == 1.0 else benign_dir, f'{image_id}.JPG')
    
    shutil.move(current_path, new_path)
