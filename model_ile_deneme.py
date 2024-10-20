import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Eğitilmiş modeli yükleyin
model = load_model('utkface_model.h5')

# Yeni görüntüler üzerinde tahmin yapın
def prepare_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

new_image_path = '/Users/izzetozguronder/Desktop/UTKFace/IMG_2317.JPG'  # Tahmin yapmak istediğiniz yeni görüntünün yolunu buraya ekleyin
new_image = prepare_image(new_image_path)

# Tahmin yapın
age_pred, gender_pred = model.predict(new_image)

# Sonuçları yorumlayın
predicted_age = age_pred[0][0]
predicted_gender = 'Female' if gender_pred[0][0] > 0.5 else 'Male'

print(f'Predicted Age: {predicted_age}')
print(f'Predicted Gender: {predicted_gender}')
