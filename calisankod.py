import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Veri setinin bulunduğu dizini belirtin
dataset_path = '/Users/izzetozguronder/Desktop/UTKFace'

# Veri setindeki resimleri ve etiketleri yükleyin
def load_utkface(dataset_path):
    images = []
    ages = []
    genders = []
    for root, dirs, files in os.walk(dataset_path):
        for filename in files:
            if filename.endswith(".jpg"):
                try:
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        age = int(parts[0])
                        gender = int(parts[1])
                        img = cv2.imread(os.path.join(root, filename))
                        if img is not None:
                            img = cv2.resize(img, (64, 64))
                            images.append(img)
                            ages.append(age)
                            genders.append(gender)
                    else:
                        print(f"Hatalı dosya adı {filename}")
                except ValueError as ve:
                    print(f"Hatalı dosya adı {filename}: {ve}")
                except Exception as e:
                    print(f'Hata {filename}: {e}')
    return np.array(images), np.array(ages), np.array(genders)

images, ages, genders = load_utkface(dataset_path)
print(f'Yüklenen görüntü sayısı: {len(images)}')
print(f'Yüklenen yaş etiketi sayısı: {len(ages)}')
print(f'Yüklenen cinsiyet etiketi sayısı: {len(genders)}')

# İlk birkaç görüntü ve etiketi göster
for i in range(5):
    plt.imshow(images[i])
    plt.title(f'Age: {ages[i]}, Gender: {"Male" if genders[i] == 0 else "Female"}')
    plt.show()

# Verileri eğitime hazırlayın
if len(images) > 0 and len(ages) == len(images) and len(genders) == len(images):
    images = images / 255.0
    X_train, X_test, y_train_ages, y_test_ages, y_train_genders, y_test_genders = train_test_split(
        images, ages, genders, test_size=0.2, random_state=42
    )
    print(f'Eğitim seti boyutu: {len(X_train)}')
    print(f'Test seti boyutu: {len(X_test)}')
else:
    print("Yeterli veri yüklenemedi veya veri sayısında uyumsuzluk var.")

# Giriş katmanı
input_layer = Input(shape=(64, 64, 3))

# Ortak katmanlar
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dropout(0.5)(x)

# Yaş tahmin katmanı
age_output = Dense(1, name='age_output')(x)

# Cinsiyet tahmin katmanı
gender_output = Dense(1, activation='sigmoid', name='gender_output')(x)

# Modeli tanımlayın
model = Model(inputs=input_layer, outputs=[age_output, gender_output])

# Modeli derleyin
model.compile(optimizer='adam',
              loss={'age_output': 'mean_squared_error', 'gender_output': 'binary_crossentropy'},
              metrics={'age_output': 'mae', 'gender_output': 'accuracy'})

# Modeli özetleyin
model.summary()

# Modeli eğitin
history = model.fit(X_train, {'age_output': y_train_ages, 'gender_output': y_train_genders},
                    validation_data=(X_test, {'age_output': y_test_ages, 'gender_output': y_test_genders}),
                    epochs=20, batch_size=32)

# Modeli değerlendirin
evaluation_results = model.evaluate(X_test, {'age_output': y_test_ages, 'gender_output': y_test_genders})

# Değerlendirme sonuçlarını yazdırın
for name, value in zip(model.metrics_names, evaluation_results):
    print(f'{name}: {value}')

# Modeli kaydedin
model.save('utkface_model.h5')

print("Model kaydedildi.")
