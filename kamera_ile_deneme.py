import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque

# Eğitilmiş modeli yükleyin
model = load_model('utkface_model.h5')

# Haar Cascade ile yüz algılama için dosya
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Yaş ve cinsiyet tahminlerini saklamak için birer kuyruk oluşturun
age_predictions = deque(maxlen=20)
gender_predictions = deque(maxlen=20)

# Yeni görüntüler üzerinde tahmin yapın
def prepare_image(face):
    img = cv2.resize(face, (64, 64))  # Modelin girdi boyutuna göre yeniden boyutlandır
    img = img / 255.0  # Normalizasyon
    img = np.expand_dims(img, axis=0)  # Modelin beklentisine uygun hale getirme
    return img

# Yaş aralığını belirleyen fonksiyon
def get_age_range(age):
    lower_bound = (age // 5) * 5
    upper_bound = lower_bound + 5
    return f'{lower_bound}-{upper_bound}'

# Kamerayı açın
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

# Çözünürlüğü ayarlayın
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Sabit yaş ve cinsiyet tahmini
fixed_age_text = ""
fixed_gender_text = ""
fixed = False
last_face_position = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Yüz algılama
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Yeni bir yüz algılandığında tahminleri sıfırla
        current_face_position = (x, y, w, h)
        if last_face_position is None or (abs(x - last_face_position[0]) > 50 or abs(y - last_face_position[1]) > 50):
            age_predictions.clear()
            gender_predictions.clear()
            fixed = False
            last_face_position = current_face_position

        if not fixed:
            face = frame[y:y+h, x:x+w]
            prepared_image = prepare_image(face)
            age_pred, gender_pred = model.predict(prepared_image)

            # Sonuçları yorumlayın
            predicted_age = int(age_pred[0][0])
            predicted_gender = 'Male' if gender_pred[0][0] > 0.5 else 'Female'

            # Tahminleri kuyruğa ekleyin
            age_predictions.append(predicted_age)
            gender_predictions.append(gender_pred[0][0])

            # 20 tahmin toplandığında sabit tahmini hesaplayın
            if len(age_predictions) == age_predictions.maxlen and len(gender_predictions) == gender_predictions.maxlen:
                avg_age = int(np.mean(age_predictions))
                age_range = get_age_range(avg_age)

                avg_gender = np.mean(gender_predictions)
                fixed_gender = 'Female' if avg_gender > 0.5 else 'Male'

                fixed_age_text = f'Age Range: {age_range}'
                fixed_gender_text = f'Gender: {fixed_gender}'
                fixed = True
        else:
            # Sabit tahmini göster
            text = f'{fixed_age_text}, {fixed_gender_text}'
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Çerçeveyi göster
    cv2.imshow('Age and Gender Prediction', frame)

    # 'q' tuşuna basıldığında çıkış yapın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı serbest bırakın ve tüm pencereleri kapatın
cap.release()
cv2.destroyAllWindows()
