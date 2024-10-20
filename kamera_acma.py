import cv2

# Kamerayı açın (indeksleri deneyin)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

# Çözünürlüğü ayarlayın
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Çerçeveyi göster
    cv2.imshow('Camera Test', frame)

    # 'q' tuşuna basıldığında çıkış yapın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı serbest bırakın ve tüm pencereleri kapatın
cap.release()
cv2.destroyAllWindows()
