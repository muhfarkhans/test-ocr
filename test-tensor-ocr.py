import cv2
import keras_ocr
import numpy as np

# ==========================
# 1️⃣ Inisialisasi pipeline
# ==========================
pipeline = keras_ocr.pipeline.Pipeline()

# ==========================
# 2️⃣ Buka kamera
# ==========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640/2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480/2)

frame_count = 0
ocr_results = []  # menyimpan hasil OCR terakhir
frame_skip = 5    # OCR setiap 5 frame

# ==========================
# 3️⃣ Loop kamera
# ==========================
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1

    # OCR setiap 'frame_skip' frame
    if frame_count % frame_skip == 0:
        ocr_results = pipeline.recognize([frame])

    # Gambar hasil OCR jika ada
    if ocr_results:
        for text, box in ocr_results[0]:
            pts = box.astype(int).reshape((-1,1,2))
            cv2.polylines(frame, [pts], True, (0,255,0), 2)
            x, y = pts[0][0]
            cv2.putText(frame, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Tampilkan frame
    cv2.imshow("OCR Live (Optimized)", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================
# 4️⃣ Tutup kamera & jendela
# ==========================
cap.release()
cv2.destroyAllWindows()
