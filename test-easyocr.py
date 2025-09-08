import cv2
import easyocr
import numpy as np

# 1️⃣ Inisialisasi EasyOCR (bahasa Inggris saja)
reader = easyocr.Reader(['en'])

# 2️⃣ Inisialisasi kamera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # 3️⃣ Ambil frame dari kamera
    ret, frame = cap.read()
    if not ret:
        continue

    # 4️⃣ Crop area tengah (fokus ke label)
    h, w, _ = frame.shape
    x1, y1 = w//4, h//4
    x2, y2 = 3*w//4, 3*h//4
    crop_frame = frame[y1:y2, x1:x2]

    # 5️⃣ Preprocessing: grayscale
    gray = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)

    # 6️⃣ OCR dengan EasyOCR
    # results = reader.readtext(gray)
    results = reader.readtext(frame)
    for (bbox, text, prob) in results:
        if prob > 0.5:  # hanya ambil teks dengan confidence > 50%
            cv2.putText(frame, text, (int(bbox[0][0]), int(bbox[0][1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.polylines(frame, [np.array(bbox, dtype=np.int32)], True, (255,0,0), 2)

    
    # 7️⃣ Gabungkan semua teks yang terbaca
    ocr_text = " ".join([text for (_, text, _) in results])

    # 8️⃣ Overlay teks & bounding box di frame asli
    # cv2.putText(frame, ocr_text, (x1, y1-10),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    # cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)

    # 9️⃣ Tampilkan frame
    cv2.imshow("EasyOCR Live", frame)

    # 10️⃣ Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
