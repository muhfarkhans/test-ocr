import cv2
from ultralytics import YOLO
import time
import easyocr

# load model YOLO kecil (nano biar ringan)
model = YOLO("yolov8n.pt")

reader = easyocr.Reader(['en'])

# buka kamera
cap = cv2.VideoCapture(0)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # jalankan prediksi
    results = model(frame)

    # gambar hasil ke frame
    annotated_frame = results[0].plot()

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        roi = frame[y1:y2, x1:x2]

        if roi.shape[0] > 20 and roi.shape[1] > 20:
            ocr_result = reader.readtext(roi)

            for (bbox, text, conf) in ocr_result:
                if conf > 0.5:
                    cv2.putText(
                        annotated_frame,
                        text,
                        (x1, y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

    # hitung FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # tampilkan FPS di layar
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # tampilkan
    cv2.imshow("YOLO + OCR", annotated_frame)

    # tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
