import cv2
import pytesseract

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Set resolusi kamera (optional, bisa bantu teks lebih jelas)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Crop area tengah (misal fokus ke label)
    height, width, _ = frame.shape
    x1, y1 = width//4, height//4
    x2, y2 = 3*width//4, 3*height//4
    crop_frame = frame[y1:y2, x1:x2]

    # Preprocessing
    gray = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # OCR
    text = pytesseract.image_to_string(thresh, lang="eng")
    text = text.strip()

    # Overlay hasil OCR ke frame asli (posisi di frame crop)
    cv2.putText(frame, text, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)

    # Tampilkan frame
    cv2.imshow("Live OCR Label", frame)
    cv2.imshow("Threshold View", thresh)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
