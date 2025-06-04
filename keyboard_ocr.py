from ultralytics import YOLO
import easyocr
import cv2

# Load YOLOv8 model and EasyOCR
model = YOLO("models/yolov8_keycap.pt")
reader = easyocr.Reader(['en'])

def detect_keyboard(frame, visualize=True):
    """
    Detect keyboard keys using YOLO and read their labels using EasyOCR.
    Returns a dictionary mapping characters to (x, y) screen positions.
    """
    layout = {}

    results = model(frame, verbose=False)[0]
    for box in results.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box[:4])
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Crop the key region
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        # Run OCR
        ocr_result = reader.readtext(cropped)
        if ocr_result:
            text = ocr_result[0][1].strip().lower()

            if len(text) == 1 and (text.isalnum() or text in [',', '.', ';', '/']):
                layout[text] = (center_x, center_y)

                if visualize:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, text, (center_x, center_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return layout
