import cv2
import os

save_dir = 'dataset/images/train'
os.makedirs(save_dir, exist_ok=True)
count = len(os.listdir(save_dir))

cap = cv2.VideoCapture(0)

print("[INFO] Click on the image to save a frame | Press 'Q' to quit.")

# Mouse callback function
def save_frame_on_click(event, x, y, flags, param):
    global count
    if event == cv2.EVENT_LBUTTONDOWN:
        filename = f'key_{count:03d}.jpg'
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, param)
        print(f"[SAVED] {filepath}")
        count += 1

cv2.namedWindow("Keyboard Dataset Capture")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)
    display = frame.copy()

    cv2.putText(display, f"Images saved: {count} | Click to save | Press 'Q' to quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Set the current frame as the param for mouse callback
    cv2.setMouseCallback("Keyboard Dataset Capture", save_frame_on_click, frame)

    cv2.imshow("Keyboard Dataset Capture", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
