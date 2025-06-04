import cv2
import os

# Where to save images
save_dir = 'dataset/images/train'
os.makedirs(save_dir, exist_ok=True)
count = len(os.listdir(save_dir))

# Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Press 'S' to save frame | Press 'Q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)

    # Instructions
    cv2.putText(frame, f"Images saved: {count} | Press 'S' to save, 'Q' to quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show window
    try:
        cv2.imshow("Keyboard Dataset Capture", frame)
    except cv2.error as e:
        print("cv2.imshow() failed — likely missing GUI support in this session.")
        print(e)
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        path = os.path.join(save_dir, f'key_{count:03d}.jpg')
        cv2.imwrite(path, frame)
        print(f"[SAVED] {path}")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
