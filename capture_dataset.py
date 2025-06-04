import cv2
import os

save_dir = 'dataset/images/train'
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = len(os.listdir(save_dir))

print("[INFO] Press 'S' to save frame | Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    cv2.imshow("Capture Dataset", frame)

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
