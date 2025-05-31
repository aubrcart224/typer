import cv2
import mediapipe as mp
from pynput import keyboard
from datetime import datetime
import threading
import time
import tkinter as tk
from collections import deque 
import math

stop_threads = False

# Calibration state
calibrated_keys = {}
calibration_keys = ['a', 's', 'd', 'f', 'j', 'k', 'l', ';']
current_cal_key_index = 0
calibrating = True

# calibrate maths 
def normalize(v):
    mag = math.sqrt(v[0]**2 + v[1]**2)
    return (v[0]/mag, v[1]/mag) if mag != 0 else (0, 0)

def subtract(p1, p2):
    return (p1[0] - p2[0], p1[1] - p2[1])

def rotate90(v):
    return (-v[1], v[0])

layout_spec = {
    # Home row
    'a': (-3, 0), 's': (-2, 0), 'd': (-1, 0), 'f': (0, 0),
    'j': (1, 0), 'k': (2, 0), 'l': (3, 0), ';': (4, 0),

    # Top row
    'q': (-3, -1), 'w': (-2, -1), 'e': (-1, -1), 'r': (0, -1),
    'u': (1, -1), 'i': (2, -1), 'o': (3, -1), 'p': (4, -1),

    # Bottom row
    'z': (-3, 1), 'x': (-2, 1), 'c': (-1, 1), 'v': (0, 1),
    'm': (1, 1), ',': (2, 1), '.': (3, 1), '/': (4, 1),
}


test_text = "hello world"
typed_text = ""
current_index = 0

# Rolling buffer of finger positions
finger_history = deque(maxlen=200)  # stores last ~10s


# Logging helper
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# ---- Webcam Thread ----
def hand_tracking():

    global stop_threads, finger_history

    log("Webcam thread starting.")
    cap = cv2.VideoCapture(0)

    while cap.isOpened() and not stop_threads:
        success, frame = cap.read()
        if not success:
            log("Camera frame read failed.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        timestamp = time.time()
        frame_data = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[hand_idx].classification[0].label  # 'Left' or 'Right'

                # Get relevant fingertip landmarks
                fingers = {
                    "Thumb": hand_landmarks.landmark[4],
                    "Index": hand_landmarks.landmark[8],
                    "Middle": hand_landmarks.landmark[12],
                    "Ring": hand_landmarks.landmark[16],
                    "Pinky": hand_landmarks.landmark[20],
                }

                # Convert normalized coords to pixels
                finger_positions = {
                    name: (int(finger.x * frame.shape[1]), int(finger.y * frame.shape[0]))
                    for name, finger in fingers.items()
                }

                frame_data.append({
                    "timestamp": timestamp,
                    "hand": handedness,
                    "fingers": finger_positions
                })

                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Save to rolling buffer
            finger_history.append({
                "timestamp": timestamp,
                "hands": frame_data
            })

        cv2.imshow("Typing Coach - Hand Tracking", frame)

        # ---- Hand Landmarks ----
        

        # ---- Stop Process ----
        if cv2.waitKey(1) & 0xFF == ord('q'):
            log("Q pressed in webcam window.")
            stop_threads = True
            break

    cap.release()
    cv2.destroyAllWindows()
    log("Webcam thread exiting.")

# ---- Keylogger Thread ----
def on_press(key):
    global stop_threads, typed_text, current_index, finger_history 
    global calibrating, calibrated_keys, current_cal_key_index


    try:
        char = key.char.lower()
    except AttributeError:
        # Handle special keys
        if key == keyboard.Key.esc:
            log("🔴 Escape pressed — exiting.")
            stop_threads = True
            return False
        log(f"Special key pressed: {key}")
        return  # Ignore non-character keys

    keypress_time = time.time()

# === CALIBRATION MODE ===
    if calibrating:
        expected_key = calibration_keys[current_cal_key_index]
        if char == expected_key:
            if finger_history:
                closest_snapshot = min(finger_history, key=lambda s: abs(s["timestamp"] - keypress_time))

                best_match = None
                min_distance = float('inf')

                for hand in closest_snapshot["hands"]:
                    hand_label = hand["hand"]
                    for finger_name, pos in hand["fingers"].items():
                        # Use simple vertical distance from screen center (or could use fixed Y target)
                        dx = pos[0]
                        dy = pos[1]
                        dist = abs(dy)  # You could also compare to previous key position if you'd like

                        if dist < min_distance:
                            min_distance = dist
                            best_match = {
                                'hand': hand_label,
                                'finger': finger_name,
                                'position': pos
                            }

                if best_match:
                    calibrated_keys[char] = best_match
                    log(f"✅ Calibrated '{char}' with {best_match['hand']} {best_match['finger']} at {best_match['position']}")
                    current_cal_key_index += 1

                    if current_cal_key_index >= len(calibration_keys):
                        log("✅ Calibration complete. Begin typing test.")
                        calibrating = False

                         # 🔽 ADD THIS: project full layout
                        global key_pixel_map  # make it accessible in other functions
                        key_pixel_map = build_key_layout(calibrated_keys, layout_spec)
                        log("🧭 Full layout projected.")
                    
                else:
                    log("❌ Couldn't find any finger position. Try again.")
        else:
            log(f"Waiting for '{expected_key}' (not '{char}')")
        return


        


    try:
        # Get character from keypress
        char = key.char.lower()
        keypress_time = time.time()

        # Match against test text
        if current_index < len(test_text):
            expected = test_text[current_index]
            correct = (char == expected)

            log(f"Typed: '{char}' | Expected: '{expected}' | {'✅' if correct else '❌'}")

            # Try to find the closest finger data frame
            if finger_history:
                closest_snapshot = min(finger_history, key=lambda s: abs(s["timestamp"] - keypress_time))
                log(f"Closest finger snapshot at {closest_snapshot['timestamp']:.2f} for key '{char}'")

                for hand in closest_snapshot["hands"]:
                    hand_label = hand["hand"]
                    for finger_name, pos in hand["fingers"].items():
                        log(f"{hand_label} {finger_name} at {pos}")

            typed_text += char
            current_index += 1

        if current_index >= len(test_text):
            log("✅ Typing test complete.")
            stop_threads = True
            return False

    except AttributeError:
        if key == keyboard.Key.esc:
            log("🔴 Escape pressed — exiting.")
            stop_threads = True
            return False
        log(f"Special key pressed: {key}")

def key_logger():
    
    log("Keylogger thread starting.")
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
    log("Keylogger thread exiting.")


def build_key_layout(calibrated_keys, layout_spec):
    """
    Takes in the calibrated keys (home row positions) and a logical keyboard layout
    Returns a dict mapping key → projected pixel position
    """

    # Use 'f' as the origin (or whatever is most central)
    origin_key = 'f'
    origin_data = calibrated_keys[origin_key]
    origin = origin_data['position']

    # Use direction from 'a' to 'f' to define x-axis
    x_axis = normalize(subtract(origin, calibrated_keys['a']['position']))
    y_axis = rotate90(x_axis)  # 90° rotation for vertical axis

    log(f"📐 x_axis: {x_axis}, y_axis: {y_axis}")

    key_pixel_map = {}

    for key, (dx, dy) in layout_spec.items():
        # Project: origin + dx * x_axis + dy * y_axis
        px = origin[0] + dx * x_axis[0] * 40 + dy * y_axis[0] * 40
        py = origin[1] + dx * x_axis[1] * 40 + dy * y_axis[1] * 40
        key_pixel_map[key] = (int(px), int(py))

    return key_pixel_map


# ---- Main Entry ----
if __name__ == "__main__":
    log("Program starting...")

    t1 = threading.Thread(target=hand_tracking)
    t2 = threading.Thread(target=key_logger)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    log("✅ Program exited cleanly.")

    if calibrating:
     log("🛠️ Starting calibration. Press: " + ' '.join(calibration_keys))