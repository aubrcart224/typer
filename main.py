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
    # Left half
    # Top row
    'q': (-2.5, -1), 'w': (-1.5, -1), 'e': (-0.5, -1), 'r': (0.5, -1), 't': (1.5, -1),
    # Home row
    'a': (-2.5, 0), 's': (-1.5, 0), 'd': (-0.5, 0), 'f': (0.5, 0), 'g': (1.5, 0),
    # Bottom row
    'z': (-2.5, 1), 'x': (-1.5, 1), 'c': (-0.5, 1), 'v': (0.5, 1), 'b': (1.5, 1),

    # Right half (notice the gap between halves)
    # Top row
    'y': (3.5, -1), 'u': (4.5, -1), 'i': (5.5, -1), 'o': (6.5, -1), 'p': (7.5, -1),
    # Home row
    'h': (3.5, 0), 'j': (4.5, 0), 'k': (5.5, 0), 'l': (6.5, 0), ';': (7.5, 0),
    # Bottom row
    'n': (3.5, 1), 'm': (4.5, 1), ',': (5.5, 1), '.': (6.5, 1), '/': (7.5, 1),
}

expected_fingers = {
    # Left hand home row
    'a': 'Left Pinky', 's': 'Left Ring', 'd': 'Left Middle', 'f': 'Left Index', 'g': 'Left Index',
    # Right hand home row
    'h': 'Right Index', 'j': 'Right Index', 'k': 'Right Middle', 'l': 'Right Ring', ';': 'Right Pinky'
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

# Get webcam dimensions once at startup
cap = cv2.VideoCapture(0)
SCREEN_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
SCREEN_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

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
        # Draw origin and axes if layout is projected
        if not calibrating and 'f' in calibrated_keys:
            origin = calibrated_keys['f']['position']
            x_axis = normalize(subtract(origin, calibrated_keys['a']['position']))
            y_axis = rotate90(x_axis)

            scale = 100  # adjust length of axis arrows

            x_end = (int(origin[0] + x_axis[0] * scale), int(origin[1] + x_axis[1] * scale))
            y_end = (int(origin[0] + y_axis[0] * scale), int(origin[1] + y_axis[1] * scale))

            # Origin dot
            cv2.circle(frame, origin, 5, (0, 0, 255), -1)  # red dot

            # X-axis (Red)
            cv2.arrowedLine(frame, origin, x_end, (0, 0, 255), 2, tipLength=0.3)

            # Y-axis (Blue)
            cv2.arrowedLine(frame, origin, y_end, (255, 0, 0), 2, tipLength=0.3)

            # Optional: draw all key points
            for key, pos in key_pixel_map.items():
                cv2.circle(frame, pos, 3, (0, 255, 0), -1)  # green dots
                cv2.putText(frame, key, (pos[0] + 5, pos[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


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

                # Get screen center using global dimensions
                screen_center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)

                # Get expected finger for this key
                expected_finger = expected_fingers.get(expected_key)
                if expected_finger:
                    expected_hand, expected_finger_name = expected_finger.split()

                for hand in closest_snapshot["hands"]:
                    hand_label = hand["hand"]
                    # Only consider the correct hand
                    if hand_label != expected_hand:
                        continue

                    # Only consider the correct finger
                    finger_name = expected_finger_name.split()[-1]
                    if finger_name in hand["fingers"]:
                        pos = hand["fingers"][finger_name]
                        dist = euclidean_distance(pos, screen_center)
                        
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

                        # Project full layout
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
            if finger_history and char in key_pixel_map:
                closest_snapshot = min(finger_history, key=lambda s: abs(s["timestamp"] - keypress_time))
                target_pos = key_pixel_map[char]
                
                # Initialize tracking variables
                min_dist = float('inf')
                closest_finger = None
                
                # Get expected hand and finger if available
                expected_hand = None
                expected_finger = None
                if char in expected_fingers:
                    expected_hand, expected_finger = expected_fingers[char].split()
                
                for hand in closest_snapshot["hands"]:
                    for finger_name, pos in hand["fingers"].items():
                        # Skip thumb for regular keys
                        if finger_name == "Thumb":
                            continue
                            
                        dist = euclidean_distance(pos, target_pos)
                        
                        # Apply weight to expected finger
                        if (expected_hand and expected_finger and 
                            hand["hand"] == expected_hand and 
                            finger_name == expected_finger.split()[-1]):
                            dist *= 0.7  # Give 30% bonus to expected finger
                            
                        if dist < min_dist:
                            min_dist = dist
                            closest_finger = f"{hand['hand']} {finger_name}"
                
                if closest_finger:
                    log(f"Closest finger to key '{char}': {closest_finger}")

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

# ---- Finger Validation ----
    if char in expected_fingers:
        expected = expected_fingers[char]
        if closest_finger == expected:
            log(f"🟢 Correct finger: {closest_finger}")
        else:
            log(f"🔴 Wrong finger! Used {closest_finger if closest_finger else 'unknown finger'}, expected {expected}")
   

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

def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

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