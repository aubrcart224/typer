import cv2
import mediapipe as mp
from pynput import keyboard
from datetime import datetime
import threading
import time
import tkinter as tk
from collections import deque 
import math
import pytesseract
import numpy as np
import os
from keyboard_ocr import detect_keyboard


# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

stop_threads = False

# Keyboard detection state
keyboard_detected = False
detected_layout = {}

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
    global stop_threads, finger_history, keyboard_detected, detected_layout

    log("Webcam thread starting.")
    cap = cv2.VideoCapture(0)

    # Initial keyboard detection phase
    while not keyboard_detected and not stop_threads:
        success, frame = cap.read()
        if not success:
            log("Camera frame read failed.")
            break

        frame = cv2.flip(frame, 1)
        
        # Display instructions
        cv2.putText(frame, "Please show your keyboard clearly to the camera", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Try to detect keyboard
        layout = detect_keyboard(frame)
        
        # If we found enough keys, consider it a successful detection
        if len(layout) >= 20:  # Require at least 20 keys to be detected
            detected_layout = layout
            keyboard_detected = True
            log(f"✅ Keyboard layout detected with {len(layout)} keys")
            break
            
        cv2.imshow("Typing Coach - Keyboard Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_threads = True
            break

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

        # Draw detected keyboard layout
        if keyboard_detected:
            for char, pos in detected_layout.items():
                cv2.circle(frame, pos, 3, (0, 255, 0), -1)  # green dots
                cv2.putText(frame, char, (pos[0] + 5, pos[1] - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("Typing Coach - Hand Tracking", frame)

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
                        key_pixel_map = build_key_layout(calibrated_keys)
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

def build_key_layout(calibrated_keys):
    """
    Takes in the calibrated keys and returns a dict mapping key → pixel position.
    Now uses the detected layout instead of a static layout spec.
    """
    global detected_layout
    
    if not keyboard_detected or not detected_layout:
        log("❌ No keyboard layout detected!")
        return {}
        
    # Use the detected layout directly
    return detected_layout

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