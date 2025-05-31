from pynput import keyboard
from datetime import datetime

def on_press(key):
    try:
        print(f"[{datetime.now()}] Pressed: {key.char}")
    except AttributeError:
        print(f"[{datetime.now()}] Special key: {key}")

# Start listener
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()
