import tkinter as tk

test_text = "hello world"
typed_text = ""
current_index = 0

def update_display():
    display_text = ""
    for i, char in enumerate(test_text):
        if i < len(typed_text):
            if typed_text[i] == char:
                display_text += f"[{char}]"
            else:
                display_text += f"({typed_text[i]})"
        elif i == current_index:
            display_text += f"_{char}_"
        else:
            display_text += f" {char} "
    label.config(text=display_text)

def on_key(event):
    global typed_text, current_index

    char = event.char.lower()
    if current_index < len(test_text):
        typed_text += char
        current_index += 1
        update_display()

    if current_index == len(test_text):
        result_label.config(text="✅ Done!")
        root.after(1500, root.destroy)  # close after short pause

# Set up the window
root = tk.Tk()
root.title("Mini MonkeyType 🐒")

label = tk.Label(root, text="", font=("Courier", 24), padx=20, pady=20)
label.pack()

result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack()

root.bind("<Key>", on_key)
update_display()

root.mainloop()
