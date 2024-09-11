import pyperclip
import keyboard
import time

def append_to_prompt():
    # Read the content of prompt.md
    with open('prompt.md', 'r') as file:
        prompt_content = file.read()

    # Get the clipboard content
    clipboard_content = pyperclip.paste()

    # Combine prompt and clipboard content
    combined_content = prompt_content + clipboard_content

    # Copy the combined content back to clipboard
    pyperclip.copy(combined_content)

    print("Prompt + clipboard has been added to your clipboard")

print("Press Enter to append clipboard content to prompt.md")
keyboard.add_hotkey('enter', append_to_prompt)

# Keep the script running
keyboard.wait()
