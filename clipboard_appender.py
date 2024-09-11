import pyperclip
import logging
import os

# Set up logging
logging.basicConfig(filename='clipboard_appender.log', level=logging.DEBUG)

def append_to_prompt():
    try:
        logging.debug("Function append_to_prompt started")
        
        # Check if prompt.md exists
        if not os.path.exists('prompt.md'):
            logging.error("prompt.md does not exist")
            print("Error: prompt.md does not exist")
            return

        # Read the content of prompt.md
        with open('prompt.md', 'r') as file:
            prompt_content = file.read()
        logging.debug("Read prompt.md successfully")

        # Get the clipboard content
        clipboard_content = pyperclip.paste()
        logging.debug("Got clipboard content")

        # Combine prompt and clipboard content
        combined_content = prompt_content + clipboard_content
        logging.debug("Combined contents")

        # Copy the combined content back to clipboard
        pyperclip.copy(combined_content)
        logging.debug("Copied combined content to clipboard")

        print("Prompt + clipboard has been added to your clipboard")
    except Exception as e:
        logging.exception("An error occurred in append_to_prompt")
        print(f"An error occurred: {str(e)}")

print("Press Enter to append clipboard content to prompt.md (or 'q' to quit)")

try:
    while True:
        user_input = input()
        if user_input.lower() == 'q':
            print("Exiting the program.")
            break
        append_to_prompt()
except KeyboardInterrupt:
    print("\nProgram interrupted. Exiting.")
except Exception as e:
    logging.exception("An error occurred in main script execution")
    print(f"An error occurred: {str(e)}")
