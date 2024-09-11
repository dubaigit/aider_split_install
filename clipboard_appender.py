import pyperclip
import logging
import os

# Set up logging
logging.basicConfig(filename='clipboard_appender.log', level=logging.DEBUG)

def append_to_prompt(user_input=''):
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

        # Clear the clipboard
        pyperclip.copy('')
        logging.debug("Cleared clipboard")

        # Combine user input (if any) and clipboard content
        new_content = f"##problem_or_log_issue#\n{user_input}\n{clipboard_content}\n"
        logging.debug("Created new content with user input and clipboard content")

        # Combine prompt and new content
        combined_content = prompt_content + new_content
        logging.debug("Combined contents")

        # Write the combined content to prompt.md
        with open('prompt.md', 'w') as file:
            file.write(combined_content)
        logging.debug("Wrote combined content to prompt.md")

        # Copy the combined content to clipboard
        pyperclip.copy(combined_content)
        logging.debug("Copied combined content to clipboard")

        print("Content has been appended to prompt.md and copied to your clipboard")
    except Exception as e:
        logging.exception("An error occurred in append_to_prompt")
        print(f"An error occurred: {str(e)}")

print("Press Enter to append clipboard content to prompt.md")
print("Or type your text and press Enter to append both your text and clipboard content")
print("Type 'q' to quit")

try:
    while True:
        user_input = input()
        if user_input.lower() == 'q':
            print("Exiting the program.")
            break
        append_to_prompt(user_input)
except KeyboardInterrupt:
    print("\nProgram interrupted. Exiting.")
except Exception as e:
    logging.exception("An error occurred in main script execution")
    print(f"An error occurred: {str(e)}")
