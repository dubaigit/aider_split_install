import re
import pyperclip
import os
from datetime import datetime

def extract_warnings_and_errors(text):
    # Regular expressions for warnings and errors
    warning_pattern = r'(?i)^.*warning:.*$'
    error_pattern = r'(?i)^.*error:.*$'
    
    # Find all warnings and errors
    warnings = re.findall(warning_pattern, text, re.MULTILINE)
    errors = re.findall(error_pattern, text, re.MULTILINE)
    
    return warnings, errors

def save_logs_to_file(warnings, errors):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if warnings:
        with open(f'warning_log_{timestamp}.txt', 'w') as f:
            f.write('\n'.join(warnings))
        print(f"Warning log saved to warning_log_{timestamp}.txt")
    
    if errors:
        with open(f'error_log_{timestamp}.txt', 'w') as f:
            f.write('\n'.join(errors))
        print(f"Error log saved to error_log_{timestamp}.txt")

def extract_from_clipboard(save_to_file=False):
    # Get clipboard content
    clipboard_content = pyperclip.paste()
    
    # Extract warnings and errors
    warnings, errors = extract_warnings_and_errors(clipboard_content)
    
    # Combine and format the results
    result = "Warnings:\n" + "\n".join(warnings) + "\n\nErrors:\n" + "\n".join(errors)
    
    # Print the extracted content
    print(result)
    
    # Copy the extracted content back to clipboard
    pyperclip.copy(result)
    
    print("Warnings and errors have been extracted, printed, and copied to your clipboard.")
    
    if save_to_file:
        save_logs_to_file(warnings, errors)

if __name__ == "__main__":
    extract_from_clipboard(save_to_file=True)
