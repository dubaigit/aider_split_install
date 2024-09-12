import re
import pyperclip

def extract_warnings_and_errors(text):
    # Regular expressions for warnings and errors
    warning_pattern = r'(?i)warning:.*'
    error_pattern = r'(?i)error:.*'
    
    # Find all warnings and errors
    warnings = re.findall(warning_pattern, text, re.MULTILINE)
    errors = re.findall(error_pattern, text, re.MULTILINE)
    
    # Combine and format the results
    result = "Warnings:\n" + "\n".join(warnings) + "\n\nErrors:\n" + "\n".join(errors)
    return result

def extract_from_clipboard():
    # Get clipboard content
    clipboard_content = pyperclip.paste()
    
    # Extract warnings and errors
    extracted_content = extract_warnings_and_errors(clipboard_content)
    
    # Print the extracted content
    print(extracted_content)
    
    # Copy the extracted content back to clipboard
    pyperclip.copy(extracted_content)
    
    print("Warnings and errors have been extracted, printed, and copied to your clipboard.")

if __name__ == "__main__":
    extract_from_clipboard()
