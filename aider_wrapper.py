import os
import sys
import argparse
import subprocess
import tempfile
import select
import git
import time
import re
import pyperclip

def read_file_content(filename):
    with open(filename, 'r') as file:
        return file.read()

def remove_urls(text):
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.sub('', text)

def create_message_content(instructions, file_contents):
    prompt = """"""

    existing_code = "\n\n".join([f"File: {filename}\n```\n{content}\n```" for filename, content in file_contents.items()])
    
    content = f"{prompt}\n\n<problem_description>\n{instructions}\n</problem_description>\n\n<existing_code>\n{existing_code}\n</existing_code>"
    return content

def enhance_user_experience():
    """
    Enhance user experience with a progress bar and better error handling.
    """
    from tqdm import tqdm
    import time

    for i in tqdm(range(10), desc="Preparing environment"):
        time.sleep(0.1)

def run_aider_command(aider_command, temp_message_file):
    print("\nExecuting aider command:")
    print(" ".join(aider_command))

    try:
        process = subprocess.Popen(aider_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

        while True:
            reads = [process.stdout.fileno(), process.stderr.fileno()]
            ret = select.select(reads, [], [])

            for fd in ret[0]:
                if fd == process.stdout.fileno():
                    read = process.stdout.readline()
                    sys.stdout.write(read)
                    sys.stdout.flush()
                if fd == process.stderr.fileno():
                    read = process.stderr.readline()
                    sys.stderr.write(read)
                    sys.stderr.flush()

            if process.poll() is not None:
                break

        rc = process.poll()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, aider_command)
    except subprocess.CalledProcessError as e:
        print(f"Error executing aider command: {e}", file=sys.stderr)
        print("The specified model may not be supported or there might be an issue with the aider configuration.")
        print("Please check your aider installation and ensure the model is correctly specified.")
        print("You can try running the aider command directly to see more detailed error messages.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        print("Please report this issue to the developers.")
        sys.exit(1)

def interactive_mode(args, file_contents):
    print("Entering interactive mode. Instructions:")
    print("1. Type your message and end with a line containing only '##END##'")
    print("2. Press Enter on an empty line to use clipboard content")
    print("3. Type 'exit' or 'quit' on a new line to end the session")

    while True:
        print("\nEnter your message (or press Enter to use clipboard):")
        message = []
        use_clipboard = False
        for line in sys.stdin:
            line = line.rstrip()
            if line.strip().lower() in ['exit', 'quit']:
                print("Exiting interactive mode.")
                return
            if line.strip() == "##END##":
                break
            if not line and not message:
                # Empty line at the start, use clipboard
                try:
                    clipboard_content = pyperclip.paste()
                    clipboard_content = remove_urls(clipboard_content)  # Remove URLs from clipboard content
                    print("Using clipboard content (after removing URLs):")
                    print(clipboard_content)
                    use_clipboard = True
                    break
                except Exception as e:
                    print(f"Error reading clipboard: {e}")
                    print("Please enter your message manually.")
                    continue
            message.append(line)

        if use_clipboard:
            message = clipboard_content
        else:
            message = "\n".join(message)
            message = remove_urls(message)

        if not message.strip():
            print("Empty message. Please try again.")
            continue

        # Create message content with the prompt and user's message or clipboard content
        full_message = create_message_content(message, file_contents)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, prefix='aider_wrapper_', suffix='.txt', dir='/tmp') as temp_message_file:
            temp_message_file.write(full_message)

        print(f"Temporary message file: {temp_message_file.name}")
        print("Prompt and clipboard/user content have been combined and sent to aider.")

        aider_command = [
            "aider",
            "--no-pretty",
            "--dark-mode",
            "--edit-format whole",
            "--yes",
            "--chat-mode", args.chat_mode,
            "--message-file", temp_message_file.name,
        ]

        if args.model:
            aider_command.extend(["--model", args.model])

        aider_command.extend(args.filenames)

        run_aider_command(aider_command, temp_message_file.name)

        print(f"\nTemporary file not deleted: {temp_message_file.name}")
        print("\nReady for next input. Press Enter to use clipboard or start typing.")

def main():
    parser = argparse.ArgumentParser(description="Wrapper for aider command")
    parser.add_argument("-i", "--instructions", help="File containing instructions")
    parser.add_argument("filenames", nargs='+', help="Filenames to process")
    parser.add_argument("--model", default="openai/o1-preview", help="Model to use for aider")
    parser.add_argument("--chat-mode", default="code", choices=["code", "ask"], help="Chat mode to use for aider")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode")

    args = parser.parse_args()

    print("Running aider wrapper with the following parameters:")
    print(f"Filenames: {', '.join(args.filenames)}")
    print(f"Instructions file: {args.instructions if args.instructions else 'Not provided'}")
    print(f"Model: {args.model}")
    print(f"Dark mode: Enabled")
    print(f"Chat mode: {args.chat_mode}")
    print(f"Interactive mode: {'Enabled' if args.interactive else 'Disabled'}")

    enhance_user_experience()

    # Add git commit before running aider
    try:
        repo = git.Repo(search_parent_directories=True)
        repo.git.add(update=True)
        repo.index.commit("Auto-commit before running aider")
        print("Git commit created successfully.")
    except git.exc.InvalidGitRepositoryError:
        print("Warning: Not a git repository. Skipping git commit.")
    except Exception as e:
        print(f"Error creating git commit: {e}")

    # Read file contents
    file_contents = {filename: read_file_content(filename) for filename in args.filenames}

    if args.interactive:
        interactive_mode(args, file_contents)
    else:
        if not args.instructions:
            print("Error: Instructions file is required when not in interactive mode.")
            sys.exit(1)

        # Read instruction file content
        instructions = read_file_content(args.instructions)

        # Create message content
        message_content = create_message_content(instructions, file_contents)

        # Write message content to a temporary file in /tmp
        with tempfile.NamedTemporaryFile(mode='w', delete=False, prefix='aider_wrapper_', suffix='.txt', dir='/tmp') as temp_message_file:
            temp_message_file.write(message_content)

        print(f"Temporary message file: {temp_message_file.name}")

        # Modify the aider command construction
        aider_command = [
            "aider",
            "--no-pretty",
            "--dark-mode",
            "--yes",
            "--chat-mode", args.chat_mode,
            "--message-file", temp_message_file.name,
        ]

        # Add the model argument separately
        if args.model:
            aider_command.extend(["--model", args.model])

        aider_command.extend(args.filenames)

        run_aider_command(aider_command, temp_message_file.name)

        print(f"\nTemporary file not deleted: {temp_message_file.name}")

if __name__ == "__main__":
    main()
