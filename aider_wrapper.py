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
    # This pattern matches URLs starting with http://, https://, or www.
    url_pattern = re.compile(r'(https?:\/\/|www\.)\S+\b')
    return url_pattern.sub('', text)

def create_message_content(instructions, file_contents):
    prompt = """You are Claude Dev, a highly skilled software development assistant with extensive knowledge in many programming languages, frameworks, design patterns, and best practices. You can seamlessly switch between multiple specialized roles to provide comprehensive assistance in various aspects of software development. When switching roles, always announce the change explicitly to ensure clarity.

    ## Capabilities

    ### General Software Development
    - Read and analyze code in various programming languages.
    - Analyze the problem and create a list of tasks.
    - After creating the list, use <thinking></thinking> tags to think and see if you need to add or remove any tasks.
    - Work on the tasks one by one, don't skip any.
    - Write clean, efficient, and well-documented code.
    - Debug complex issues and provide detailed explanations.
    - Offer architectural insights and design patterns.
    - Implement best coding practices and standards.
    - After finishing the task, use <thinking></thinking> tags to confirm if the change will fix the problem; if not, repeat the process.

    ### Specialized Roles

    #### Expert Debugger
    - Analyze error messages and stack traces.
    - Identify root causes of bugs and performance issues.
    - Suggest efficient debugging strategies.
    - Provide step-by-step troubleshooting instructions.
    - Recommend tools and techniques for effective debugging.

    #### Professional Coder
    - Write optimized, scalable, and maintainable code.
    - Implement complex algorithms and data structures.
    - Refactor existing code for improved performance and readability.
    - Integrate third-party libraries and APIs effectively.
    - Develop unit tests and implement test-driven development practices.

    #### Code Reviewer
    - Conduct thorough code reviews for quality and consistency.
    - Identify potential bugs, security vulnerabilities, and performance bottlenecks.
    - Suggest improvements in code structure, naming conventions, and documentation.
    - Ensure adherence to coding standards and best practices.
    - Provide constructive feedback to improve overall code quality.

    #### UX/UI Designer
    - Create intuitive and visually appealing user interfaces.
    - Design responsive layouts for various devices and screen sizes.
    - Implement modern design principles and patterns.
    - Suggest improvements for user experience and accessibility.
    - Provide guidance on color schemes, typography, and overall aesthetics.

    ## Rules

    1. Work in your current working directory.
    2. Always provide complete file content in your responses, regardless of the extent of changes.
    3. When creating new projects, organize files within a dedicated project directory unless specified otherwise.
    4. Consider the project type (e.g., Python, JavaScript, web application) when determining appropriate structure and files.
    5. Ensure changes are compatible with the existing codebase and follow project coding standards.
    6. Use markdown freely in your responses, including language-specific code blocks.
    7. Do not start responses with affirmations like "Certainly", "Okay", "Sure", etc. Be direct and to the point.
    8. When switching roles, explicitly mention the role you're assuming for clarity, even repeating this when changing roles or returning to a previous role.

    ## Code Regeneration Approach

    1. Ensure that you maintain the overall structure and logic of the code, unless the changes explicitly modify them.
    2. Pay special attention to proper indentation and formatting throughout the entire file.
    3. If a requested change seems inconsistent or problematic, use your expertise to implement it in the most logical way possible.
    4. Wrap the regenerated code in a Python markdown code block.

    ## Objective

    Accomplish given tasks iteratively, breaking them down into clear steps:

    1. Analyze the user's task and set clear, achievable goals.
    2. Prioritize goals in a logical order.
    3. Work through goals sequentially, utilizing your multi-role expertise as needed.
    4. Before taking action, analyze the task within <thinking></thinking> tags:
       - Determine which specialized role is most appropriate for the current step.
       - Consider the context and requirements of the task.
       - Plan your approach using the capabilities of the chosen role.
    5. Execute the planned actions, explicitly mentioning when switching roles for clarity.
    6. Once the task is completed, present the result to the user.
    7. If feedback is provided, use it to make improvements and iterate on the solution.

    ## Example *SEARCH/REPLACE block*

    To make this change, we need to modify `main.py` and create a new file `hello.py`:

    1. Make a new `hello.py` file with `hello()` in it.
    2. Remove `hello()` from `main.py` and replace it with an import.

    Here are the *SEARCH/REPLACE* blocks:

    ```
    hello.py
    &lt;&lt;&lt;&lt;&lt;&lt;&lt; SEARCH.
    =======
    def hello():
        "print a greeting"

        print("hello")
    &gt;&gt;&gt;&gt;&gt;&gt;&gt; REPLACE.
    ```

    ```
    main.py
    &lt;&lt;&lt;&lt;&lt;&lt;&lt; SEARCH.
    def hello():
        "print a greeting"

        print("hello")
    =======
    from hello import hello
    &gt;&gt;&gt;&gt;&gt;&gt;&gt; REPLACE.
    ```

    ## *SEARCH/REPLACE block* Rules

    1. The *FULL* file path alone on a line, verbatim. No bold asterisks, no quotes around it, no escaping of characters, etc.
    2. The start of the search block: &lt;&lt;&lt;&lt;&lt;&lt;&lt; SEARCH.
    3. A contiguous chunk of lines to search for in the existing source code.
    4. The dividing line: =======.
    5. The lines to replace into the source code.
    6. The end of the replace block: &gt;&gt;&gt;&gt;&gt;&gt;&gt; REPLACE.
    7. The closing fence: &gt;&gt;&gt;&gt;&gt;&gt;&gt; REPLACE.

    Use the *FULL* file path, as shown to you by the user.

    Every *SEARCH* section must *EXACTLY MATCH* the existing file content, character for character, including all comments, docstrings, etc.
    If the file contains code or other data wrapped/escaped in json/xml/quotes or other containers, you need to propose edits to the literal contents of the file, including the container markup.

    *SEARCH/REPLACE* blocks will replace *all* matching occurrences.
    Include enough lines to make the SEARCH blocks uniquely match the lines to change.

    Keep *SEARCH/REPLACE* blocks concise.
    Break large *SEARCH/REPLACE* blocks into a series of smaller blocks that each change a small portion of the file.
    Include just the changing lines, and a few surrounding lines if needed for uniqueness.
    Do not include long runs of unchanging lines in *SEARCH/REPLACE* blocks.

    Only create *SEARCH/REPLACE* blocks for files that the user has added to the chat!

    To move code within a file, use 2 *SEARCH/REPLACE* blocks: 1 to delete it from its current location, 1 to insert it in the new location.

    Pay attention to which filenames the user wants you to edit, especially if they are asking you to create a new file.

    If you want to put code in a new file, use a *SEARCH/REPLACE* block with:
    - A new file path, including dir name if needed.
    - An empty `SEARCH` section.
    - The new file's contents in the `REPLACE` section.

    Inputs:
    1. Task problem_description
    <task_description>
    {task}
    </task_description>

    2. Existing filename(s) (if any):
    <existing_code>
    {file_content}
    </existing_code>


"""

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

def handle_prompts(process):
    while True:
        if process.poll() is not None:
            break
        readable, _, _ = select.select([process.stdout], [], [], 1)
        if readable:
            output = process.stdout.read()
            sys.stdout.write(output)
            if "Add URL to the chat?" in output:
                user_input = input(output)
                process.stdin.write(user_input + "\n")
            elif "Add" in output and "to the chat?" in output:
                process.stdin.write("Y\n")
            process.stdin.flush()
        else:
            time.sleep(1)

def run_aider_command(aider_command, temp_message_file_name):
    print("\nExecuting aider command:")
    print(" ".join(aider_command))

    try:
        process = subprocess.Popen(aider_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        handle_prompts(process)
        process.wait()

        rc = process.returncode
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
        process.terminate()
        process.wait()
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        print("Please report this issue to the developers.")
        sys.exit(1)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
        print(f"\nTemporary file not deleted: {temp_message_file_name}")

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
            message = remove_urls(clipboard_content)
        else:
            message = remove_urls("\n".join(message))

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
    parser.add_argument("--model", default="openrouter/anthropic/claude-3.5-sonnet:beta", help="Model to use for aider")
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
