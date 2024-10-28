# Aider Split Install

A multi-threaded tool designed to enhance AI development workflows using aider. This tool allows concurrent execution of aider tasks, improving development efficiency and productivity.

## Features

- **Concurrent Task Execution**: Run multiple aider tasks simultaneously using Python's asyncio
- **Configurable Concurrency**: Set maximum concurrent tasks via command line arguments
- **Smart File Handling**: Validates file existence before processing
- **Structured Task Management**: Organized approach to handling multiple development tasks
- **ANSI Color Support**: Enhanced terminal output with color-coded messages

## Installation

```bash
# Clone the repository
git clone [your-repo-url]
cd aider-split-install

# Make the script executable (Unix-like systems)
chmod +x aider_split_install.py

# Optional: Set up as a binary (requires --setup-bin flag)
python aider_split_install.py --setup-bin
```

## Usage

```bash
python aider_split_install.py [OPTIONS] file1.py file2.py ...

Options:
  --max-concurrent N    Set maximum number of concurrent tasks (default: 5)
  --setup-bin          Set up the script as a binary
  --help              Show help message
```

## Example

```bash
# Run with default settings
python aider_split_install.py app.py utils.py models.py

# Run with custom concurrency
python aider_split_install.py --max-concurrent 3 app.py utils.py models.py
```

## Task Format

Tasks should be structured in a specific format for optimal processing:

```
[task number]. [task title] ([filename])
- Target: [filename]
- Location:
  * Package: [package name]
  * Module: [module name]
  * Class: [class name]
  * Method: [method name(s)]
  * Section: [specific code section]
- Changes:
  * [detailed change 1]
  * [detailed change 2]
  * [detailed change 3]
- Dependencies:
  * Add: [new imports/requirements]
  * Remove: [old imports/code]
- Expected: [expected outcome]
```

## Progress Monitoring
- Real-time status updates for concurrent tasks
- Color-coded output for better visibility
- Clear task boundaries and progress indicators
- Success/failure status for each task

## Efficiency Tips

1. **Parallel Planning**
   - Group related changes by file
   - Ensure changes don't conflict
   - Distribute work evenly

2. **Resource Management**
   - Monitor system resources
   - Adjust concurrent tasks if needed
   - Each task has its own memory space

3. **Error Handling**
   - Isolated error handling per task
   - Failed tasks don't affect others
   - Easy to retry specific tasks

## Common Use Cases

1. **Full System Update**
```bash
aider_split --max-concurrent 5 src/*.py
```

2. **Selected Files**
```bash
aider_split --max-concurrent 5 auth.py models.py cache.py
```

3. **Test Files**
```bash
aider_split --max-concurrent 5 tests/*.py
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
