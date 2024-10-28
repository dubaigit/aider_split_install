#!/usr/bin/env python3

import os
import re
import subprocess
import sys
import shutil
from typing import List, Optional
from pathlib import Path
import logging
from datetime import datetime
import tempfile
import asyncio
from dataclasses import dataclass

# ANSI color codes for terminal output
INTRO_MESSAGE = """You are Claude Dev, a highly skilled software development assistant with extensive knowledge in many programming languages, frameworks, design patterns, and best practices.


{{task}}"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ANSIColors:
    """ANSI color codes for terminal output."""
    RED = "31"
    GREEN = "32"
    YELLOW = "33"
    BLUE = "34"
    CYAN = "36"
    RESET = "0"

    @staticmethod
    def colorize(text: str, color_code: str) -> str:
        """Applies ANSI color codes to text."""
        return f"\033[{color_code}m{text}\033[{ANSIColors.RESET}m"

@dataclass
class CodeLocation:
    """Represents a specific location in code (class/function/method)."""
    filename: str
    target: Optional[str] = None  # class/function/method name

@dataclass
class Task:
    """Represents a single task from the instructions file."""
    number: str
    content: str
    temp_file: Optional[str] = None
    locations: List[CodeLocation] = None

    def __post_init__(self):
        self.locations = self._parse_locations()

    def _parse_locations(self) -> List[CodeLocation]:
        """Parse code locations from task content."""
        locations = []
        lines = self.content.split('\n')
        current_file = None
        
        for line in lines:
            if line.startswith('- Target:'):
                current_file = line.split(':')[1].strip()
            elif line.strip().startswith('* Class:') or line.strip().startswith('* Method:') or line.strip().startswith('* Function:'):
                target = line.split(':')[1].strip()
                if current_file:
                    locations.append(CodeLocation(current_file, target))
        
        return locations or [CodeLocation(current_file)] if current_file else []

class InstructionValidator:
    """Handles validation and processing of instruction files."""
    
    @staticmethod
    def validate_format(content: str) -> bool:
        """Checks if the instruction file content is in the correct format."""
        return bool(re.search(r'^\d+\.\s', content, re.MULTILINE))
    
    @staticmethod
    def get_instruction_tasks(filename: str) -> List[Task]:
        """Reads and parses instruction tasks from file."""
        try:
            with open(filename, 'r') as f:
                content = f.read()
            if not InstructionValidator.validate_format(content):
                raise ValueError("Invalid instruction format")
            
            tasks = []
            for line in content.splitlines():
                match = re.match(r'^(\d+)\.\s(.*)', line)
                if match:
                    tasks.append(Task(
                        number=match.group(1),
                        content=match.group(2)
                    ))
            
            if not tasks:
                raise ValueError("No tasks found in instruction file")
                
            return tasks
        except FileNotFoundError:
            logger.error(f"Instruction file '{filename}' not found")
            raise
        except ValueError as e:
            logger.error(f"Invalid instruction format: {e}")
            raise

class BinSetup:
    """Handles setup of the script in /usr/local/bin/."""
    
    @staticmethod
    def create_bash_wrapper(script_path: Path) -> str:
        """Creates the content for the bash wrapper script."""
        return f'''#!/bin/bash

# Path to the Python script
SCRIPT_PATH="{script_path}"

# Check if Python 3 is available
if command -v python3 &> /dev/null; then
    python3 "$SCRIPT_PATH" "$@"
elif command -v python &> /dev/null; then
    python "$SCRIPT_PATH" "$@"
else
    echo "Error: Python 3 is required but not found"
    exit 1
fi
'''

    @staticmethod
    def setup(script_path: str) -> None:
        """Sets up the script in /usr/local/bin/."""
        script_path = Path(script_path).resolve()
        lib_dir = Path("/usr/local/lib/aider_split")
        bin_dir = Path("/usr/local/bin")
        script_dest = lib_dir / "aider_split.py"
        bin_dest = bin_dir / "aider_split"
        
        # Check if script exists first
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        # Check permissions before attempting sudo
        needs_sudo = not (
            os.access('/usr/local/bin', os.W_OK) and 
            os.access('/usr/local/lib', os.W_OK)
        )
        
        if needs_sudo:
            logger.info(ANSIColors.colorize(
                "🔑 Root permissions required. Re-running with sudo...", 
                ANSIColors.YELLOW
            ))
            try:
                subprocess.check_call(['sudo', 'python3', str(script_path), '--setup-bin'])
                return
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to gain root permissions: {e}")
                raise

        try:
            # Create directory if it doesn't exist
            os.makedirs(lib_dir, exist_ok=True)
            
            # Copy Python script to lib directory
            shutil.copy(script_path, script_dest)
            script_dest.chmod(0o644)  # rw-r--r--
            
            # Create bash wrapper script
            wrapper_content = BinSetup.create_bash_wrapper(script_dest)
            with open(bin_dest, 'w') as f:
                f.write(wrapper_content)
            bin_dest.chmod(0o755)  # rwxr-xr-x
            
            logger.info(ANSIColors.colorize(
                "✅ Installation completed:\n"
                f"  - Python script installed to: {script_dest}\n"
                f"  - Executable wrapper installed to: {bin_dest}\n"
                "You can now run 'aider_split' globally!", 
                ANSIColors.GREEN
            ))
        except Exception as e:
            logger.error(f"Failed to install script: {e}")
            raise

class AsyncAiderRunner:
    """Handles running multiple aider commands concurrently."""
    
    def __init__(self, intro_message: str, max_concurrent: int = 5):
        self.intro_message = intro_message
        self.instruction_file = 'fix_instructions.txt'
        self.max_concurrent = max_concurrent
        self.temp_dir = tempfile.mkdtemp(prefix='aider_split_')
        self.location_locks = {}  # Dictionary to store locks for each code location
    
    def _get_location_key(self, location: CodeLocation) -> str:
        """Generate a unique key for a code location."""
        return f"{location.filename}:{location.target if location.target else ''}"
    
    async def _acquire_location_locks(self, task: Task):
        """Acquire locks for all locations in a task."""
        locks = []
        for location in task.locations:
            key = self._get_location_key(location)
            if key not in self.location_locks:
                self.location_locks[key] = asyncio.Lock()
            locks.append(self.location_locks[key])
        
        # Sort locks to prevent deadlocks
        locks.sort(key=str)
        for lock in locks:
            await lock.acquire()
        return locks
    
    def _release_location_locks(self, locks: List[asyncio.Lock]):
        """Release all acquired locks."""
        for lock in locks:
            lock.release()

    async def _run_aider_task(self, task: Task, filenames: List[str]) -> None:
        """Runs a single aider task asynchronously."""
        logger.info(ANSIColors.colorize(
            f"\n\n==================== 🟢 STARTING TASK {task.number} ====================\n",
            ANSIColors.CYAN
        ))
        
        # Acquire locks for all locations in this task
        location_locks = await self._acquire_location_locks(task)
        
        try:
            prompt_file = self._create_temp_prompt_file(task)
            
            try:
                process = await asyncio.create_subprocess_exec(
                    "aider",
                    "--model", "anthropic/claude-3-5-sonnet-20241022",
                    "--no-pretty",
                    "--message-file", prompt_file,
                    *filenames,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    logger.info(stdout.decode())
                    self._update_instruction_file(task.number)
                    logger.info(ANSIColors.colorize(
                        f"\n==================== 🔵 FINISHED TASK {task.number} ====================\n",
                        ANSIColors.BLUE
                    ))
                else:
                    logger.error(ANSIColors.colorize(
                        f"Error in task {task.number}:\n{stderr.decode()}",
                        ANSIColors.RED
                    ))
            finally:
                self._cleanup_temp_file(prompt_file)
        finally:
            # Release all locks
            self._release_location_locks(location_locks)

    def _create_temp_prompt_file(self, task: Task) -> str:
        """Creates a temporary file containing the prompt."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir=self.temp_dir) as tmp:
                tmp.write(task.content)
                return tmp.name
        except IOError as e:
            logger.error(f"Failed to create temporary prompt file: {e}")
            raise

    def _cleanup_temp_file(self, filepath: str) -> None:
        """Safely removes the temporary prompt file."""
        try:
            os.unlink(filepath)
        except OSError as e:
            logger.warning(f"Failed to remove temporary file {filepath}: {e}")

    def _update_instruction_file(self, task_num: str) -> None:
        """Updates the instruction file by removing completed tasks."""
        try:
            subprocess.run([
                "sed", "-i.bak", f"/^{task_num}\\./,/^[0-9]+\\./d", self.instruction_file
            ], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to update instruction file: {e}")
            raise

    async def run(self, filenames: List[str]) -> None:
        """Runs multiple aider tasks concurrently."""
        try:
            # Validate instruction file exists
            if not Path(self.instruction_file).exists():
                raise FileNotFoundError(
                    f"Instruction file '{self.instruction_file}' not found. "
                    "Please create it before running the script."
                )
            
            # Get tasks directly as Task objects
            task_objects = InstructionValidator.get_instruction_tasks(self.instruction_file)
            
            if not task_objects:
                logger.warning("No tasks found in instruction file.")
                return
            
            # Create semaphore to limit concurrent tasks
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def run_with_semaphore(task):
                async with semaphore:
                    await self._run_aider_task(task, filenames)
            
            # Run tasks concurrently with semaphore
            await asyncio.gather(
                *(run_with_semaphore(task) for task in task_objects)
            )
            
            logger.info(ANSIColors.colorize("🎉 All tasks completed.", ANSIColors.GREEN))
        except Exception as e:
            logger.error(f"An error occurred while running tasks: {e}")
            raise

def main():
    """Main entry point of the script."""
    if "--help" in sys.argv or "-h" in sys.argv:
        print(ANSIColors.colorize("""
Usage: aider_split [OPTIONS] FILENAMES...

A tool to run aider with multiple tasks from an instruction file.

Options:
    --setup-bin      Install the script globally in /usr/local/bin
    --help, -h      Show this help message

Arguments:
    FILENAMES       One or more files to process with aider

## Creating Python-Specific Instructions File (fix_instructions.txt)

### Format Requirements
1. Each task must specify:
   - Task number and title
   - Target filename(s)
   - Exact code location using:
     * Module/package name
     * Class name (if applicable)
     * Function/method name (if applicable)
     * Specific code section (imports, class attributes, etc.)
   - Desired changes
   - Expected behavior

### Example fix_instructions.txt:
```
1. Update database connection handling in database.py
- Target: database.py
- Location: 
  * Class: DatabaseManager
  * Method: __init__, connect
  * Add to imports: from contextlib import contextmanager
- Changes:
  * Add connection pooling
  * Implement context manager pattern
  * Add connection timeout handling
- Expected: Thread-safe connection management with timeouts

2. Add input validation to user_service.py
- Target: user_service.py
- Location:
  * Class: UserService
  * Method: create_user
  * After imports: Add validation schemas
- Changes:
  * Add email format validation
  * Add password strength checks
  * Add username constraints
- Expected: Validated user inputs before processing

3. Enhance error handling in utils.py
- Target: utils.py
- Location:
  * Function: process_file
  * Add to imports: from typing import Optional, Union
- Changes:
  * Add type hints
  * Wrap file operations in try-except
  * Add custom exceptions
- Expected: Graceful error handling with proper type hints

4. Update logging in auth_middleware.py
- Target: auth_middleware.py
- Location:
  * Class: AuthMiddleware
  * Method: authenticate
  * At class start: Add logger configuration
- Changes:
  * Add structured logging
  * Include request ID in logs
  * Add timing information
- Expected: Detailed authentication logs with timing
```

### Key Components for Python Tasks:
1. Code Location:
   ```
   - Target: [filename.py]
   - Location:
     * Package: [if applicable]
     * Module: [module name]
     * Class: [class name]
     * Method/Function: [method/function name]
     * Specific section: [imports/class vars/etc.]
   ```

2. Changes:
   ```
   - Changes:
     * [specific code change 1]
     * [specific code change 2]
     * Dependencies to add/remove
   ```

3. Context:
   ```
   - Before: [relevant existing code]
   - After: [expected code structure]
   - Expected: [expected behavior]
   ```

### Common Python Code Sections:
1. Module Level:
   - Imports
   - Constants
   - Global variables
   - Type definitions

2. Class Level:
   - Class attributes
   - Class methods
   - Properties
   - Inner classes

3. Function Level:
   - Function signature
   - Type hints
   - Docstrings
   - Function body

4. Special Sections:
   - __init__.py contents
   - Module docstrings
   - Configuration sections
...
""", ANSIColors.CYAN))
        sys.exit(0)

    if "--setup-bin" in sys.argv:
        BinSetup.setup(os.path.abspath(__file__))
        sys.exit(0)

    # Get max concurrent tasks from command line
    try:
        max_concurrent_idx = sys.argv.index("--max-concurrent")
        max_concurrent = int(sys.argv[max_concurrent_idx + 1])
        # Remove the option and its value from argv
        sys.argv.pop(max_concurrent_idx)
        sys.argv.pop(max_concurrent_idx)
    except (ValueError, IndexError):
        max_concurrent = 5

    filenames = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    if not filenames:
        logger.error(ANSIColors.colorize(
            "Please provide filenames to pass to aider, or use '--help' for more information.",
            ANSIColors.RED
        ))
        sys.exit(1)
    
    # Validate all input files exist
    missing_files = [f for f in filenames if not Path(f).exists()]
    if missing_files:
        logger.error(ANSIColors.colorize(
            f"The following files were not found: {', '.join(missing_files)}",
            ANSIColors.RED
        ))
        sys.exit(1)
    
    runner = AsyncAiderRunner(INTRO_MESSAGE, max_concurrent)
    try:
        asyncio.run(runner.run(filenames))
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
