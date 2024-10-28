# aider_split

A powerful parallel code modification tool that leverages aider CLI to run multiple concurrent code refactoring tasks, supporting parallel modifications within the same file using precise search/replace patterns.

> ⚠️ **IMPORTANT NOTE**: 
> - Always start with `--max-concurrent 1` for initial runs
> - Gradually increase to maximum of 5 based on:
>   * File size
>   * Complexity of changes
>   * System resources
> - Aider caches file state between runs, multiple concurrent changes might cause state mismatches
> - For large files, use smaller batches of concurrent tasks
> - Monitor changes carefully when scaling up concurrency

[![GitHub](https://img.shields.io/github/license/dubaigit/aider_split_install)](https://github.com/dubaigit/aider_split_install/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)

## Features

- 🚀 Run up to 5 concurrent code modification tasks
- 📝 Concurrent modifications within the same file using search/diff patterns
- 🎯 Non-conflicting parallel changes in single file sections
- 🔄 Automatic task management and cleanup
- 🎨 Color-coded progress monitoring
- 🛡️ Isolated error handling per task

## Prerequisites

- Python 3.7+
- aider CLI tool (`pip install aider-chat`)
- Anthropic API key (for Claude access)

## Recommended Scaling Strategy

Start with:
```bash
# Initial run
aider_split --max-concurrent 1 app.py

# After successful run, increase gradually
aider_split --max-concurrent 2 app.py
aider_split --max-concurrent 3 app.py
# ... and so on up to 5
```

Monitor each run for:
- Successful pattern matching
- Correct file state handling
- No conflicting changes
- System resource usage

## Quick Start

### Linux/MacOS
```bash
# Install globally
sudo aider_split --setup-bin

# Run with gradually increasing concurrency
aider_split --max-concurrent 1 app.py  # Start here
aider_split --max-concurrent 5 app.py  # Scale up when ready
```

### Windows
```powershell
# Navigate to script directory
cd path\to\aider_split

# Run directly with Python - start with low concurrency
python aider_split.py --max-concurrent 1 app.py

# Scale up when stable
python aider_split.py --max-concurrent 5 app.py

# Or, from any location using full path
python C:\path\to\aider_split\aider_split.py --max-concurrent 5 app.py
```

## Using with CLINE

```markdown
1. **Load File and Identify Issues**:
   - Read the contents of the specified file and analyze it to understand any errors or issues requiring fixes.
   - Document specific issues and the code sections needing modification.

2. **Write Instructions**:
   - Write detailed, clear instructions in `fix_instructions.txt`, outlining the exact changes required.
   - Ensure clarity, specifying exact lines or code sections to modify.
      2. Structure each chunk in fix_instructions.txt as follows:
         ```
         1. [Task Title]
         - Specific goal/outcome
         - Any constraints or requirements
         - Expected behavior
         - Files to modify
         ```

3. **Use Aider Command**:
   - Execute the following command to apply fixes:  
     ```bash
     aider_split --max-concurrent 1 file1.py file2.py  # Start with 1
     ```

4. **Verify and Reiterate**:
   - Reread the file to confirm if all specified changes were correctly applied.
   - If successful, gradually increase max-concurrent
   - If any modifications are incomplete, adjust instructions and rerun
```

> **Note**: 
> - When using CLINE, you may need to remind it once to use aider by saying "Let's use aider to modify these files" at the start of your conversation.
> - CLINE may struggle with very large files. Using aider_split with precise search/replace patterns is recommended for better handling of large codebases.

## Complete Example Usage

1. **Create your Python file (app.py)**:
```python
import os
import sys

class UserManager:
    def __init__(self):
        self.users = {}

    def create_user(self, data):
        username = data.get('username')
        self.users[username] = data
        return True

    def delete_user(self, user_id):
        if user_id in self.users:
            del self.users[user_id]
```

2. **Create fix_instructions.txt**:
```
1. Update method signatures (app.py)
- Target: app.py
- Location:
  * Class: UserManager
  * Method: create_user
  * Search Pattern: def create_user(self, data):
- Changes:
  * Add type hints
  * Add validation
  * Add docstring
- Expected: Type-safe user creation

2. Add error handling (app.py)
- Target: app.py
- Location:
  * Class: UserManager
  * Method: delete_user
  * Search Pattern: def delete_user(self, user_id):
- Changes:
  * Add try-except
  * Add logging
  * Add return value
- Expected: Robust error handling

3. Update imports (app.py)
- Target: app.py
- Location:
  * Section: imports
  * Search Pattern: import os\nimport sys
- Changes:
  * Add typing imports
  * Add logging import
  * Add validation imports
- Expected: Complete imports

4. Add class docstring (app.py)
- Target: app.py
- Location:
  * Class: UserManager
  * Search Pattern: class UserManager:
- Changes:
  * Add class docstring
  * Add type hints for class attrs
- Expected: Documented class

5. Add validation methods (app.py)
- Target: app.py
- Location:
  * Class: UserManager
  * After: delete_user
- Changes:
  * Add validate_user method
  * Add user existence check
- Expected: Input validation
```

3. **Run aider_split with progressive scaling**:

Linux/MacOS:
```bash
# Start with single task
aider_split --max-concurrent 1 app.py

# If successful, scale up gradually
aider_split --max-concurrent 3 app.py
aider_split --max-concurrent 5 app.py
```

Windows:
```powershell
# Start with single task
python aider_split.py --max-concurrent 1 app.py

# Scale up if successful
python aider_split.py --max-concurrent 3 app.py
python aider_split.py --max-concurrent 5 app.py
```

4. **Modified app.py result**:
```python
from typing import Dict, Optional, Any
import os
import sys
import logging
from validators import validate_username

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserManager:
    """
    Manages user operations including creation, deletion, and validation.
    
    Attributes:
        users (Dict[str, Dict[str, Any]]): Storage for user data
    """
    
    def __init__(self):
        self.users: Dict[str, Dict[str, Any]] = {}

    def validate_user(self, username: str) -> bool:
        """
        Validate user data before operations.
        
        Args:
            username (str): Username to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        return bool(username and validate_username(username))

    def create_user(self, data: Dict[str, Any]) -> bool:
        """
        Create a new user with validated data.
        
        Args:
            data (Dict[str, Any]): User data containing username
            
        Returns:
            bool: Success status
            
        Raises:
            ValueError: For invalid input
        """
        try:
            username = data.get('username')
            if not self.validate_user(username):
                raise ValueError("Invalid username")
            
            if username in self.users:
                raise ValueError("Username exists")
                
            self.users[username] = data
            logger.info(f"User {username} created")
            return True
            
        except Exception as e:
            logger.error(f"Create user failed: {e}")
            return False

    def delete_user(self, user_id: str) -> bool:
        """
        Delete user by ID.
        
        Args:
            user_id (str): User ID
            
        Returns:
            bool: Success status
        """
        try:
            if user_id not in self.users:
                logger.warning(f"User {user_id} not found")
                return False
                
            del self.users[user_id]
            logger.info(f"User {user_id} deleted")
            return True
            
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False
```

## Task Template for Concurrent Changes

### Template Structure:
```
[task number]. [task title] ([filename])
- Target: [filename]
- Location:
  * Section: [specific part of file]
  * Class/Function: [name]
  * Search Pattern: [exact code to match]
- Changes:
  * [detailed change 1]
  * [detailed change 2]
  * [detailed change 3]
- Dependencies:
  * Add: [new imports/requirements]
  * Remove: [old imports/code]
- Expected: [expected outcome]
```

## Efficiency Tips

1. **Pattern Matching**
   - Use unique, exact code matches
   - Include surrounding context if needed
   - Avoid overlapping patterns

2. **Resource Management**
   - Monitor concurrent task load
   - Balance task size and complexity
   - Consider system resources

3. **Error Handling**
   - Each task has isolated error handling
   - Failed tasks don't affect others
   - Easy to retry specific tasks

## Command Reference
```bash
# Full options
aider_split --help

Options:
  --max-concurrent 5   Run 5 tasks simultaneously (start with 1)
  --setup-bin         Install globally
  --help, -h         Show this help message
```

## Common Use Cases

1. **Large File Refactoring**
```bash
# Start with single task
aider_split --max-concurrent 1 large_module.py
# Scale up if successful
aider_split --max-concurrent 5 large_module.py
```

2. **Multiple Files**
```bash
# Start conservatively
aider_split --max-concurrent 1 *.py
# Increase if stable
aider_split --max-concurrent 5 *.py
```

3. **Targeted Changes**
```bash
# Begin with single task
aider_split --max-concurrent 1 specific_file.py
# Scale as needed
aider_split --max-concurrent 5 specific_file.py
```

## Progress Monitoring
- Real-time status for concurrent tasks
- Color-coded output
- Clear task boundaries
- Success/failure indicators

## Contributing

1. Fork the [repository](https://github.com/dubaigit/aider_split_install)
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see [LICENSE](https://github.com/dubaigit/aider_split_install/blob/main/LICENSE) file for details

## Author

[dubaigit](https://github.com/dubaigit)