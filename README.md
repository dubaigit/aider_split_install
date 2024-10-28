# aider_split Usage Guide (Max 5 Concurrent Tasks)

## Quick Start
```bash
# Install globally
aider_split --setup-bin

# Run with 5 concurrent tasks
aider_split --max-concurrent 5 file1.py file2.py file3.py file4.py file5.py
```

## Writing fix_instructions.txt for Parallel Processing

### Example with 5 Concurrent Tasks:
```
1. Update user authentication (auth.py)
- Target: auth.py
- Location:
  * Class: Authentication
  * Method: verify_token, create_token
  * Add imports: from jwt import encode, decode
- Changes:
  * Implement JWT token handling
  * Add token expiration
  * Add refresh token logic
- Expected: JWT-based auth with refresh tokens

2. Enhance database models (models.py)
- Target: models.py
- Location:
  * Class: BaseModel
  * Method: __init__, save
  * Add imports: from sqlalchemy.sql import func
- Changes:
  * Add timestamps (created_at, updated_at)
  * Add soft delete functionality
  * Add schema validation
- Expected: Improved model with tracking

3. Implement caching (cache.py)
- Target: cache.py
- Location:
  * Class: CacheManager
  * Method: get_or_set
  * Add imports: from redis import Redis
- Changes:
  * Add Redis connection
  * Implement cache expiry
  * Add cache invalidation
- Expected: Redis-based caching system

4. Add API rate limiting (middleware.py)
- Target: middleware.py
- Location:
  * Class: RateLimiter
  * Method: check_limit
  * Add imports: import time, threading
- Changes:
  * Implement token bucket algorithm
  * Add user-based limits
  * Add burst handling
- Expected: Flexible rate limiting

5. Update logging system (logger.py)
- Target: logger.py
- Location:
  * Class: CustomLogger
  * Method: log_request
  * Add imports: import structlog
- Changes:
  * Add structured logging
  * Add request tracing
  * Add performance metrics
- Expected: Comprehensive logging system
```

### Tips for 5 Concurrent Tasks:
1. **File Independence**
   - Each task should work on different files
   - Avoid interdependent changes
   - Clearly specify file boundaries

2. **Resource Usage**
   - Each task gets its own aider instance
   - Monitor system resources
   - Consider memory usage

3. **Task Organization**
   ```
   Task 1 (auth.py):      Authentication changes
   Task 2 (models.py):    Database model updates
   Task 3 (cache.py):     Caching implementation
   Task 4 (middleware.py): Rate limiting
   Task 5 (logger.py):    Logging system
   ```

## Running Tasks

### Basic Command
```bash
aider_split --max-concurrent 5 *.py
```

### With Specific Files
```bash
aider_split --max-concurrent 5 auth.py models.py cache.py middleware.py logger.py
```

### Progress Display
```
==================== 🟢 STARTING TASK 1 ====================
[Auth updates running...]

==================== 🟢 STARTING TASK 2 ====================
[Model updates running...]

==================== 🟢 STARTING TASK 3 ====================
[Cache implementation running...]

==================== 🟢 STARTING TASK 4 ====================
[Rate limiting running...]

==================== 🟢 STARTING TASK 5 ====================
[Logging updates running...]
```

## Task Template for Each File

### Template Structure:
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

## Efficiency Tips

1. **Parallel Planning**
   - Group related changes by file
   - Ensure changes don't conflict
   - Distribute work evenly

2. **Resource Management**
   - 5 concurrent tasks use more memory
   - Monitor system performance
   - Adjust if needed

3. **Error Handling**
   - Each task has isolated error handling
   - Failed tasks don't affect others
   - Easy to retry specific tasks

## Command Reference
```bash
# Full options
aider_split --help

Options:
  --max-concurrent 5   Run 5 tasks simultaneously
  --setup-bin         Install globally
  --help, -h         Show this help message
```

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

## Progress Monitoring
- Real-time status for 5 concurrent tasks
- Color-coded output
- Clear task boundaries
- Success/failure indicators

Need any clarification or specific examples?

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
