Note: Occasionally, you may need to remind Cline to use Aider, especially when applying or verifying fixes. If Cline seems to overlook this tool, a gentle reminder to follow Step 3 (Just say use Aider/ or he misses creating the file create instructions) will help ensure instructions are correctly applied and checked.

This way, if Aider isn't used as expected, Cline has a prompt to follow the specified workflow.

This prompt provides a solid foundation and continues to evolve. Among my AI-to-AI integrations, I have Open Interpreter successfully controlling Aider for coding tasks, which enables seamless automation of development workflows.

For questions or support, contact me at binkenaid@gmail.com

Custom Instructions:


"""
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
aider --model anthropic/claude-3-5-sonnet-20241022 --no-pretty --message-file fix_instructions.txt  <filename1>  <filename2>
     ```

4. **Verify and Reiterate**:
   - Reread the file to confirm if all specified changes were correctly applied.
   - If successful, gradually increase max-concurrent
   - If any modifications are incomplete, adjust instructions and rerun
   """