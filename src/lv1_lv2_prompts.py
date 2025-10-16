"""
Prompt templates for Level 1 and Level 2 query generation.
This module contains all the prompt templates used in the MCP server analysis and query generation pipeline.
"""

PLATFORM_ID_PROMPT = """
# ROLE & GOAL
You are a highly intelligent text analysis engine. Your task is to identify the primary **user-facing Platform, Brand, or Product Keyword** from the provided text.

# INSTRUCTIONS
1.  Read the "Text to Analyze" carefully.
2.  Identify the main, specific, proper noun that represents the core service or brand being offered.
3.  This keyword should be a unique, user-recognizable identifier, like "GitHub", "Notion", "UseGrant", "Stripe", or "PaddleOCR".
4.  **Crucially, do NOT extract generic terms OR technology protocols.** This includes common words like "API", "Server", "Platform", "Service", and especially technical standards like **"MCP"** or **"Model Context Protocol"**.
5.  If you cannot find a specific, unique platform keyword after ignoring the terms above, the keyword is "N/A".
6.  Your output MUST be a single line containing the keyword **wrapped in `<keyword></keyword>` tags**.

# EXAMPLES
### Example 1 (Ignoring "MCP" and "API")
Text to Analyze: "This is a Model Context Protocol (MCP) server for interacting with the UseGrant API. It provides a set of tools for managing providers and clients through the UseGrant platform."
Your Output:
<keyword>UseGrant</keyword>

### Example 2 (Specific Technology Brand)
Text to Analyze: "A server for performing optical character recognition (OCR) using the powerful PaddleOCR engine."
Your Output:
<keyword>PaddleOCR</keyword>

### Example 3 (No Specific Brand Found)
Text to Analyze: "An MCP server that provides comprehensive architectural expertise through specialized agents, resources, and tools."
Your Output:
<keyword>N/A</keyword>

---

# YOUR TASK
**Text to Analyze**: "{text_to_analyze}"
**Your Output**:
"""

TASK_TYPE_PROMPT = """
# ROLE & GOAL
You are a pragmatic AI assistant designer. Your task is to analyze a given tool and classify its primary user intent. You need to determine if this tool represents a **"Final Goal Task"** or a **"Middleware Task"**.

# DEFINITIONS
- **Final Goal Task**: A task that a user would directly ask an AI assistant to perform as a complete, standalone goal. These tasks provide direct value to the user. (Examples: "search_for_videos", "download_a_file", "send_an_email", "translate_text").
- **Middleware Task**: A task that is usually an intermediate or prerequisite step required to achieve a larger goal. Users rarely, if ever, ask for this task directly. (Examples: "login", "authenticate", "get_api_key", "list_available_regions", "check_status").

# INSTRUCTIONS
1.  Read the tool's name and description.
2.  Based on the definitions, decide if it's a "Final Goal Task" or a "Middleware Task".
3.  Your output MUST be a single line containing the classification **wrapped in `<task_type></task_type>` tags**. The value must be either `final_goal` or `middleware`.

# EXAMPLE 1
## Tool Name: "search_youtube_videos"
## Tool Description: "Searches for videos on YouTube based on a query."
## Your Output:
<task_type>final_goal</task_type>

# EXAMPLE 2
## Tool Name: "youtube_login"
## Tool Description: "Authenticates the user and obtains an access token for the YouTube API."
## Your Output:
<task_type>middleware</task_type>

---

# YOUR TASK
## Tool Name: "{tool_name}"
## Tool Description: "{tool_description}"
## Your Output:
"""

GENERATOR_L1_PROMPT_FINAL = """
# ROLE & GOAL
You are an expert user of a specific software tool. Your task is to write 2 user queries in English. These queries should be direct commands to an AI assistant, demonstrating how a real user would request to use the specified tool on its target platform. The queries must be specific, self-contained, and unambiguous.

# CONTEXT FOR YOUR TASK
You are generating queries for the following specific tool:

- **Platform Name**: "{platform_name}" 
  (The user-facing brand or service, e.g., "GitHub", "Heroku", "Pyodide")
- **Server Name**: "{server_name}" 
  (The name of the software package providing the tool, e.g., "github-mcp-server")
- **Tool Name**: "{tool_name}"
  (The specific function to be executed, e.g., "create_repository")
- **Tool Description**: "{tool_description}"
  (What the tool does in plain English)

# INSTRUCTIONS
1.  **Direct Command Tone**: Your queries should be direct commands, not questions.
2.  **Mention Platform**: Each query MUST explicitly mention the **Platform Name** ("{platform_name}"). This is crucial for context.
3.  **Reflect Tool's Function**: The command's intent MUST be a direct application of the **Tool Description** and its **Tool Inputs**.
4.  **AVOID AMBIGUOUS REFERENCES (VERY IMPORTANT)**:
    - Do NOT use vague pointers like "this code", "that file", "the script".
    - For tools that execute code, embed a short, realistic code snippet directly in the query.
    - For tools that use files, specify a plausible filename.
5.  **Be Specific**: Incorporate realistic example values for the parameters listed in **Tool Inputs**. This makes the query more concrete and useful for training.
6.  **Format**: You MUST wrap each generated query in `<query></query>` tags.

# EXAMPLE
## Context:
- **Platform Name**: "GitHub"
- **Server Name**: "github-mcp-server"
- **Tool Name**: "create_repository"
- **Tool Description**: "Creates a new repository on GitHub."

## Your Output:
<query>On GitHub, create a new private repository named 'my-secret-project' with the description 'This is for the new API'.</query>
<query>Use GitHub to create a public repository called 'awesome-list'.</query>

---

# YOUR TASK
Now, using the context provided at the top, generate 2 user queries following all the rules.

# YOUR OUTPUT
"""

VERIFIER_PROMPT = """
# ROLE & GOAL
You are a meticulous AI System Analyst. Your task is to perform a strict validation of a user query against a specific tool. The query must be a perfect example of a command a user would give for this tool.

# CONTEXT
- **Platform Name**: "{platform_name}"
- **Tool Name**: "{tool_name}"
- **Tool Description**: "{tool_description}"

# VALIDATION CRITERIA
You must check the query against ALL of the following rules. If ANY rule is violated, the result is `false`.

1.  **Intent Match**: Does the user's primary goal in the query directly and logically map to the tool's function described in the **Tool Description**?
2.  **Platform Mention**: Does the query EXPLICITLY mention the required **Platform Name** ("{platform_name}")?
3.  **Self-Contained & Unambiguous**: Is the query understandable on its own without needing prior conversation? It must NOT use vague references like "this code", "that file" unless it also provides a concrete example (e.g., embedding code, providing a filename).
4.  **No Meta-Language**: Does the query sound like a real user? It must NOT refer to the tool itself (e.g., "use this tool", "run the function"). The command must be direct.

# YOUR TASK
Analyze the query below based on all the criteria.

## Query to Verify: "{query}"

# YOUR RESPONSE
Structure your output as follows:
- First, provide the boolean judgement (`true` or `false`) **wrapped in `<is_match></is_match>` tags**.
- Second, on a new line, provide a one-sentence explanation for your decision, specifically mentioning which rule was passed or failed.

# EXAMPLE 1: Perfect Match (All Rules Pass)
## Context:
- Platform Name: "GitHub"
- Tool Name: "create_repository"
- Tool Description: "Creates a new repository on GitHub."
## Query to Verify: "On GitHub, please create a new repository for me named 'my-next-project'."
## Your Output:
<is_match>true</is_match>
The query matches the tool's intent, mentions the platform 'GitHub', is self-contained, and uses natural user language.

# EXAMPLE 2: Failed (No Platform Mention)
## Context:
- Platform Name: "GitHub"
- Tool Name: "create_repository"
- Tool Description: "Creates a new repository on GitHub."
## Query to Verify: "Create a new repository for me named 'my-next-project'."
## Your Output:
<is_match>false</is_match>
The query fails validation because it does not explicitly mention the required platform 'GitHub'.

# EXAMPLE 3: Failed (Ambiguous Reference)
## Context:
- Platform Name: "Pyodide"
- Tool Name: "execute-python"
- Tool Description: "Executes a string of Python code."
## Query to Verify: "Run this Python code using Pyodide."
## Your Output:
<is_match>false</is_match>
The query fails validation because "this Python code" is an ambiguous reference, violating the self-contained rule.

# EXAMPLE 4: Failed (Meta-Language)
## Context:
- Platform Name: "LocalFS"
- Tool Name: "delete_file"
- Tool Description: "Deletes a file from the filesystem."
## Query to Verify: "Use the LocalFS tool to delete the file 'temp.log'."
## Your Output:
<is_match>false</is_match>
The query fails validation because "Use the LocalFS tool" is meta-language, not natural user language.
"""

PLATFORM_COUPLING_CLASSIFIER_PROMPT = """
# ROLE & GOAL
You are a Product Analyst specializing in software tools and user behavior. Your task is to determine if a tool's core function is tightly coupled with its specific platform, or if it represents a generic concept that exists across many platforms.

# DEFINITIONS
- **Tightly Coupled**: The tool's main concept or terminology is unique to its platform and doesn't make sense outside of it. A user would HAVE to mention the platform name to be understood.
  - *Examples*: "Managing Heroku Dynos", "Creating a GitHub Pull Request", "Resolving a Jira Transition". The concepts "Dyno", "Pull Request", and "Jira Transition" are iconic to their platforms.
- **Generic Concept**: The tool's function is a common action that many different platforms offer under similar names. A user could describe this task without mentioning a specific brand.
  - *Examples*: "Creating a file", "Sending an email", "Uploading an image", "Renaming a project".

# INSTRUCTIONS
1.  Analyze the tool's name, description, and the platform it belongs to.
2.  Decide if a typical user would naturally mention the platform when asking for this task.
3.  Your output MUST be a single line containing the classification **wrapped in `<coupling></coupling>` tags**. The value must be either `tightly_coupled` or `generic_concept`.

# EXAMPLE 1
## Platform: "GitHub"
## Tool Name: "create_pull_request"
## Tool Description: "Creates a new pull request to merge changes from one branch to another."
## Your Output:
<coupling>tightly_coupled</coupling>

# EXAMPLE 2
## Platform: "Google Drive"
## Tool Name: "create_document"
## Tool Description: "Creates a new blank document in the user's drive."
## Your Output:
<coupling>generic_concept</coupling>

# EXAMPLE 3
## Platform: "Stripe"
## Tool Name: "create_invoice"
## Tool Description: "Generates a new invoice for a customer."
## Your Output:
<coupling>generic_concept</coupling>

---

# YOUR TASK
## Platform: "{platform_name}"
## Tool Name: "{tool_name}"
## Tool Description: "{tool_description}"

## Your Output:
"""

GENERATOR_L2_PROMPT = """
# ROLE & GOAL
You are an expert user of various software, commanding an AI assistant. Your task is to write 2 natural-sounding user queries. The queries should reflect how a real human would ask for a task, based on how tightly the task is tied to its platform.

# CONTEXT FOR YOUR TASK
- **Platform Name**: "{platform_name}"
- **Tool Name**: "{tool_name}"
- **Tool Description**: "{tool_description}"
- **Tool Inputs**:
{formatted_schema}

# CORE INSTRUCTION
- **Platform Mention Rule**: {platform_mention_rule}

# GENERAL INSTRUCTIONS
1.  **Follow the Platform Mention Rule**: This is the most important instruction. Adhere strictly to whether you should or should not mention the platform name.
2.  **Embody the User**: Your tone must be that of a user giving a command.
3.  **AVOID META-LANGUAGE**: Do NOT refer to the tool itself (e.g., "use the tool").
4.  **Be Specific and Actionable**: Incorporate realistic example values for the parameters listed in "Tool Inputs".
5.  **Format**: Wrap each query in `<query></query>` tags.

# EXAMPLE 1: Tightly Coupled
## Platform Mention Rule: "You MUST mention the platform 'GitHub' because the function is iconic to it."
## Context: Tool is 'create_pull_request' on GitHub.
## Correct Output:
<query>Create a pull request on GitHub to merge the 'feature-x' branch into 'main'.</query>
<query>I need to open a new GitHub pull request for my latest changes.</query>

# EXAMPLE 2: Generic Concept
## Platform Mention Rule: "You MUST NOT mention the platform 'Google Drive'. The function is a generic concept."
## Context: Tool is 'create_document' on Google Drive.
## Correct Output:
<query>Create a new document for me titled 'Meeting Notes Q3'.</query>
<query>I need to start a new doc.</query>

---

# YOUR TASK
Now, using the context and the CORE INSTRUCTION provided at the top, generate 2 user queries.

# YOUR OUTPUT
"""

VERIFIER_L2_PROMPT = """
# ROLE & GOAL
You are a highly discerning AI Routing Analyst. Your task is to verify if a generated user query is a high-quality, natural-sounding training example for a specific tool, following a given rule.

# CONTEXT
- **Tool's Platform**: "{platform_name}"
- **Tool Name**: "{tool_name}"
- **Tool Description**: "{tool_description}"

# VALIDATION CRITERIA
You must check the query against ALL of the following rules. If ANY rule is violated, the result is `false`.

1.  **Platform Mention Rule Adherence**: The query must strictly follow this rule: **{platform_mention_rule}**
2.  **Logical Routing Match**: Is the tool a direct and sensible way to fulfill the user's request?
3.  **Self-Contained & Unambiguous**: Is the query understandable on its own? It must not use vague references like "this code" unless a concrete example is embedded.
4.  **No Meta-Language**: Does the query sound like a real user? It must not refer to the tool itself (e.g., "use this tool", "run the function").

# YOUR TASK
Analyze the query below based on all the criteria, especially the Platform Mention Rule.

## Query to Verify: "{query}"

# YOUR RESPONSE
Structure your output as follows:
- First, provide the boolean judgement (`true` or `false`) **wrapped in `<is_match></is_match>` tags**.
- Second, on a new line, provide a one-sentence explanation for your decision, specifically mentioning which rule was passed or failed.
"""

USER_FACING_CLASSIFIER_PROMPT = """
# ROLE & GOAL
You are an AI Product Manager. Your task is to classify a given tool based on its intended user. You need to determine if it's a **"User-Facing Task"** or a **"System-Facing Task"**.

# DEFINITIONS
- **User-Facing Task**: An action that a typical end-user (like a writer, designer, project manager, or even a developer using a platform) would directly command an AI assistant to perform to achieve a personal or business goal. These tasks operate on user-understandable concepts like documents, repositories, images, emails, or playlists.
- **System-Facing Task**: An action related to system administration, infrastructure management, backend debugging, or managing abstract, non-visible resources. These tasks are typically performed by system administrators or developers maintaining a service, not using it. They operate on concepts like caches, database indexes, memory entries, or container pods.

# INSTRUCTIONS
1.  Read the tool's name and description carefully.
2.  Based on the definitions, decide if it's a "User-Facing Task" or a "System-Facing Task".
3.  Your output MUST be a single line containing the classification **wrapped in `<classification></classification>` tags**. The value must be either `user_facing` or `system_facing`.

# EXAMPLE 1
## Tool Name: "create_spreadsheet"
## Tool Description: "Creates a new spreadsheet in the user's cloud drive."
## Your Output:
<classification>user_facing</classification>

# EXAMPLE 2
## Tool Name: "clear_redis_cache"
## Tool Description: "Purges all keys from the specified Redis cache instance to free up memory."
## Your Output:
<classification>system_facing</classification>

# EXAMPLE 3
## Tool Name: "delete_memory_entry"
## Tool Description: "Deletes a specific memory entry for a user from the agent's long-term memory service."
## Your Output:
<classification>system_facing</classification>

---

# YOUR TASK
## Tool Name: "{tool_name}"
## Tool Description: "{tool_description}"
## Your Output:
"""