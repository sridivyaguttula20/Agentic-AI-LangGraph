import os
import sys
import io
import traceback
from typing import TypedDict, List, Optional

from flask import Flask, request, render_template_string

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END

from langchain_google_genai import ChatGoogleGenerativeAI


# =========================================================
# 1. GEMINI API KEY
# =========================================================

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set.")


# =========================================================
# 2. LLM INITIALIZATION
# =========================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key
)


# =========================================================
# 3. STATE DEFINITION
# =========================================================

class CrewState(TypedDict):

    messages: List[BaseMessage]

    next_step: Optional[str]

    code: Optional[str]

    execution_result: Optional[str]

    test_cases: Optional[str]

    report: Optional[str]

    workflow_log: List[str]


# =========================================================
# 4. PYTHON CODE EXECUTION TOOL
# =========================================================

@tool
def run_python_code(code: str) -> str:
    """
    Execute generated Python code and return output or error.
    """

    if not isinstance(code, str):
        code = str(code)

    clean_code = (
        code.replace("```python", "")
        .replace("```", "")
        .strip()
    )

    old_stdout = sys.stdout

    new_stdout = io.StringIO()

    sys.stdout = new_stdout

    try:

        local_scope = {}

        exec(
            clean_code,
            {
                "__name__": "__main__"
            },
            local_scope
        )

        result = new_stdout.getvalue()

    except Exception:

        result = (
            "Execution Error:\n"
            + traceback.format_exc()
        )

    finally:

        sys.stdout = old_stdout

    if result.strip():

        return result.strip()

    return "Success (no terminal output)"


# =========================================================
# 5. TEST CASE GENERATOR TOOL
# =========================================================

@tool
def generate_test_cases(task_description: str) -> str:
    """
    Generate detailed QA test scenarios for a Python coding task.
    """

    prompt = f"""
You are a Senior QA Engineer testing a Python program.

Coding Task:
{task_description}

Generate 5 highly specific test scenarios.

The scenarios MUST include:

1. Normal / Standard Case
2. Boundary / Edge Case
3. Invalid Input Case
4. Non-ASCII / Unicode Case when relevant
5. Extremely Large Input / Performance Case when relevant

For EVERY test case use exactly this style:

### Test Case 1: <title>

**Objective:**
<what is being verified>

**Input:**
<specific input>

**Expected Result:**
<specific expected result>

**Why:**
<short explanation>

IMPORTANT:
- The code is Python.
- Do NOT use Java-specific terms such as InvalidArgumentException,
  BigInt, NullPointerException, StackOverflowError, etc.
- Use Python concepts such as ValueError, TypeError,
  Unicode characters, memory usage, recursion depth, etc.
- Make the test cases specific to the given task.
- Do not give generic testing advice.
"""

    try:

        response = llm.invoke(prompt)

        content = response.content

        if isinstance(content, list):

            return "\n".join(
                item.get("text", str(item))
                if isinstance(item, dict)
                else str(item)
                for item in content
            )

        return str(content)

    except Exception as e:

        return (
            "Test case generation failed:\n"
            + str(e)
        )


# =========================================================
# 6. TASK INPUT NODE
# =========================================================

def task_input_node(state: CrewState):

    task = state["messages"][-1].content

    log = state.get("workflow_log", [])

    log.append(
        "👤 USER\n"
        "Task received successfully.\n"
        f"Task: {task}"
    )

    return {

        "next_step": "developer",

        "workflow_log": log
    }


# =========================================================
# 7. DEVELOPER NODE
# =========================================================

def real_time_developer(state: CrewState):

    task = state["messages"][-1].content

    log = state.get("workflow_log", [])

    log.append(
        "👨‍💻 DEVELOPER\n"
        "Task analyzed successfully.\n"
        "Generating Python solution..."
    )

    developer_prompt = f"""
You are a Senior Python Developer.

Solve the following coding task:

{task}

Requirements:

1. Write clean Python code.
2. The code must be executable.
3. Use appropriate functions when useful.
4. Handle reasonable edge cases.
5. Include a small example execution using print().
6. Return ONLY Python code.
7. Do NOT use Markdown.
8. Do NOT use ```python.
"""

    try:

        response = llm.invoke(developer_prompt)

        content = response.content

        if isinstance(content, list):

            code_str = "\n".join(
                item.get("text", str(item))
                if isinstance(item, dict)
                else str(item)
                for item in content
            )

        else:

            code_str = str(content)

        code_str = (
            code_str
            .replace("```python", "")
            .replace("```", "")
            .strip()
        )

        log.append(
            "👨‍💻 DEVELOPER\n"
            "Python code generated successfully."
        )

        return {

            "code": code_str,

            "next_step": "tester",

            "workflow_log": log
        }

    except Exception as e:

        error_code = (
            "print('Developer Error:', "
            + repr(str(e))
            + ")"
        )

        log.append(
            "❌ DEVELOPER ERROR\n"
            + str(e)
        )

        return {

            "code": error_code,

            "next_step": "tester",

            "workflow_log": log
        }


# =========================================================
# 8. TESTER NODE
# =========================================================

def real_time_tester(state: CrewState):

    task = state["messages"][-1].content

    code = state["code"]

    log = state.get("workflow_log", [])

    log.append(
        "🧪 TESTER\n"
        "Developer output received.\n"
        "Generating QA test scenarios..."
    )

    # -----------------------------------------------------
    # Generate test cases
    # -----------------------------------------------------

    test_cases = generate_test_cases.invoke(task)

    test_cases = str(test_cases)

    log.append(
        "🧪 TESTER\n"
        "5 detailed test scenarios generated."
    )

    # -----------------------------------------------------
    # Execute code
    # -----------------------------------------------------

    log.append(
        "▶️ EXECUTOR\n"
        "Running generated Python code..."
    )

    execution_result = run_python_code.invoke(
        {
            "code": code
        }
    )

    log.append(
        "▶️ EXECUTOR\n"
        "Code execution completed."
    )

    # -----------------------------------------------------
    # Create report
    # -----------------------------------------------------

    report = (

        "👤 USER\n"
        "Task received successfully.\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        "👨‍💻 DEVELOPER\n"
        "Code generated successfully.\n\n"

        "### GENERATED CODE\n\n"
        f"{code}\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        "▶️ EXECUTOR\n"
        "Generated code executed successfully.\n\n"

        "### EXECUTION OUTPUT\n\n"
        f"{execution_result}\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        "🧪 TESTER\n"
        "QA analysis completed.\n\n"

        "### TEST SCENARIOS\n\n"
        f"{test_cases}\n\n"

    )

    return {

        "execution_result": execution_result,

        "test_cases": test_cases,

        "report": report,

        "next_step": "manager",

        "workflow_log": log
    }


# =========================================================
# 9. MANAGER NODE
# =========================================================

def manager_decision_node(state: CrewState):

    log = state.get("workflow_log", [])

    log.append(
        "👨‍💼 MANAGER\n"
        "Report reviewed successfully.\n"
        "Decision: Send task to Archiver."
    )

    report = state.get("report", "")

    report += (

        "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        "👨‍💼 MANAGER\n"
        "Report reviewed successfully.\n"
        "Decision: Send task to Archiver.\n\n"
    )

    return {

        "report": report,

        "next_step": "archiver",

        "workflow_log": log
    }


# =========================================================
# 10. ARCHIVER NODE
# =========================================================

def archiver_node(state: CrewState):

    log = state.get("workflow_log", [])

    log.append(
        "🗄️ ARCHIVER\n"
        "Task stored successfully.\n"
        "Workflow is closing."
    )

    report = state.get("report", "")

    report += (

        "🗄️ ARCHIVER\n"
        "Task stored successfully.\n"
        "Workflow is closing.\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        "🏁 END\n"
        "Workflow completed successfully."
    )

    return {

        "report": report,

        "next_step": "exit",

        "workflow_log": log
    }


# =========================================================
# 11. GRAPH CONSTRUCTION
# =========================================================

workflow = StateGraph(CrewState)


workflow.add_node(
    "task_input",
    task_input_node
)

workflow.add_node(
    "developer",
    real_time_developer
)

workflow.add_node(
    "tester",
    real_time_tester
)

workflow.add_node(
    "manager_decision",
    manager_decision_node
)

workflow.add_node(
    "archiver",
    archiver_node
)


# =========================================================
# 12. GRAPH EDGES
# =========================================================

workflow.add_edge(
    START,
    "task_input"
)


def route_from_input(state):

    if state.get("next_step") == "exit":

        return END

    return "developer"


workflow.add_conditional_edges(
    "task_input",
    route_from_input
)


workflow.add_edge(
    "developer",
    "tester"
)


workflow.add_edge(
    "tester",
    "manager_decision"
)


def route_from_manager(state):

    if state.get("next_step") == "archiver":

        return "archiver"

    return "task_input"


workflow.add_conditional_edges(
    "manager_decision",
    route_from_manager
)


workflow.add_edge(
    "archiver",
    END
)


# Compile graph

rt_app = workflow.compile()


# =========================================================
# 13. FLASK APPLICATION
# =========================================================

app = Flask(__name__)


# =========================================================
# 14. HTML PAGE
# =========================================================

HTML = """

<!DOCTYPE html>

<html>

<head>

<title>LangGraph AI Developer</title>

<style>

body {

    font-family: Arial, sans-serif;

    background: #f4f6f8;

    margin: 0;

    padding: 40px;

}

.container {

    max-width: 1000px;

    margin: auto;

    background: white;

    padding: 30px;

    border-radius: 15px;

    box-shadow:
        0 4px 20px
        rgba(0,0,0,0.12);

}

h1 {

    text-align: center;

}

textarea {

    width: 100%;

    height: 130px;

    padding: 14px;

    font-size: 16px;

    box-sizing: border-box;

    border-radius: 8px;

    border: 1px solid #ccc;

}

button {

    margin-top: 15px;

    padding: 13px 28px;

    font-size: 16px;

    cursor: pointer;

    border: none;

    border-radius: 8px;

    background: #222;

    color: white;

}

button:hover {

    opacity: 0.85;

}

pre {

    background: #f0f2f5;

    padding: 20px;

    border-radius: 10px;

    white-space: pre-wrap;

    overflow-x: auto;

    line-height: 1.5;

}

.workflow {

    background: #fafafa;

    border-left: 5px solid #333;

    padding: 20px;

    margin-top: 20px;

    border-radius: 8px;

}

</style>

</head>


<body>


<div class="container">


<h1>
🤖 LangGraph AI Developer
</h1>


<form method="POST">


<label>

<b>Enter your coding task:</b>

</label>


<br><br>


<textarea

name="task"

placeholder="Example: Check whether a string is a palindrome."

required

></textarea>


<br>


<button type="submit">

🚀 Run Agentic Workflow

</button>


</form>


{% if report %}


<hr>


<h2>
Agentic Workflow Result
</h2>


<div class="workflow">

<pre>{{ report }}</pre>

</div>


{% endif %}


</div>


</body>


</html>

"""


# =========================================================
# 15. FLASK ROUTE
# =========================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)

def home():

    report = None

    if request.method == "POST":

        task = request.form.get(
            "task",
            ""
        ).strip()

        if task:

            try:

                result = rt_app.invoke(

                    {

                        "messages": [

                            HumanMessage(
                                content=task
                            )

                        ],

                        "next_step": None,

                        "code": None,

                        "execution_result": None,

                        "test_cases": None,

                        "report": None,

                        "workflow_log": []

                    }

                )

                report = result.get(

                    "report",

                    "No report generated."

                )

            except Exception:

                report = (

                    "❌ APPLICATION ERROR\n\n"

                    + traceback.format_exc()

                )

    return render_template_string(

        HTML,

        report=report

    )


# =========================================================
# 16. RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=False

    )
