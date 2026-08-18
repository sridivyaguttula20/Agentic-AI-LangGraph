
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
    raise ValueError(
        "GEMINI_API_KEY is not set. Please set your Gemini API key first."
    )


# =========================================================
# 2. LLM INITIALIZATION
# =========================================================

llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key
)

llm = llm_flash


# =========================================================
# 3. STATE DEFINITION
# =========================================================

class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]


# =========================================================
# 4. TOOLS
# =========================================================

@tool
def run_python_code(code: str) -> str:
    """Execute Python code and return output or error."""

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

        exec(clean_code, {}, local_scope)

        result = new_stdout.getvalue()

    except Exception:
        result = f"Execution Error:\n{traceback.format_exc()}"

    finally:
        sys.stdout = old_stdout

    if result.strip():
        return result.strip()

    return "Success (no terminal output)"


@tool
def generate_test_cases(task_description: str) -> str:
    """Generate test scenarios for a coding task."""

    prompt = (
        "You are a Senior QA Engineer. "
        "Generate 3 to 5 highly specific test scenarios "
        f"for the following coding task: '{task_description}'. "
        "Include normal cases and edge cases. "
        "Return them as a numbered list."
    )

    try:
        response = llm.invoke(prompt)

        if hasattr(response, "content"):
            content = response.content

            if isinstance(content, list):
                return "\n".join(
                    item.get("text", str(item))
                    if isinstance(item, dict)
                    else str(item)
                    for item in content
                )

            return str(content)

        return str(response)

    except Exception as e:
        return f"Test case generation failed:\n{str(e)}"


# =========================================================
# 5. GRAPH NODES
# =========================================================

def task_input_node(state: CrewState):

    return {
        "next_step": "developer"
    }


def real_time_developer(state: CrewState):

    task = state["messages"][-1].content

    dev_prompt = (
        "Write a clean Python script to solve this coding task:\n"
        f"{task}\n\n"
        "Requirements:\n"
        "1. Return only Python code.\n"
        "2. Do not use Markdown.\n"
        "3. Do not include ```python.\n"
        "4. Make the program simple and executable."
    )

    try:
        response = llm_flash.invoke(dev_prompt)

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

        return {
            "code": code_str.strip()
        }

    except Exception as e:

        return {
            "code": f"# Developer Error\nprint('Error generating code: {str(e)}')"
        }


def real_time_tester(state: CrewState):

    task = state["messages"][-1].content

    # Generate test cases
    test_cases = generate_test_cases.invoke(task)

    cases_str = str(test_cases)

    # Execute generated Python code
    execution_result = run_python_code.invoke(
        {
            "code": state["code"]
        }
    )

    report = (
        "### GENERATED CODE\n\n"
        f"{state['code']}\n\n"

        "### EXECUTION OUTPUT\n\n"
        f"{execution_result}\n\n"

        "### TEST SCENARIOS\n\n"
        f"{cases_str}"
    )

    return {
        "report": report
    }


def manager_decision_node(state: CrewState):

    return {
        "next_step": "archiver"
    }


def archiver_node(state: CrewState):

    return {
        "next_step": "exit"
    }


# =========================================================
# 6. GRAPH CONSTRUCTION
# =========================================================

rt_workflow = StateGraph(CrewState)

rt_workflow.add_node(
    "task_input",
    task_input_node
)

rt_workflow.add_node(
    "developer",
    real_time_developer
)

rt_workflow.add_node(
    "tester",
    real_time_tester
)

rt_workflow.add_node(
    "manager_decision",
    manager_decision_node
)

rt_workflow.add_node(
    "archiver",
    archiver_node
)


# START → TASK INPUT

rt_workflow.add_edge(
    START,
    "task_input"
)


# TASK INPUT → DEVELOPER

def route_from_input(state: CrewState):

    if state.get("next_step") == "exit":
        return END

    return "developer"


rt_workflow.add_conditional_edges(
    "task_input",
    route_from_input
)


# DEVELOPER → TESTER

rt_workflow.add_edge(
    "developer",
    "tester"
)


# TESTER → MANAGER

rt_workflow.add_edge(
    "tester",
    "manager_decision"
)


# MANAGER → ARCHIVER

def route_from_decision(state: CrewState):

    if state.get("next_step") == "archiver":
        return "archiver"

    return "task_input"


rt_workflow.add_conditional_edges(
    "manager_decision",
    route_from_decision
)


# ARCHIVER → END

rt_workflow.add_edge(
    "archiver",
    END
)


# Compile graph

rt_app = rt_workflow.compile()


# =========================================================
# 7. FLASK WEB APP
# =========================================================

app = Flask(__name__)


HTML = """
<!DOCTYPE html>

<html>

<head>

    <title>LangGraph AI Developer</title>

    <style>

        body {
            font-family: Arial, sans-serif;
            background: #f4f4f4;
            margin: 0;
            padding: 40px;
        }

        .container {
            max-width: 900px;
            margin: auto;
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        h1 {
            text-align: center;
        }

        textarea {
            width: 100%;
            height: 120px;
            padding: 12px;
            font-size: 16px;
            box-sizing: border-box;
            border-radius: 8px;
            border: 1px solid #ccc;
        }

        button {
            margin-top: 15px;
            padding: 12px 25px;
            font-size: 16px;
            cursor: pointer;
            border-radius: 8px;
            border: none;
            background: #222;
            color: white;
        }

        button:hover {
            opacity: 0.85;
        }

        pre {
            background: #f0f0f0;
            padding: 20px;
            border-radius: 8px;
            white-space: pre-wrap;
            overflow-x: auto;
        }

        .error {
            background: #ffe5e5;
            padding: 15px;
            border-radius: 8px;
        }

    </style>

</head>


<body>

<div class="container">

    <h1>🤖 LangGraph AI Developer</h1>

    <form method="POST">

        <label>
            <b>Enter your coding task:</b>
        </label>

        <br><br>

        <textarea
            name="task"
            placeholder="Example: Write a Python program to check whether a number is prime."
            required
        ></textarea>

        <br>

        <button type="submit">
            Generate
        </button>

    </form>


    {% if report %}

        <hr>

        <h2>Generated Result</h2>

        <pre>{{ report }}</pre>

    {% endif %}


</div>

</body>

</html>
"""


# =========================================================
# 8. FLASK ROUTE
# =========================================================

@app.route("/", methods=["GET", "POST"])
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
                            HumanMessage(content=task)
                        ],
                        "next_step": None,
                        "code": None,
                        "report": None
                    }
                )

                report = result.get(
                    "report",
                    "No report generated."
                )

            except Exception as e:

                report = (
                    "Application Error:\n\n"
                    f"{traceback.format_exc()}"
                )

    return render_template_string(
        HTML,
        report=report
    )


# =========================================================
# 9. RUN FLASK
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
