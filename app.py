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
# 2. GEMINI LLM
# =========================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key
)


# =========================================================
# 3. STATE
# =========================================================

class CrewState(TypedDict):
    messages: List[BaseMessage]

    next_step: Optional[str]

    code: Optional[str]

    test_cases: Optional[str]

    execution_result: Optional[str]

    manager_decision: Optional[str]

    report: Optional[str]

    # Architecture information
    task_input: Optional[str]
    developer_input: Optional[str]
    developer_output: Optional[str]

    tester_input: Optional[str]
    tester_output: Optional[str]

    manager_input: Optional[str]
    manager_output: Optional[str]

    archiver_input: Optional[str]
    archiver_output: Optional[str]


# =========================================================
# 4. TOOLS
# =========================================================

@tool
def run_python_code(code: str) -> str:
    """
    Execute Python code and return the output.
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
            "Execution Error:\n\n"
            + traceback.format_exc()
        )

    finally:

        sys.stdout = old_stdout

    if result.strip():

        return result.strip()

    return "Success (no terminal output)"


# =========================================================
# TEST CASE GENERATOR
# =========================================================

@tool
def generate_test_cases(task_description: str) -> str:
    """
    Generate Python-specific test scenarios.
    """

    prompt = f"""
You are a Senior Python QA Engineer.

Generate 3 to 5 highly specific test scenarios
for this Python coding task:

{task_description}

Requirements:

1. Include normal cases.
2. Include edge cases.
3. Include invalid input cases when appropriate.
4. Include performance cases when appropriate.
5. All scenarios must be specific to Python.
6. Do NOT use Java-specific terms such as
   InvalidArgumentException or BigInt.
7. Clearly mention:
   - Input
   - Expected Output
   - Objective

Return as a numbered list.
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

        return f"Test generation failed: {str(e)}"


# =========================================================
# 5. TASK INPUT AGENT
# =========================================================

def task_input_node(state: CrewState):

    task = state["messages"][-1].content

    return {

        "task_input": task,

        "developer_input": task,

        "next_step": "developer"
    }


# =========================================================
# 6. DEVELOPER AGENT
# =========================================================

def real_time_developer(state: CrewState):

    task = state["task_input"]

    prompt = f"""
You are an expert Python Developer.

Solve this coding task:

{task}

Requirements:

1. Write clean Python code.
2. Make the code executable.
3. Include a useful example execution.
4. Use print statements where appropriate.
5. Return ONLY Python code.
6. Do not return Markdown.
7. Do not use ```python.
"""

    try:

        response = llm.invoke(prompt)

        content = response.content

        if isinstance(content, list):

            code = "\n".join(
                item.get("text", str(item))
                if isinstance(item, dict)
                else str(item)
                for item in content
            )

        else:

            code = str(content)

        code = (
            code.replace("```python", "")
            .replace("```", "")
            .strip()
        )

        return {

            "code": code,

            "developer_output": code,

            "tester_input": code,

            "next_step": "tester"
        }

    except Exception as e:

        error_code = (
            "# Developer Agent Error\n"
            f"print({str(e)!r})"
        )

        return {

            "code": error_code,

            "developer_output": error_code,

            "tester_input": error_code,

            "next_step": "tester"
        }


# =========================================================
# 7. TESTER AGENT
# =========================================================

def real_time_tester(state: CrewState):

    task = state["task_input"]

    code = state["code"]

    # Generate test cases
    test_cases = generate_test_cases.invoke(task)

    # Execute generated code
    execution_result = run_python_code.invoke(
        {
            "code": code
        }
    )

    tester_output = (
        "TEST CASES\n\n"
        f"{test_cases}\n\n"
        "EXECUTION RESULT\n\n"
        f"{execution_result}"
    )

    return {

        "test_cases": test_cases,

        "execution_result": execution_result,

        "tester_output": tester_output,

        "manager_input": tester_output,

        "next_step": "manager"
    }


# =========================================================
# 8. MANAGER AGENT
# =========================================================

def manager_decision_node(state: CrewState):

    execution_result = state["execution_result"]

    if "Execution Error" in execution_result:

        decision = (
            "⚠️ Execution error detected. "
            "Result archived with error information."
        )

    else:

        decision = (
            "✅ Code execution completed successfully. "
            "Proceeding to archive the result."
        )

    return {

        "manager_input": (
            "Tester provided execution result:\n\n"
            + execution_result
        ),

        "manager_decision": decision,

        "manager_output": decision,

        "archiver_input": decision,

        "next_step": "archiver"
    }


# =========================================================
# 9. ARCHIVER AGENT
# =========================================================

def archiver_node(state: CrewState):

    final_report = (
        "ARCHIVED SUCCESSFULLY\n\n"
        "The complete Developer → Tester → Manager "
        "workflow has been completed."
    )

    return {

        "archiver_output": final_report,

        "report": final_report,

        "next_step": "exit"
    }


# =========================================================
# 10. BUILD LANGGRAPH
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
    "manager",
    manager_decision_node
)

workflow.add_node(
    "archiver",
    archiver_node
)


# START → TASK INPUT

workflow.add_edge(
    START,
    "task_input"
)


# TASK INPUT → DEVELOPER

workflow.add_edge(
    "task_input",
    "developer"
)


# DEVELOPER → TESTER

workflow.add_edge(
    "developer",
    "tester"
)


# TESTER → MANAGER

workflow.add_edge(
    "tester",
    "manager"
)


# MANAGER → ARCHIVER

workflow.add_edge(
    "manager",
    "archiver"
)


# ARCHIVER → END

workflow.add_edge(
    "archiver",
    END
)


# Compile

rt_app = workflow.compile()


# =========================================================
# 11. FLASK
# =========================================================

app = Flask(__name__)


# =========================================================
# 12. HTML
# =========================================================

HTML = """

<!DOCTYPE html>

<html>

<head>

<title>Agentic AI - LangGraph Architecture</title>


<style>


* {
    box-sizing: border-box;
}


body {

    margin: 0;

    padding: 30px;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

    background:
        linear-gradient(
            135deg,
            #eef2ff,
            #f8fafc
        );

    color: #1e293b;
}


.container {

    max-width: 1250px;

    margin: auto;
}


/* =====================================================
   HEADER
   ===================================================== */

.header {

    background: white;

    padding: 30px;

    border-radius: 20px;

    text-align: center;

    box-shadow:
        0 10px 30px
        rgba(0,0,0,0.08);

    margin-bottom: 25px;
}


.header h1 {

    margin: 0;

    font-size: 32px;
}


.header p {

    color: #64748b;

    margin-bottom: 0;
}


/* =====================================================
   INPUT
   ===================================================== */

.input-card {

    background: white;

    padding: 25px;

    border-radius: 18px;

    box-shadow:
        0 8px 25px
        rgba(0,0,0,0.07);

    margin-bottom: 30px;
}


textarea {

    width: 100%;

    height: 110px;

    padding: 15px;

    border:
        2px solid #e2e8f0;

    border-radius: 12px;

    font-size: 16px;

    resize: vertical;
}


button {

    margin-top: 15px;

    padding:
        13px 28px;

    border: none;

    border-radius: 10px;

    background:
        #4f46e5;

    color: white;

    font-size: 16px;

    font-weight: bold;

    cursor: pointer;
}


button:hover {

    background:
        #4338ca;
}


/* =====================================================
   ARCHITECTURE
   ===================================================== */

.architecture-title {

    text-align: center;

    margin:
        25px 0;
}


.flow {

    display: flex;

    flex-direction: column;

    align-items: center;

    gap: 12px;
}


.agent-card {

    width: 90%;

    background: white;

    border-radius: 18px;

    padding: 22px;

    box-shadow:
        0 8px 25px
        rgba(0,0,0,0.08);

    border-left:
        7px solid #4f46e5;

    transition:
        transform 0.2s;
}


.agent-card:hover {

    transform:
        translateY(-3px);
}


.agent-header {

    display: flex;

    justify-content:
        space-between;

    align-items: center;

    margin-bottom: 15px;
}


.agent-name {

    font-size: 22px;

    font-weight: bold;
}


.agent-number {

    background:
        #4f46e5;

    color: white;

    padding:
        7px 12px;

    border-radius: 20px;

    font-size: 13px;
}


.io-grid {

    display: grid;

    grid-template-columns:
        1fr 1fr;

    gap: 15px;
}


.io-box {

    padding: 15px;

    border-radius: 12px;

    background:
        #f8fafc;

    border:
        1px solid #e2e8f0;
}


.io-title {

    font-weight: bold;

    margin-bottom: 8px;
}


.io-content {

    white-space: pre-wrap;

    word-break: break-word;

    font-family:
        Consolas,
        monospace;

    font-size: 13px;

    max-height: 280px;

    overflow-y: auto;
}


.arrow {

    font-size: 32px;

    color: #4f46e5;

    font-weight: bold;
}


/* =====================================================
   FINAL RESULT
   ===================================================== */

.final-result {

    margin-top: 35px;

    background: white;

    padding: 25px;

    border-radius: 18px;

    box-shadow:
        0 8px 30px
        rgba(0,0,0,0.1);
}


.final-title {

    font-size: 25px;

    font-weight: bold;

    margin-bottom: 20px;
}


.result-section {

    margin-bottom: 20px;

    padding: 18px;

    background:
        #f8fafc;

    border-radius: 12px;
}


.result-section h3 {

    margin-top: 0;
}


pre {

    white-space: pre-wrap;

    word-break: break-word;

    font-family:
        Consolas,
        monospace;

    font-size: 14px;
}


/* =====================================================
   STATUS
   ===================================================== */

.status {

    text-align: center;

    margin-top: 20px;

    padding: 12px;

    background:
        #ecfdf5;

    color:
        #047857;

    border-radius: 10px;

    font-weight: bold;
}


/* =====================================================
   MOBILE
   ===================================================== */

@media(max-width: 700px) {

    body {

        padding: 15px;
    }


    .agent-card {

        width: 100%;
    }


    .io-grid {

        grid-template-columns:
            1fr;
    }


    .agent-header {

        flex-direction:
            column;

        align-items:
            flex-start;

        gap: 10px;
    }

}


</style>

</head>


<body>


<div class="container">


<!-- ===================================================
     HEADER
     =================================================== -->

<div class="header">

    <h1>
        🤖 Agentic AI Developer System
    </h1>

    <p>
        Multi-Agent Workflow powered by
        LangGraph + Gemini
    </p>

</div>


<!-- ===================================================
     USER INPUT
     =================================================== -->

<div class="input-card">

    <h2>👤 User Task</h2>

    <form method="POST">

        <textarea
            name="task"
            placeholder="Example: Write a Python program to check whether a number is prime."
            required
        ></textarea>

        <br>

        <button type="submit">
            🚀 Run Agent Workflow
        </button>

    </form>

</div>


{% if data %}


<!-- ===================================================
     ARCHITECTURE TITLE
     =================================================== -->

<h2 class="architecture-title">

    🔗 Agentic AI Architecture & Execution Flow

</h2>


<div class="flow">


<!-- ===================================================
     USER
     =================================================== -->

<div class="agent-card">

    <div class="agent-header">

        <div class="agent-name">
            👤 USER
        </div>

        <div class="agent-number">
            INPUT
        </div>

    </div>


    <div class="io-grid">

        <div class="io-box">

            <div class="io-title">
                📥 Input
            </div>

            <div class="io-content">
{{ data.task }}
            </div>

        </div>


        <div class="io-box">

            <div class="io-title">
                📤 Output
            </div>

            <div class="io-content">
Task sent to Task Input Agent
            </div>

        </div>

    </div>

</div>


<div class="arrow">
    ↓
</div>


<!-- ===================================================
     TASK INPUT AGENT
     =================================================== -->

<div class="agent-card">

    <div class="agent-header">

        <div class="agent-name">
            🟦 TASK INPUT AGENT
        </div>

        <div class="agent-number">
            NODE 1
        </div>

    </div>


    <div class="io-grid">

        <div class="io-box">

            <div class="io-title">
                📥 Input
            </div>

            <div class="io-content">
{{ data.task }}
            </div>

        </div>


        <div class="io-box">

            <div class="io-title">
                📤 Output
            </div>

            <div class="io-content">
Task prepared and routed to Developer Agent
            </div>

        </div>

    </div>

</div>


<div class="arrow">
    ↓
</div>


<!-- ===================================================
     DEVELOPER
     =================================================== -->

<div class="agent-card">

    <div class="agent-header">

        <div class="agent-name">
            👨‍💻 DEVELOPER AGENT
        </div>

        <div class="agent-number">
            NODE 2
        </div>

    </div>


    <div class="io-grid">

        <div class="io-box">

            <div class="io-title">
                📥 Input
            </div>

            <div class="io-content">
{{ data.task }}
            </div>

        </div>


        <div class="io-box">

            <div class="io-title">
                📤 Output — Generated Python Code
            </div>

            <div class="io-content">{{ data.code }}</div>

        </div>

    </div>

</div>


<div class="arrow">
    ↓
</div>


<!-- ===================================================
     TESTER
     =================================================== -->

<div class="agent-card">

    <div class="agent-header">

        <div class="agent-name">
            🧪 TESTER AGENT
        </div>

        <div class="agent-number">
            NODE 3
        </div>

    </div>


    <div class="io-grid">


        <div class="io-box">

            <div class="io-title">
                📥 Input — Developer Code
            </div>

            <div class="io-content">{{ data.code }}</div>

        </div>


        <div class="io-box">

            <div class="io-title">
                📤 Output — Test Results
            </div>

            <div class="io-content">{{ data.tester_output }}</div>

        </div>


    </div>

</div>


<div class="arrow">
    ↓
</div>


<!-- ===================================================
     MANAGER
     =================================================== -->

<div class="agent-card">

    <div class="agent-header">

        <div class="agent-name">
            🧠 MANAGER AGENT
        </div>

        <div class="agent-number">
            NODE 4
        </div>

    </div>


    <div class="io-grid">


        <div class="io-box">

            <div class="io-title">
                📥 Input
            </div>

            <div class="io-content">
{{ data.manager_input }}
            </div>

        </div>


        <div class="io-box">

            <div class="io-title">
                📤 Decision
            </div>

            <div class="io-content">
{{ data.manager_decision }}
            </div>

        </div>


    </div>

</div>


<div class="arrow">
    ↓
</div>


<!-- ===================================================
     ARCHIVER
     =================================================== -->

<div class="agent-card">

    <div class="agent-header">

        <div class="agent-name">
            🗄️ ARCHIVER AGENT
        </div>

        <div class="agent-number">
            NODE 5
        </div>

    </div>


    <div class="io-grid">


        <div class="io-box">

            <div class="io-title">
                📥 Input
            </div>

            <div class="io-content">
{{ data.archiver_input }}
            </div>

        </div>


        <div class="io-box">

            <div class="io-title">
                📤 Output
            </div>

            <div class="io-content">
{{ data.archiver_output }}
            </div>

        </div>


    </div>

</div>


</div>


<!-- ===================================================
     FINAL RESULT
     =================================================== -->

<div class="final-result">


    <div class="final-title">
        🏁 FINAL RESULT
    </div>


    <div class="status">
        ✅ Complete Agent Workflow Executed Successfully
    </div>


    <div class="result-section">

        <h3>
            💻 Generated Python Code
        </h3>

        <pre>{{ data.code }}</pre>

    </div>


    <div class="result-section">

        <h3>
            ▶️ Execution Output
        </h3>

        <pre>{{ data.execution_result }}</pre>

    </div>


    <div class="result-section">

        <h3>
            🧪 QA Test Scenarios
        </h3>

        <pre>{{ data.test_cases }}</pre>

    </div>


    <div class="result-section">

        <h3>
            🧠 Manager Decision
        </h3>

        <pre>{{ data.manager_decision }}</pre>

    </div>


</div>


{% endif %}


</div>


</body>

</html>

"""


# =========================================================
# 13. FLASK ROUTE
# =========================================================

@app.route("/", methods=["GET", "POST"])
def home():

    data = None

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

                        "test_cases": None,

                        "execution_result": None,

                        "manager_decision": None,

                        "report": None,

                        "task_input": None,

                        "developer_input": None,

                        "developer_output": None,

                        "tester_input": None,

                        "tester_output": None,

                        "manager_input": None,

                        "manager_output": None,

                        "archiver_input": None,

                        "archiver_output": None
                    }
                )


                data = {

                    "task":
                        result.get(
                            "task_input",
                            task
                        ),

                    "code":
                        result.get(
                            "code",
                            "No code generated."
                        ),

                    "tester_output":
                        result.get(
                            "tester_output",
                            "No tester output."
                        ),

                    "test_cases":
                        result.get(
                            "test_cases",
                            "No test cases generated."
                        ),

                    "execution_result":
                        result.get(
                            "execution_result",
                            "No execution result."
                        ),

                    "manager_input":
                        result.get(
                            "manager_input",
                            "No manager input."
                        ),

                    "manager_decision":
                        result.get(
                            "manager_decision",
                            "No manager decision."
                        ),

                    "archiver_input":
                        result.get(
                            "archiver_input",
                            "No archiver input."
                        ),

                    "archiver_output":
                        result.get(
                            "archiver_output",
                            "No archiver output."
                        )
                }


            except Exception as e:

                data = {

                    "task": task,

                    "code":
                        "Application Error",

                    "tester_output":
                        traceback.format_exc(),

                    "test_cases":
                        "",

                    "execution_result":
                        "",

                    "manager_input":
                        "",

                    "manager_decision":
                        "",

                    "archiver_input":
                        "",

                    "archiver_output":
                        ""
                }


    return render_template_string(
        HTML,
        data=data
    )


# =========================================================
# 14. START APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
