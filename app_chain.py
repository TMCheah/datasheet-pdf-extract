import os
from typing import List
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# ==========================================
# 1. THE TOOLS (Slightly modified to guide the tiny model)
# ==========================================

@tool
def list_pdfs_in_directory(directory_path: str) -> List[str]:
    """Scans a local directory path and returns a list of absolute file paths for all PDFs found inside it."""
    clean_path = directory_path.strip().strip("'\"")
    if not os.path.exists(clean_path):
        return [f"Error: Path '{clean_path}' does not exist."]
    return [os.path.abspath(os.path.join(clean_path, f)) for f in os.listdir(clean_path) if f.lower().endswith('.pdf')]


@tool
def extract_data_from_pdf(file_path: str) -> str:
    """Reads a single PDF file path, and extracts capacitor values and tolerance specs."""
    # Note: We NO LONGER call structured_llm here. We just return the raw text, 
    # and let the autonomous agent read it and structure it itself!
    from pypdf import PdfReader
    try:
        reader = PdfReader(file_path)
        text = "".join([page.extract_text() for page in reader.pages[:1] if page.extract_text()])
        return f"Content of {os.path.basename(file_path)}:\n{text[:1500]}"
    except Exception as e:
        return f"Error reading {file_path}: {str(e)}"


# ==========================================
# 2. SETTING UP THE FULLY AUTONOMOUS AGENT
# ==========================================

# Initialize the base model once
llm_model = "qwen3.5:0.8b"
llm = ChatOllama(model=llm_model, temperature=0.0, reasoning = True)

# Define the tools array
tools = [list_pdfs_in_directory, extract_data_from_pdf]

# System instructions MUST be incredibly strict for a 0.8B model to be autonomous
system_prompt = """You are an autonomous data extraction expert. You must follow these steps:
    1. Look at the folder path provided by the user. Call 'list_pdfs_in_directory' to see the files.
    2. Review the tool output. For EACH file path returned, call 'extract_data_from_pdf' to read it. 
    3. Keep track of which files you have read. Do not read the same file twice.
    4. Once you have read all files, write a final summary listing the Component Value and Tolerance for each file, then stop.
    """

agent = create_agent(
    model=llm, 
    tools=tools, 
    system_prompt=system_prompt,
    debug=False
)


# ==========================================
# 3. RUN IT
# ==========================================
if __name__ == "__main__":
    
    # Change this to whatever test folder path you actually have locally
    test_folder = "C:\\Users\\User\\Desktop\\RandomTesting\\pdf" 
    
    user_query = f"Check the folder '{test_folder}', read the PDFs inside, and give me their specs."
    
    print("Starting Fully Autonomous Agent...")
    response = agent.invoke({"messages": [{"role": "user", "content": f"{user_query}"}]})
    print(response["messages"][-1].additional_kwargs["reasoning_content"])
    
    print("\n================ FINAL REPORT ================")
    print(response["messages"][-1].content)