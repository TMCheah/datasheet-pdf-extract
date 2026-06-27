import os
import io
from typing import List
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from PIL import Image


# ==============================
# 0. Other Setup
# ==============================

class CapacitorSpecs(BaseModel):
    component_value: str = Field(description="The capacitance value (e.g., 10uF, 100nF, 4.7pF)")
    tolerance: str = Field(description="The tolerance level (e.g., ±10%, ±5%, J, K)")

# ==============================
# 1. Setup Model
# ==============================
# Initialize the base model once

llm_model = "gemma4:12b"
llm = ChatOllama(model=llm_model, temperature=0.0, timeout=120.0, streaming=False, reasoning = True)
structured_llm = llm.with_structured_output(CapacitorSpecs, method="json_schema")

# ==============================
# 2. Graph State
# ==============================
class State(TypedDict):
    path: str
    pdfs: List[str]
    file_count: int
    info: dict
    save_status: str
    error: str

# ==============================
# 3. AI Tools (Graph Nodes)
# ==============================
# Initialize the base model once

def list_pdfs_in_directory(state: State):
    """Scans a local directory path and returns a list of absolute file paths for all PDFs found inside it."""
    
    state["error"] = ""
    
    clean_path = state["path"].strip().strip("'\"")
    if not os.path.exists(clean_path):
        return {"pdfs" : [], "error" : f"Error: Path '{clean_path}' does not exist."}
    
    pdf_list = [os.path.abspath(os.path.join(clean_path, f)) for f in os.listdir(clean_path) if f.lower().endswith('.pdf')]
    
    print(pdf_list)
    
    return {"pdfs" : pdf_list, "file_count" : len(pdf_list)}


def extract_data_from_pdf(state: State):
    """Reads a single PDF file path, and extracts capacitor values and tolerance specs."""
    from pypdf import PdfReader
    
    state["error"] = ""
    print(state["file_count"])
    print(f"Extracting {state["pdfs"][0]} ...")
    
    try:
        reader = PdfReader(state["pdfs"][0])
        text = "".join([page.extract_text() for page in reader.pages[:1] if page.extract_text()])[:1000]
        
        prompt = f"Carefully read this technical component datasheet text snippet. Extract the type of component, exact manufacturer part number, values, tolerances, and packaging data strictly matching the structure format.\n\n Text content:\n{text}"
        
        result: ComponentSpecs = structured_llm.invoke(prompt)
        
        print(result)
        print(result.model_dump())
        
        return {"info" : result.model_dump()}
    except Exception as e:
        return {"info" : {}, "error" : f"Error reading {file_path}: {str(e)}"}

def save_to_sqlite(state: State):
    """Save the capaccitor information into database"""
    #simple testing only. populate the sqlite db here
    
    state["error"] = ""
    
    print(state["info"])
    print(f"{state["pdfs"][0]} is saved.")
    
    return {"save_status" : f"{state["pdfs"].pop(0)} is saved."}


def count_pdf_list(state: State):
    """Count the pdf left in the list """
    if len(state["pdfs"]) == 0:
        state["file_count"] = 0
        return "False"
    
    state["file_count"] = len(state["pdfs"])
    
    print(f"file count remaining: {len(state["pdfs"])}")
    
    return "True"



# ==============================
# 4. Build workflow
# ==============================

#list of pdf    ->  list_pdfs_in_directory
#extract info   ->  extract_data_from_pdf
#save pdf       ->  save_to_sqlite
#check count    ->  count_pdf_list
#loop back

workflow = StateGraph(State)

# Add nodes
workflow.add_node("List_Path", list_pdfs_in_directory)
workflow.add_node("Read_Pdf", extract_data_from_pdf)
workflow.add_node("Save_Info", save_to_sqlite)

# Add edges to connect nodes
workflow.add_edge(START, "List_Path")
workflow.add_edge("List_Path", "Read_Pdf")
workflow.add_edge("Read_Pdf", "Save_Info")
workflow.add_conditional_edges("Save_Info", count_pdf_list, {"False": END, "True" : "Read_Pdf"})

# Compile
chain = workflow.compile()

# Show workflow
image_bytes = chain.get_graph().draw_mermaid_png()

img = Image.open(io.BytesIO(image_bytes))
img.show()


# ==============================
# Testing Setup
# ==============================

test_folder = "C:\\Users\\User\\Desktop\\RandomTesting\\pdf" 
user_query = f"Check the folder '{test_folder}', read the PDFs inside, and give me their specs."

print("Start Graph Flow...")

state = chain.invoke({"path": test_folder})


