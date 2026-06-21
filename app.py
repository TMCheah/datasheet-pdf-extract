import os
from typing import List
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from pypdf import PdfReader
from pydantic import BaseModel, Field


# Define exactly what data you want to extract from the datasheet
class CapacitorSpecs(BaseModel):
    component_value: str = Field(description="The capacitance value (e.g., 10uF, 100nF, 4.7pF)")
    tolerance: str = Field(description="The tolerance level (e.g., ±10%, ±5%, J, K)")

# ==========================================
# 1. DEFINE OUR LOCAL TOOLS
# ==========================================

@tool
def list_pdfs_in_directory(directory_path: str) -> List[str]:
    """
    Scans a local directory path provided by the user and returns a list 
    of absolute file paths for all PDF files found inside it.
    """
    clean_path = directory_path.strip().strip("'\"")
    if not os.path.exists(clean_path):
        return [f"Error: The directory path '{clean_path}' does not exist."]
    
    pdf_files = []
    for file in os.listdir(clean_path):
        if file.lower().endswith('.pdf'):
            full_path = os.path.abspath(os.path.join(clean_path, file))
            pdf_files.append(full_path)
            
    if not pdf_files:
        return [f"No PDF files found in '{clean_path}'."]
    return pdf_files


@tool
def extract_data_from_pdf(file_path: str) -> str:
    """
    Reads a single PDF file given its absolute file path, extracts text content, 
    and asks the LLM to pull out specific capacitor specifications.
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
        
    try:
        reader = PdfReader(file_path)
        extracted_text = ""
        # 0.8B models have a tight context window, so read only the first 2-3 pages 
        # where the main specifications table usually lives.
        for page in reader.pages[:1]: 
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
        
        # Clean text slightly to save context tokens
        cleaned_text = " ".join(extracted_text.split())[:3000]
        #print(cleaned_text)
        
        if not cleaned_text.strip():
            return f"[{os.path.basename(file_path)}]: No text could be read."

        # Prompt the model directly with the data chunk
        prompt = f"From the following component text snippet, extract the component value and tolerance:\n\n{cleaned_text}"
        #print(prompt)
        
        llm = base_llm.with_structured_output(CapacitorSpecs, method="json_schema")
        
        # The result will be a pure Pydantic object matching our CapacitorSpecs schema!
        #result: CapacitorSpecs = structured_llm.invoke(prompt)
        result: CapacitorSpecs = llm.invoke(prompt)
        print(result)
        
        # Return a nice, clean one-line string back to your main workflow loop
        return f"[{os.path.basename(file_path)}] Value: {result.component_value} | Tolerance: {result.tolerance}"

    except Exception as e:
        return f"Error processing {os.path.basename(file_path)}: {str(e)}"


# Map string tool names to actual functions for easy execution
tool_map = {
    "list_pdfs_in_directory": list_pdfs_in_directory,
    "extract_data_from_pdf": extract_data_from_pdf
}

# ==========================================
# 2. INITIALIZE OLLAMA QWEN MODEL WITH TOOLS
# ==========================================

llm_model = "llama3.1:8b"

# Using low temperature to make the 0.8B model stick strictly to the facts
base_llm = ChatOllama(model = llm_model, temperature = 0.0) 

# ==========================================
# 3. MANUAL AGENT LOOP (Guarantees execution for 0.8B)
# ==========================================

def run_pdf_workflow(user_prompt: str):
    print(f"\n[User Request]: {user_prompt}\n")
    
    # 1. We keep it simple with just ONE human message for the tool call
    messages = [
        HumanMessage(content=f"You are a helpful assistant. Please use the tools to fulfill this request: {user_prompt}")
    ]
    
    llm = base_llm.bind_tools([list_pdfs_in_directory, extract_data_from_pdf])
    
    print("🤖 Agent deciding first action...")
    ai_msg = llm.invoke(messages)
    print(ai_msg)
    
    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            print(f"🛠️ Executing Tool: {tool_call['name']}")
            tool_fn = tool_map[tool_call['name']]
            tool_output = tool_fn.invoke(tool_call['args'])
            
            # Inside your run_pdf_workflow function...
            if isinstance(tool_output, list) and len(tool_output) > 0 and not tool_output[0].startswith("Error"):
                print(f"📂 Found {len(tool_output)} PDF files. Extracting specifications...\n")
                
                print("================ DATA EXTRACTION REPORT ================")
                # Loop through every single file automatically
                for pdf_path in tool_output:
                    # This calls the updated function that returns the clean one-liner string
                    clean_spec_line = extract_data_from_pdf.invoke({"file_path": pdf_path})
                    print(clean_spec_line)
                print("========================================================")
                return
            else:
                print(f"❌ Error or No files: {tool_output}")
    else:
        print("⚠️ Model did not trigger tool call directly. Prompt output:")
        print(ai_msg.content)
        
# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    # Change this to whatever test folder path you actually have locally
    test_folder = "C:/Users/User/Desktop/RandomTesting/pdf" 
    
    # Create sample dummy folder for testing if it doesn't exist
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
        print(f"Created a dummy folder at '{test_folder}'. Put your PDFs inside it and run again!")
    
    prompt = f"Look into the directory '{test_folder}', read any PDFs inside it, and tell me what they are about."
    run_pdf_workflow(prompt)