import streamlit as st
import base64
import tempfile
import os
import google.generativeai as genai
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# ==========================================
# 1. Define the LangGraph State
# ==========================================
class PaperState(TypedDict):
    api_key: str
    marks: int
    subject: str
    grade_class: str
    q_type: str
    comments: str
    file_data: list  # Store base64 encoded files with mime types
    generated_paper: str

# ==========================================
# 2. Define Graph Nodes
# ==========================================
def generate_question_paper(state: PaperState):
    """Node that calls Gemini to generate the paper."""
    
    # Initialize the LLM with the user's key
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash", 
        api_key=state["api_key"]
    )
    
    # Construct the prompt text
    prompt_text = f"""
    You are an expert teacher. Generate a question paper strictly based on the provided images (if any) and the following criteria:
    - Subject: {state['subject']}
    - Class/Grade: {state['grade_class']}
    - Total Marks: {state['marks']}
    - Question Type: {state['q_type']}
    - Additional Instructions: {state['comments']}
    
    Format the output clearly with sections, question numbers, and marks per question.
    """
    
    # Build the message content handling both text and multiple images
    message_content = [{"type": "text", "text": prompt_text}]
    
    for file_info in state.get("file_data", []):
        mime_type = file_info["mime_type"]
        
        if mime_type == "application/pdf":
            if "file_uri" in file_info:
                message_content.append({
                    "type": "media",
                    "mime_type": mime_type,
                    "file_uri": file_info["file_uri"]
                })
            else:
                message_content.append({
                    "type": "media",
                    "mime_type": mime_type,
                    "data": file_info.get("data", "")
                })
        elif mime_type.startswith("image/"):
            b64_data = file_info["data"]
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}
            })
        
    message = HumanMessage(content=message_content)
    
    # Generate the response
    response = llm.invoke([message])
    
    content = response.content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        text_content = "\n".join(text_parts)
    else:
        text_content = str(content)
        
    return {"generated_paper": text_content}

# ==========================================
# 3. Build the Workflow Graph
# ==========================================
workflow = StateGraph(PaperState)
workflow.add_node("generator", generate_question_paper)
workflow.set_entry_point("generator")
workflow.add_edge("generator", END)
app_graph = workflow.compile()

# ==========================================
# 4. Streamlit UI
# ==========================================
st.set_page_config(page_title="AI Question Paper Generator", layout="wide")
st.title("📄 AI Question Paper Generator")

# Sidebar for API Key
with st.sidebar:
    st.header("Authentication")
    user_api_key = st.text_input("Enter your Google Gemini API Key:", type="password")
    st.caption("Get your key from [Google AI Studio](https://aistudio.google.com/).")

# Main Form
with st.form("paper_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        subject = st.text_input("Subject (e.g., Physics, History)")
    with col2:
        grade_class = st.text_input("Class/Grade (e.g., 10th, University)")
    with col3:
        marks = st.number_input("Total Marks", min_value=1, max_value=200, value=50)
        
    q_type = st.selectbox(
        "Question Type Format", 
        ["All MCQ", "All One Word", "Mixed / Balanced"]
    )
    
    uploaded_files = st.file_uploader(
        "Upload Source Material (Images & PDFs)", 
        type=["png", "jpg", "jpeg", "pdf"], 
        accept_multiple_files=True
    )
    
    comments = st.text_area("Optional Comments/Specific Instructions", placeholder="e.g., Make the questions high difficulty, focus on application-based concepts...")
    
    submit_button = st.form_submit_button("Generate Question Paper")

# ==========================================
# 5. Execution Logic
# ==========================================
if submit_button:
    if not user_api_key:
        st.error("Please enter your Gemini API Key in the sidebar.")
    elif not subject or not grade_class:
        st.error("Please fill in the Subject and Class fields.")
    else:
        with st.spinner("Analyzing context and generating paper..."):
            
            # Convert uploaded files for the API (PDFs need File API)
            file_data_list = []
            if uploaded_files:
                genai.configure(api_key=user_api_key)
                for file in uploaded_files:
                    if file.type == "application/pdf":
                        # Upload PDF to Google GenAI File API
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(file.read())
                            tmp_path = tmp.name
                        
                        uploaded_gfile = genai.upload_file(tmp_path, mime_type="application/pdf")
                        file_data_list.append({
                            "mime_type": file.type,
                            "file_uri": uploaded_gfile.uri
                        })
                        os.remove(tmp_path)
                    else:
                        file_data_list.append({
                            "mime_type": file.type,
                            "data": base64.b64encode(file.read()).decode("utf-8")
                        })
            
            # Prepare inputs for LangGraph
            initial_state = {
                "api_key": user_api_key,
                "marks": marks,
                "subject": subject,
                "grade_class": grade_class,
                "q_type": q_type,
                "comments": comments,
                "file_data": file_data_list
            }
            
            # Run the graph
            try:
                result = app_graph.invoke(initial_state)
                
                st.success("Paper Generated Successfully!")
                st.markdown("---")
                st.markdown(result["generated_paper"])
                
                # Create a Word Document from the Markdown
                from docx import Document
                import io
                import re
                
                def create_docx(text):
                    doc = Document()
                    for line in text.split('\n'):
                        line = line.strip()
                        if not line:
                            continue
                            
                        # Handle simple Markdown Headings
                        if line.startswith('# '):
                            doc.add_heading(line[2:], level=1)
                        elif line.startswith('## '):
                            doc.add_heading(line[3:], level=2)
                        elif line.startswith('### '):
                            doc.add_heading(line[4:], level=3)
                        elif line.startswith('#### '):
                            doc.add_heading(line[5:], level=4)
                        else:
                            p = doc.add_paragraph()
                            # Basic bold parsing for **text**
                            parts = re.split(r'(\*\*.*?\*\*)', line)
                            for part in parts:
                                if part.startswith('**') and part.endswith('**'):
                                    run = p.add_run(part[2:-2])
                                    run.bold = True
                                else:
                                    p.add_run(part)
                    
                    buffer = io.BytesIO()
                    doc.save(buffer)
                    buffer.seek(0)
                    return buffer

                docx_buffer = create_docx(result["generated_paper"])
                
                # Add a download button
                st.download_button(
                    label="Download Paper as Word (.docx)",
                    data=docx_buffer,
                    file_name=f"{subject}_paper.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            except Exception as e:
                st.error(f"An error occurred: {e}")
