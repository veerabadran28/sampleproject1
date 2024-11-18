Sure, here's a Python function to check if a given string is a palindrome:

```python
def is_palindrome(s):
    """
    Checks if a given string is a palindrome.
    
    Args:
        s (str): The string to be checked.
        
    Returns:
        bool: True if the string is a palindrome, False otherwise.
    """
    # Remove non-alphanumeric characters and convert to lowercase
    clean_s = ''.join(char.lower() for char in s if char.isalnum())
    
    # Check if the cleaned string is equal to its reverse
    return clean_s == clean_s[::-1]
```

This function takes a string `s` as input and returns `True` if the string is a palindrome, and `False` otherwise.

Here's how it works:

1. The function first removes all non-alphanumeric characters (e.g., spaces, punctuation marks) from the input string `s` using a list comprehension and the `isalnum()` method. It also converts all characters to lowercase using the `lower()` method. The resulting cleaned string is stored in the `clean_s` variable.

2. The function then checks if `clean_s` is equal to its reverse, which is obtained using the slice notation `[::-1]`. This reverses the string by taking every element from start to end, but in reverse order.

3. If `clean_s` is equal to its reverse, the function returns `True`, indicating that the input string is a palindrome. Otherwise, it returns `False`.

Here are some examples of how to use the `is_palindrome()` function:

```python
print(is_palindrome("racecar"))     # Output: True
print(is_palindrome("A man a plan a canal Panama"))  # Output: True
print(is_palindrome("hello"))       # Output: False
print(is_palindrome("Python"))      # Output: False
```

Note that the function considers only alphanumeric characters when checking for palindromes, ignoring non-alphanumeric characters and case sensitivity.



====================================

import streamlit as st
import os
import pandas as pd
import base64
from PyPDF2 import PdfReader
import re
from datetime import datetime
import boto3
import json
from io import StringIO, BytesIO
from typing import Literal
import time
from collections import deque

# Configuration Constants
BANKS = ["Lloyds", "Santander"]
SUB_TABS = [
    "Select a file",
    "Display File Content",
    "Short Summary",
    "Detailed Summary"
]

BUCKET_NAME = 'myawstests3buckets1'
BASE_PATH = 'financebenchmarking1'

BANK_PATHS = {
    'Lloyds': {
        'base': f'{BASE_PATH}/lloyds/2024',
        'folders': ['quarter1', 'quarter2_halfyear', 'quarter3']
    },
    'Santander': {
        'base': f'{BASE_PATH}/santander/2024',
        'folders': ['quarter1', 'halfyear', 'quarter2']
    }
}

# Initialize AWS clients
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

s3 = boto3.client('s3')

# Custom message container styles
def message_container(message: str, is_user: bool = False, key: str = None):
    if is_user:
        avatar_url = "🧑‍💼"
        background_color = "linear-gradient(to right, #DCF2F1, #e5e5f7)"
        align_message = "flex-end"
        message_color = "#000000"
    else:
        avatar_url = "🤖"
        background_color = "linear-gradient(to right, #F5F5F5, #e5e5f7)"
        align_message = "flex-start"
        message_color = "#000000"

    st.markdown(
        f"""
        <div style="
            display: flex;
            align-items: flex-start;
            margin-bottom: 1rem;
            justify-content: {align_message};
            gap: 1rem;
            ">
            <div style="
                min-width: 40px;
                height: 40px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 20px;
                ">
                {avatar_url}
            </div>
            <div style="
                background: {background_color};
                padding: 1rem;
                border-radius: 0.5rem;
                max-width: 80%;
                color: {message_color};
                ">
                {message}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

class ChatMessage:
    def __init__(self, role: Literal["user", "assistant"], content: str):
        self.role = role
        self.content = content
        self.timestamp = time.time()

def prepare_prompt(prompt, context):
    return f"""
    Context: {context}

    Question: {prompt}

    Please provide a detailed answer based solely on the provided context.
    Include specific page numbers and source references for any information used.
    Format your response professionally with clear sections.
    If the question is not within the scope of the context, notify the user that the requested information cannot be provided.

    Make sure to cite specific parts of the document.
    """

def get_claude_response_up(prompt, context=""):
    enhanced_prompt = prepare_prompt(prompt, context)
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8191,
        "messages": [
            {
                "role": "user",
                "content": enhanced_prompt
            }
        ],
        "temperature": 0.7
    })
    print("Inside Claude Response UP")
    try:
        response = bedrock.invoke_model(
            modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
            body=body
        )
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
    except Exception as e:
        st.error(f"Error calling Claude: {str(e)}")
        return None

def get_claude_response(prompt, context=""):
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8191,
        "messages": [
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {prompt}"
            }
        ],
        "temperature": 0.7
    })
    print("Inside Claude Response Summary")
    try:
        response = bedrock.invoke_model(
            modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
            body=body
        )
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
    except Exception as e:
        st.error(f"Error calling Claude: {str(e)}")
        return None

def get_s3_files(prefix):
    """Get files from S3 bucket"""
    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
        return [obj['Key'] for obj in response.get('Contents', [])
                if obj['Key'] != prefix and (obj['Key'].endswith('.pdf') or obj['Key'].endswith('.xlsx'))]
    except Exception as e:
        st.error(f"Error accessing S3: {str(e)}")
        return []

def read_file_from_s3(file_key):
    """Read file content from S3"""
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=file_key)
        return BytesIO(response['Body'].read())
    except Exception as e:
        st.error(f"Error reading file from S3: {str(e)}")
        return None

def extract_text_from_pdf(file_content):
    """Extract text content from PDF file"""
    pdf = PdfReader(file_content)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

def extract_text_from_excel(file_content):
    """Extract text content from Excel file"""
    dfs = pd.read_excel(file_content, sheet_name=None)
    text = StringIO()
    for sheet_name, df in dfs.items():
        text.write(f"\nSheet: {sheet_name}\n")
        text.write(df.to_string())
    return text.getvalue()

def get_document_content(file_key):
    """Get content from either PDF or Excel file in S3"""
    file_content = read_file_from_s3(file_key)
    if not file_content:
        return ""

    if file_key.endswith('.pdf'):
        return extract_text_from_pdf(file_content)
    elif file_key.endswith(('.xlsx', '.xls')):
        return extract_text_from_excel(file_content)
    return ""

def generate_document_summary(file_key, content):
    """Generate both short and detailed summaries using Claude"""
    short_prompt = "Provide a brief 2-3 sentence summary of the main points in this document."
    short_summary = get_claude_response(short_prompt, content)
    #short_summary = ""

    detailed_prompt = "Provide a detailed summary of this document, including key findings, important metrics, and notable trends. Format the response with appropriate headers and bullet points. Also generate commentary related to opportunities/threats to banking industry in next quartet/year based linked to macro-economic factors, etc. which will aid in decision making."
    detailed_summary = get_claude_response(detailed_prompt, content)
    #detailed_summary = ""

    return short_summary, detailed_summary

def initialize_chat_state():
    """Initialize all session state variables"""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = deque(maxlen=10)
    if "current_file" not in st.session_state:
        st.session_state.current_file = None
    if "document_content" not in st.session_state:
        st.session_state.document_content = {}
    if "summaries" not in st.session_state:
        st.session_state.summaries = {}
    if "current_bank" not in st.session_state:
        st.session_state.current_bank = None
    if "current_period" not in st.session_state:
        st.session_state.current_period = {}  # Store period per bank
    if "selected_file_key" not in st.session_state:
        st.session_state.selected_file_key = None

def get_cached_summaries(file_key, content):
    """Get summaries from cache or generate if not available"""
    if file_key not in st.session_state.summaries:
        print(f"Cache miss - Generating summaries for {file_key}")
        short_summary, detailed_summary = generate_document_summary(file_key, content)
        st.session_state.summaries[file_key] = {
            'short': short_summary,
            'detailed': detailed_summary
        }
    return st.session_state.summaries[file_key]['short'], st.session_state.summaries[file_key]['detailed']

def clear_file_cache(file_key):
    """Clear cache for a specific file"""
    if file_key in st.session_state.document_content:
        del st.session_state.document_content[file_key]
    if file_key in st.session_state.summaries:
        del st.session_state.summaries[file_key]
    if st.session_state.current_file == file_key:
        st.session_state.current_file = None

def get_cached_document_content(file_key):
    """Get document content from cache or generate if not available"""
    if file_key not in st.session_state.document_content:
        print(f"Cache miss - Loading content for {file_key}")
        content = get_document_content(file_key)
        st.session_state.document_content[file_key] = content
    return st.session_state.document_content[file_key]

def display_excel_content(file_key):
    try:
        file_content = read_file_from_s3(file_key)
        xl = pd.ExcelFile(file_content)
        sheet_names = [sheet for sheet in xl.sheet_names if sheet not in ["Cover", "Contents"]]

        st.write(f"Excel file: {os.path.basename(file_key)}")

        sheet_titles = {}
        for sheet in sheet_names:
            df = pd.read_excel(file_content, sheet_name=sheet, header=None)
            if not df.empty:
                sheet_titles[sheet] = df.iloc[0, 0]
            else:
                sheet_titles[sheet] = sheet

        selected_title = st.selectbox("Select data to view:", list(sheet_titles.values()))
        selected_sheet = [sheet for sheet, title in sheet_titles.items() if title == selected_title][0]

        df = pd.read_excel(file_content, sheet_name=selected_sheet, header=2)
        df = df.dropna(axis=1, how='all')
        df = df.loc[:, ~df.iloc[1:].isna().all()]
        df = df.dropna(how='all')

        # Convert datetime objects to strings
        df = df.map(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, datetime) else x)

        st.write(f"Displaying content of: {selected_title}")
        st.dataframe(df)

        # Add download button for CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name=f"{selected_title}.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Error reading Excel file: {str(e)}")

def display_pdf_content(file_key):
    try:
        file_content = read_file_from_s3(file_key)
        base64_pdf = base64.b64encode(file_content.getvalue()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying PDF file: {str(e)}")

def display_file_content(file_key):
    if file_key.endswith(('.xlsx', '.xls')):
        display_excel_content(file_key)
    elif file_key.endswith('.pdf'):
        display_pdf_content(file_key)
    else:
        st.error("Unsupported file type")

def display_chat_interface(file_key):
    st.write("### Document Assistant")
    st.write("Ask questions about the document and I'll help you find the answers.")

    chat_container = st.container()

    if st.session_state.current_file != file_key:
        st.session_state.chat_messages = deque(maxlen=10)
        st.session_state.current_file = file_key

    with chat_container:
        for msg in st.session_state.chat_messages:
            message_container(
                msg.content,
                is_user=(msg.role == "user"),
                key=f"{msg.role}_{msg.timestamp}"
            )

    if prompt := st.chat_input("Ask your question about the document..."):
        user_msg = ChatMessage(role="user", content=prompt)
        st.session_state.chat_messages.append(user_msg)
        message_container(prompt, is_user=True)

        # Get context from the current file's cached content
        doc_content = get_cached_document_content(file_key)
        response = get_claude_response_up(prompt, doc_content)
        #response = ""

        if response:
            assistant_msg = ChatMessage(role="assistant", content=response)
            st.session_state.chat_messages.append(assistant_msg)
            message_container(response, is_user=False)

        st.rerun()

def display_file_selection(bank_name):
    """Display file selection for a specific bank"""
    bank_info = BANK_PATHS[bank_name]

    # Initialize current_period for this bank if not exists
    if bank_name not in st.session_state.current_period:
        st.session_state.current_period[bank_name] = None

    selected_period = st.selectbox(
        f"Select period for {bank_name}:",
        bank_info['folders'],
        key=f"{bank_name}_period"
    )

    # Handle period change
    if st.session_state.current_period[bank_name] != selected_period:
        st.session_state.current_period[bank_name] = selected_period
        if st.session_state.current_bank == bank_name:
            st.session_state.selected_file_key = None

    prefix = f"{bank_info['base']}/{selected_period}/"
    files = [''] + get_s3_files(prefix)

    if not files:
        st.warning(f"No files found for {bank_name} in {selected_period}")
        return None

    # Get current file for this bank
    current_file_options = [os.path.basename(f) for f in files]
    file_index = 0
    if (st.session_state.selected_file_key and
        os.path.basename(st.session_state.selected_file_key) in current_file_options):
        file_index = current_file_options.index(
            os.path.basename(st.session_state.selected_file_key)
        )

    selected_file = st.selectbox(
        "Select a file to analyze:",
        current_file_options,
        index=file_index,
        key=f"{bank_name}_file"
    )

    if selected_file:
        new_file_key = f"{prefix}{selected_file}"

        # Only update if actually changing files
        if st.session_state.selected_file_key != new_file_key:
            st.session_state.current_bank = bank_name
            st.session_state.selected_file_key = new_file_key
            # Clear existing cache for the old file
            if st.session_state.current_file:
                clear_file_cache(st.session_state.current_file)
            st.session_state.current_file = new_file_key
            # Load new content
            get_cached_document_content(new_file_key)

        return new_file_key

def display_file_content_tab(file_key):
    """Display file content tab"""
    display_file_content(file_key)

def display_short_summary_tab(file_key):
    """Display short summary tab"""
    doc_content = get_cached_document_content(file_key)
    short_summary, _ = get_cached_summaries(file_key, doc_content)
    st.write("### Quick Summary")
    st.write(short_summary)

def display_detailed_summary_tab(file_key):
    """Display detailed summary tab"""
    doc_content = get_cached_document_content(file_key)
    _, detailed_summary = get_cached_summaries(file_key, doc_content)
    st.write("### Detailed Summary")
    st.write(detailed_summary)

def display_bank_content(bank_name):
    """Display content for a specific bank"""
    # Create sub-tabs dynamically
    tabs = st.tabs(SUB_TABS)

    # Handle file selection
    with tabs[0]:  # "Select a file" tab
        file_key = display_file_selection(bank_name)

    # Only display content if this is the active bank and we have a file selected
    if st.session_state.current_bank == bank_name and file_key:
        with tabs[1]:  # "Display File Content" tab
            display_file_content_tab(file_key)

        with tabs[2]:  # "Short Summary" tab
            display_short_summary_tab(file_key)

        with tabs[3]:  # "Detailed Summary" tab
            display_detailed_summary_tab(file_key)

# Main UI setup
def main():
    st.set_page_config(page_title="GenAI Document Processing Solution", page_icon=":chart_with_upwards_trend:", layout="wide")
    st.header("Finance Benchmarking")

    # Initialize session state
    initialize_chat_state()

    # Main content container
    main_content = st.container()

    with main_content:
        # Create tabs dynamically for each bank
        bank_tabs = st.tabs(BANKS)

        # Display content for each bank
        for i, bank_name in enumerate(BANKS):
            with bank_tabs[i]:
                display_bank_content(bank_name)

    # Add a separator
    st.markdown("---")

    # Fixed chat interface at the bottom
    st.markdown("""
        <style>
        .chat-container {
            margin-top: 2rem;
            padding: 1rem;
            border-radius: 0.5rem;
            background: #f8f9fa;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    if "selected_file_key" in st.session_state and st.session_state.selected_file_key:
        display_chat_interface(st.session_state.selected_file_key)
    else:
        st.info("Please select a file to start chatting")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

==============================================
