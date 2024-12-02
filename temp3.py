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
BUCKET_NAME = 'myawstests3buckets1'
BASE_PATH = 'financebenchmarking1'

SUPPORTED_FILE_TYPES = {
    'document': ['.pdf', '.xlsx', '.xls', '.csv'],
    'audio': ['.mp3', '.wav', '.m4a'],
    'video': ['.mp4', '.avi', '.mov']
}

BANKS = ["Lloyds", "Santander"]
SUB_TABS = [
    "Select a file",
    "Display File Content",
    "Short Summary",
    "Detailed Summary"
]

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

def get_years_for_bank(bank_name):
    """Get available years for a bank"""
    prefix = f"{BASE_PATH}/{bank_name.lower()}/"
    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix, Delimiter='/')
        years = []
        for prefix in response.get('CommonPrefixes', []):
            year = prefix.get('Prefix').split('/')[-2]
            if year.isdigit():  # Ensure it's a valid year
                years.append(year)
        return sorted(years, reverse=True)  # Most recent years first
    except Exception as e:
        st.error(f"Error accessing years: {str(e)}")
        return []

def get_periods_for_year(bank_name, year):
    """Get available periods for a bank and year"""
    prefix = f"{BASE_PATH}/{bank_name.lower()}/{year}/"
    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix, Delimiter='/')
        periods = []
        for prefix in response.get('CommonPrefixes', []):
            period = prefix.get('Prefix').split('/')[-2]
            periods.append(period)
        return sorted(periods)
    except Exception as e:
        st.error(f"Error accessing periods: {str(e)}")
        return []

def get_files_in_period(bank_name, year, period):
    """Get all files in a specific period"""
    prefix = f"{BASE_PATH}/{bank_name.lower()}/{year}/{period}/"
    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
        files = []
        for obj in response.get('Contents', []):
            file_path = obj['Key']
            if any(file_path.lower().endswith(ext) for ext_list in SUPPORTED_FILE_TYPES.values() for ext in ext_list):
                files.append(file_path)
        return sorted(files)
    except Exception as e:
        st.error(f"Error accessing files: {str(e)}")
        return []
    
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
    if "file_cache" not in st.session_state:
        st.session_state.file_cache = {}

def cache_file_content(file_key):
    """Cache file content when first loaded"""
    if file_key not in st.session_state.file_cache:
        try:
            file_content = read_file_from_s3(file_key)
            if file_content:
                st.session_state.file_cache[file_key] = file_content
                return file_content
        except Exception as e:
            st.error(f"Error caching file: {str(e)}")
            return None
    return st.session_state.file_cache.get(file_key)

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
    """Clear all caches related to a specific file"""
    if file_key in st.session_state.document_content:
        del st.session_state.document_content[file_key]
    if file_key in st.session_state.summaries:
        del st.session_state.summaries[file_key]
    if file_key in st.session_state.file_cache:
        del st.session_state.file_cache[file_key]
    # Clear sheet titles and dataframe caches
    cache_keys = [k for k in st.session_state.keys() if k.startswith(f"{file_key}_")]
    for key in cache_keys:
        del st.session_state[key]
    if st.session_state.current_file == file_key:
        st.session_state.current_file = None

def get_cached_document_content(file_key):
    """Get document content from cache or generate if not available"""
    if file_key not in st.session_state.document_content:
        print(f"Cache miss - Loading content for {file_key}")
        # Use cached file content if available
        file_content = st.session_state.file_cache.get(file_key)
        if not file_content:
            file_content = cache_file_content(file_key)
            
        if file_content:
            if file_key.endswith('.pdf'):
                content = extract_text_from_pdf(file_content)
            elif file_key.endswith(('.xlsx', '.xls')):
                content = extract_text_from_excel(file_content)
            else:
                content = ""
            st.session_state.document_content[file_key] = content
    return st.session_state.document_content.get(file_key, "")

def display_excel_content(file_key):
    try:
        # Use cached content instead of reading from S3 again
        file_content = st.session_state.file_cache.get(file_key)
        if not file_content:
            file_content = cache_file_content(file_key)
            
        if not file_content:
            st.error("Failed to load file content")
            return

        xl = pd.ExcelFile(file_content)
        sheet_names = [sheet for sheet in xl.sheet_names if sheet not in ["Cover", "Contents"]]

        st.write(f"Excel file: {os.path.basename(file_key)}")

        # Cache sheet titles
        cache_key = f"{file_key}_sheet_titles"
        if cache_key not in st.session_state:
            sheet_titles = {}
            for sheet in sheet_names:
                df = pd.read_excel(file_content, sheet_name=sheet, header=None)
                if not df.empty:
                    sheet_titles[sheet] = df.iloc[0, 0]
                else:
                    sheet_titles[sheet] = sheet
            st.session_state[cache_key] = sheet_titles
        else:
            sheet_titles = st.session_state[cache_key]

        selected_title = st.selectbox("Select data to view:", list(sheet_titles.values()))
        selected_sheet = [sheet for sheet, title in sheet_titles.items() if title == selected_title][0]

        # Cache dataframe for each sheet
        df_cache_key = f"{file_key}_{selected_sheet}_df"
        if df_cache_key not in st.session_state:
            df = pd.read_excel(file_content, sheet_name=selected_sheet, header=2)
            df = df.dropna(axis=1, how='all')
            df = df.loc[:, ~df.iloc[1:].isna().all()]
            df = df.dropna(how='all')
            df = df.map(lambda x: x.strftime('%Y-%m-%d') if isinstance(x, datetime) else x)
            st.session_state[df_cache_key] = df
        else:
            df = st.session_state[df_cache_key]

        st.write(f"Displaying content of: {selected_title}")
        st.dataframe(df)

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
        # Use cached content instead of reading from S3 again
        file_content = st.session_state.file_cache.get(file_key)
        if not file_content:
            file_content = cache_file_content(file_key)
            
        if not file_content:
            st.error("Failed to load file content")
            return

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
    """Display file selection using Streamlit columns"""
    # Create three columns for the filters
    col1, col2, col3 = st.columns(3)
    
    # First column: Year selection
    with col1:
        years = get_years_for_bank(bank_name)
        if not years:
            st.warning(f"No data available for {bank_name}")
            return None

        selected_year = st.selectbox(
            "Year",
            years,
            key=f"{bank_name}_year"
        )

    # Second column: Period selection
    with col2:
        periods = get_periods_for_year(bank_name, selected_year)
        if not periods:
            st.warning(f"No periods available for {selected_year}")
            return None

        selected_period = st.selectbox(
            "Period",
            periods,
            key=f"{bank_name}_period"
        )

    # Third column: File selection
    with col3:
        files = get_files_in_period(bank_name, selected_year, selected_period)
        if not files:
            st.warning(f"No files available for {selected_period}")
            return None

        # Create a mapping of display names to full paths
        file_mapping = {os.path.basename(f): f for f in files}
        file_options = [''] + list(file_mapping.keys())
        
        # Get the currently selected file's basename if it exists
        current_file_basename = (os.path.basename(st.session_state.selected_file_key) 
                               if st.session_state.selected_file_key else '')
        
        # Set default index
        default_index = 0
        if current_file_basename in file_options:
            default_index = file_options.index(current_file_basename)
        
        selected_file = st.selectbox(
            "File",
            file_options,
            index=default_index,
            key=f"{bank_name}_file_{selected_year}_{selected_period}"  # Make key unique
        )

    if selected_file:
        # Get the full path from our mapping
        full_path = file_mapping[selected_file]
        
        # Only update if the file has actually changed
        if st.session_state.selected_file_key != full_path:
            st.session_state.current_bank = bank_name
            st.session_state.selected_file_key = full_path
            if st.session_state.current_file:
                clear_file_cache(st.session_state.current_file)
            st.session_state.current_file = full_path
            # Pre-cache the content
            cache_file_content(full_path)
            # Force a rerun to update the UI
            st.rerun()

        return full_path
    
    return None


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
        st.info("Please select a file..")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    #Initialize all session state variables
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
    main()
