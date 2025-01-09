import streamlit as st
import boto3
import pandas as pd
from typing import Dict, List, Optional, Any
import os
from datetime import datetime
import json
import io
from src.services.s3_service import S3Service
from src.services.knowledge_base_service import KnowledgeBaseService
from src.services.document_processor import DocumentProcessor
from src.services.bedrock_service import BedrockService
from src.services.guardrails_service import GuardrailsService
from src.utils.chat_history import ChatHistory
import time
import base64
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class DocumentAssistantUI:
    def __init__(self, common_config: Dict):
        """Initialize Document Assistant with configuration."""
        self.config = common_config
        self.s3_service = S3Service(common_config['s3_config'])
        self.kb_service = KnowledgeBaseService(common_config['knowledge_base_config'])
        self.doc_processor = DocumentProcessor(common_config['document_processing'])
        self.bedrock_service = BedrockService(common_config['model_config'])
        self.guardrails_service = GuardrailsService(common_config)
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize session state variables."""
        if 'doc_assistant_messages' not in st.session_state:
            st.session_state.doc_assistant_messages = []
        if 'current_document' not in st.session_state:
            st.session_state.current_document = None
        if 'selected_filters' not in st.session_state:
            st.session_state.selected_filters = {
                'bank': None,
                'year': None,
                'period': None
            }
        if 'folder_structure' not in st.session_state:
            st.session_state.folder_structure = self._get_folder_structure()
        if 'file_cache' not in st.session_state:
            st.session_state.file_cache = {}  # Initialize as an empty dictionary

    def render(self):
        """Main render method for Document Assistant."""
        self._apply_custom_css()
        
        if st.session_state.current_view == "documents":
            self._render_document_selection()
        elif st.session_state.current_view == "manage_documents":
            self._render_manage_documents()
        elif st.session_state.current_view == "explore_documents":
            self._render_explore_documents()

    def _apply_custom_css(self):
        """Apply custom CSS styling."""
        st.markdown("""
            <style>
            /* Chat Interface Styling */
            .chat-header {
                background-color: #0051A2;
                color: white;
                padding: 12px 15px;
                border-radius: 8px 8px 0 0;
                font-weight: bold;
                margin-bottom: 0;
            }
            
            .messages-area {
                flex-grow: 1;
                overflow-y: auto;
                padding: 1rem;
                scroll-behavior: smooth;
            }
            
            .chat-input-area {
                border-top: 1px solid #ddd;
                padding: 1rem;
                background: white;
            }
            
            /* Message Styling */
            .stChatMessage {
                max-width: 80%;
                margin: 0.5rem 0;
                padding: 0.5rem;
                border-radius: 8px;
            }
            
            .stChatMessage[data-testid="chat-message-user"] {
                margin-left: auto;
                background-color: #E3F2FD;
                text-align: right;
            }
            
            .stChatMessage[data-testid="chat-message-assistant"] {
                margin-right: auto;
                background-color: #F5F5F5;
                text-align: left;
            }
            
            .citation {
                font-size: 0.8em;
                color: #666;
                font-style: italic;
                margin-top: 0.5rem;
            }
            
            /* Hide Streamlit elements */
            .stDeployButton {
                display: none !important;
            }
            
            [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
                gap: 0rem !important;
            }
            </style>
        """, unsafe_allow_html=True)

    def _render_document_selection(self):
        """Render the document selection view with two main options."""
        col1, col2, col3 = st.columns([0.01, 10.5, 1])
        with col2:
            st.markdown("### Document Assistant")
        with col3:
            if st.button("← Back", key="document_back", use_container_width=True):
                st.session_state.current_view = 'main'
                st.rerun()

        # Two main options in separate columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class="item-card">
                    <h3>📁 Manage Documents</h3>
                    <p>Upload, organize, and manage your documents</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Open", key="manage_docs_btn", use_container_width=True):
                st.session_state.current_view = "manage_documents"
                st.rerun()

        with col2:
            st.markdown("""
                <div class="item-card">
                    <h3>🔍 Explore Documents</h3>
                    <p>Search and analyze document contents</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Open", key="explore_docs_btn", use_container_width=True):
                st.session_state.current_view = "explore_documents"
                st.rerun()

    def _render_manage_documents(self):
        """Render document management interface."""
        # Header with back button
        col1, col2, col3 = st.columns([0.01, 10.5, 1])
        with col2:
            st.markdown("### Manage Documents")
        with col3:
            if st.button("← Back", key="manage_back", use_container_width=True):
                st.session_state.current_view = "documents"
                st.rerun()

        # Upload Section
        #st.markdown("#### Upload New Document")
        with st.expander("Upload New Document"):
        
            # Initialize or get folder structure from session state
            if 'folder_structure' not in st.session_state:
                st.session_state.folder_structure = self._get_folder_structure()
                
            folder_structure = st.session_state.folder_structure

            # Initialize manage filters if not exists
            if 'manage_filters' not in st.session_state:
                st.session_state.manage_filters = {
                    'bank': None,
                    'year': None,
                    'period': None
                }

            # Initialize upload success flag in session state if not exists
            if 'upload_success' not in st.session_state:
                st.session_state.upload_success = False

            # Main container for folder selection and file upload
            with st.container():
                # Folder Selection
                st.markdown("#### Select folder option:")
                folder_choice = st.radio(
                    "",
                    ["Use Existing Folder", "Create New Folder"],
                    horizontal=True,
                    label_visibility="collapsed"
                )

                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if folder_choice == "Use Existing Folder":
                        banks = sorted(folder_structure.keys()) if folder_structure else []
                        current_bank_index = banks.index(st.session_state.manage_filters['bank']) + 1 if st.session_state.manage_filters['bank'] in banks else 0
                        selected_bank = st.selectbox(
                            "Bank",
                            options=[""] + banks,
                            help="Select the bank folder",
                            key="manage_bank",
                            index=current_bank_index
                        )
                        if selected_bank != st.session_state.manage_filters['bank']:
                            st.session_state.manage_filters['bank'] = selected_bank
                            st.session_state.manage_filters['year'] = None
                            st.session_state.manage_filters['period'] = None
                            st.rerun()
                    else:
                        selected_bank = st.text_input("Bank Name", help="Enter new bank name")
                        if selected_bank and not selected_bank.replace('_', '').isalnum():
                            st.error("Bank name should only contain letters, numbers, and underscores")
                            selected_bank = None

                with col2:
                    if folder_choice == "Use Existing Folder":
                        years = (sorted(folder_structure[selected_bank].keys()) 
                                if selected_bank and selected_bank in folder_structure else [])
                        current_year_index = years.index(st.session_state.manage_filters['year']) + 1 if st.session_state.manage_filters['year'] in years else 0
                        selected_year = st.selectbox(
                            "Year",
                            options=[""] + years,
                            key="manage_year",
                            disabled=not selected_bank,
                            index=current_year_index
                        )
                        if selected_year != st.session_state.manage_filters['year']:
                            st.session_state.manage_filters['year'] = selected_year
                            st.session_state.manage_filters['period'] = None
                            st.rerun()
                    else:
                        selected_year = st.text_input("Year", help="Enter year (YYYY)")
                        if selected_year and (not selected_year.isdigit() or len(selected_year) != 4):
                            st.error("Year should be a 4-digit number")
                            selected_year = None

                with col3:
                    if folder_choice == "Use Existing Folder":
                        periods = (sorted(folder_structure[selected_bank][selected_year].keys())
                                if selected_bank and selected_year and 
                                selected_bank in folder_structure and 
                                selected_year in folder_structure[selected_bank] else [])
                        current_period_index = periods.index(st.session_state.manage_filters['period']) + 1 if st.session_state.manage_filters['period'] in periods else 0
                        selected_period = st.selectbox(
                            "Period",
                            options=[""] + periods,
                            key="manage_period",
                            disabled=not (selected_bank and selected_year),
                            index=current_period_index
                        )
                        if selected_period != st.session_state.manage_filters['period']:
                            st.session_state.manage_filters['period'] = selected_period
                    else:
                        selected_period = st.text_input("Period", help="Enter period name")
                        if selected_period and not selected_period.replace('_', '').isalnum():
                            st.error("Period should only contain letters, numbers, and underscores")
                            selected_period = None

            # Show folder preview and file upload only if folder is selected
            if all([selected_bank, selected_year, selected_period]):
                main_folder = self.config['s3_config']['main_folder']
                st.markdown(
                    f"""
                    <div class="folder-preview">
                        <p>Selected folder path:</p>
                        <code>/{main_folder}/{selected_bank}/{selected_year}/{selected_period}/</code>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Reset file uploader if previous upload was successful
                if st.session_state.upload_success:
                    st.session_state.upload_success = False
                    st.rerun()

                 # File Upload Container
                with st.container():
                    uploaded_file = st.file_uploader(
                        "Drop your document here or click to upload",
                        type=self._get_supported_formats(),
                        key=f"file_uploader_{selected_bank}_{selected_year}_{selected_period}"
                    )

                    if uploaded_file:
                        st.markdown(
                            f"""
                            <div class="file-info">
                                <p>Selected file: <strong>{uploaded_file.name}</strong></p>
                                <p>Upload location: <code>/{main_folder}/{selected_bank}/{selected_year}/{selected_period}/{uploaded_file.name}</code></p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        if st.button("Upload and Process Document", type="primary", use_container_width=True):
                            try:
                                # Create a placeholder for progress messages
                                progress_placeholder = st.empty()

                                # Step 1: Upload to S3
                                progress_placeholder.info("Uploading to S3...")
                                s3_file = io.BytesIO(uploaded_file.getvalue())
                                s3_file.name = uploaded_file.name
                                s3_path = self.s3_service.upload_file(s3_file, selected_bank, selected_year, selected_period)

                                # Step 2: Process document
                                progress_placeholder.info("Processing document...")
                                process_file = io.BytesIO(uploaded_file.getvalue())
                                process_file.name = uploaded_file.name
                                chunks = self.doc_processor.process_document(
                                    file=process_file,
                                    filename=uploaded_file.name
                                )

                                # Step 3: Index in OpenSearch
                                progress_placeholder.info("Indexing document...")
                                self.opensearch_service.index_document(
                                    chunks,
                                    s3_path,
                                    metadata={
                                        "bank": selected_bank,
                                        "year": selected_year,
                                        "period": selected_period,
                                        "filename": uploaded_file.name,
                                        "file_type": os.path.splitext(uploaded_file.name)[1].lower(),
                                        "upload_time": datetime.utcnow().isoformat()
                                    }
                                )

                                # Show success message and clear the progress placeholder
                                progress_placeholder.empty()
                                st.success("Document processed successfully!")

                                # Set upload success flag
                                if 'upload_success' not in st.session_state:
                                    st.session_state.upload_success = True

                                # Add small delay before rerun to ensure message is shown
                                time.sleep(2)
                                st.rerun()

                            except Exception as e:
                                st.error(f"Error processing document: {str(e)}")
                                st.error("Please try again or contact support if the issue persists.")

    def _display_file_content(self, file_key: str):
        """Display file content based on type."""
        try:
            if file_key.endswith(('.xlsx', '.xls')):
                self._display_excel_content(file_key)
            elif file_key.endswith('.pdf'):
                self._display_pdf_content(file_key)
            else:
                st.error("Unsupported file type")
        except Exception as e:
            st.error(f"Error displaying file: {str(e)}")   

    def _display_excel_content(self, file_key: str):
        """Display Excel file content."""
        try:
            # Cache or fetch content from S3
            file_content = st.session_state.file_cache.get(file_key)
            if not file_content:
                file_content = self.s3_service.get_document_content_explore(file_key)
                st.session_state.file_cache[file_key] = file_content

            xl = pd.ExcelFile(io.BytesIO(file_content))
            sheet_name = st.selectbox("Select Sheet", options=xl.sheet_names)
            if sheet_name:
                df = xl.parse(sheet_name=sheet_name)
                st.dataframe(df, use_container_width=True)

                # Download option
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=csv_data,
                    file_name=f"{sheet_name}.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")

    def _display_pdf_content(self, file_key: str):
        """Display PDF file content."""
        try:
            # Fetch or retrieve from cache
            file_content = st.session_state.file_cache.get(file_key)
            #print("file_content1")
            
            if not file_content:
                file_content = self.s3_service.get_document_content_explore(file_key)
                #print("file_content2")
                st.session_state.file_cache[file_key] = file_content

            # Convert PDF content to Base64 and display in iframe
            base64_pdf = base64.b64encode(file_content).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error displaying PDF file: {str(e)}")
            
    def _render_explore_documents(self):
        """Render document exploration interface with cascading filters."""
        self._apply_custom_css()

        # Header with back button
        col1, col2, col3 = st.columns([0.01, 10.5, 1])
        with col2:
            st.markdown("### Explore Documents")
        with col3:
            if st.button("← Back", key="explore_back", use_container_width=True):
                st.session_state.current_view = "documents"
                st.rerun()

        # Initialize or get folder structure from session state
        if 'folder_structure' not in st.session_state:
            st.session_state.folder_structure = self._get_folder_structure()

        folder_structure = st.session_state.folder_structure        

        # Initialize filters in session state
        if 'explore_filters' not in st.session_state:
            st.session_state.explore_filters = {
                'bank': None,
                'year': None,
                'period': None,
                'document': None
            }

        # Function to reset chat messages
        def reset_chat_messages():
            if "doc_assistant_messages" in st.session_state:
                st.session_state.doc_assistant_messages = []

        # Document Selection Filters in Sidebar
        with st.sidebar:
            st.markdown("### Document Filters")

            # Bank Selection
            banks = sorted(folder_structure.keys()) if folder_structure else []
            banks_capitalized = [bank.capitalize() for bank in banks]
            current_bank_index = banks_capitalized.index(st.session_state.explore_filters['bank']) + 1 if st.session_state.explore_filters['bank'] in banks_capitalized else 0
            selected_bank = st.selectbox(
                "Select Bank",
                options=[""] + banks_capitalized,
                key="explore_bank",
                index=current_bank_index
            )
            selected_bank_actual = banks[banks_capitalized.index(selected_bank)] if selected_bank else None
            if selected_bank_actual != st.session_state.explore_filters['bank']:
                st.session_state.explore_filters['bank'] = selected_bank_actual
                st.session_state.explore_filters['year'] = None
                st.session_state.explore_filters['period'] = None
                st.session_state.explore_filters['document'] = None
                reset_chat_messages()
                st.rerun()

            # Year Selection
            years = (sorted(folder_structure[selected_bank_actual].keys()) 
                    if selected_bank_actual and selected_bank_actual in folder_structure else [])
            current_year_index = years.index(st.session_state.explore_filters['year']) + 1 if st.session_state.explore_filters['year'] in years else 0
            selected_year = st.selectbox(
                "Select Year",
                options=[""] + years,
                key="explore_year",
                disabled=not selected_bank_actual,
                index=current_year_index
            )
            if selected_year != st.session_state.explore_filters['year']:
                st.session_state.explore_filters['year'] = selected_year
                st.session_state.explore_filters['period'] = None
                st.session_state.explore_filters['document'] = None
                reset_chat_messages()
                st.rerun()

            # Period Selection
            periods = (sorted(folder_structure[selected_bank_actual][selected_year].keys())
                    if selected_bank_actual and selected_year and 
                    selected_bank_actual in folder_structure and 
                    selected_year in folder_structure[selected_bank_actual] else [])
            periods_capitalized = [period.capitalize() for period in periods]
            current_period_index = periods_capitalized.index(st.session_state.explore_filters['period']) + 1 if st.session_state.explore_filters['period'] in periods_capitalized else 0
            selected_period = st.selectbox(
                "Select Period",
                options=[""] + periods_capitalized,
                key="explore_period",
                disabled=not (selected_bank_actual and selected_year),
                index=current_period_index
            )
            selected_period_actual = periods[periods_capitalized.index(selected_period)] if selected_period else None
            if selected_period_actual != st.session_state.explore_filters['period']:
                st.session_state.explore_filters['period'] = selected_period_actual
                st.session_state.explore_filters['document'] = None
                reset_chat_messages()
                st.rerun()

            # Document Selection (also in sidebar)
            if all([selected_bank_actual, selected_year, selected_period_actual]):
                documents = folder_structure[selected_bank_actual][selected_year][selected_period_actual]
                if documents:
                    selected_doc = st.selectbox(
                        "Select Document",
                        options=[""] + documents,
                        format_func=lambda x: os.path.basename(x) if x else "Select a document",
                        key="explore_document"
                    )
                    if selected_doc != st.session_state.explore_filters['document']:
                        st.session_state.explore_filters['document'] = selected_doc
                        reset_chat_messages()
                        st.rerun()
                else:
                    st.info("No documents found in the selected folder")

        # Main content area
        selected_doc = st.session_state.explore_filters.get('document')
        if selected_doc:
            st.session_state.current_document = selected_doc

            # Document viewer in expandable section
            with st.expander("View Document Content", expanded=False):
                st.write(f"Selected Document: {selected_doc}")
                self._display_file_content(selected_doc)

            # Chat interface
            st.markdown('<div class="chat-header">Financial Document Assistant</div>', unsafe_allow_html=True)
            self._render_chat_interface(selected_doc)

    def _get_folder_structure(self) -> Dict:
        """Get complete folder structure from S3."""
        try:
            folder_structure = {}
            response = self.s3_service.s3_client.list_objects_v2(
                Bucket=self.s3_service.bucket,
                Prefix=f"{self.s3_service.main_folder}/"
            )
            
            for obj in response.get('Contents', []):
                path_parts = obj['Key'].split('/')
                if len(path_parts) >= 5 and path_parts[-1]:
                    bank = path_parts[1]
                    year = path_parts[2]
                    period = path_parts[3]
                    file_path = obj['Key']
                    
                    if bank not in folder_structure:
                        folder_structure[bank] = {}
                    if year not in folder_structure[bank]:
                        folder_structure[bank][year] = {}
                    if period not in folder_structure[bank][year]:
                        folder_structure[bank][year][period] = []
                    
                    folder_structure[bank][year][period].append(file_path)
            
            return folder_structure
        except Exception as e:
            st.error(f"Error getting folder structure: {str(e)}")
            return {}

    def create_dynamodb_table():
        try:
            dynamodb = boto3.client('dynamodb')
            
            table = dynamodb.create_table(
                TableName='document_chat_history',
                KeySchema=[
                    {
                        'AttributeName': 'chat_id',
                        'KeyType': 'HASH'  # Partition key
                    }
                ],
                AttributeDefinitions=[
                    {
                        'AttributeName': 'chat_id',
                        'AttributeType': 'S'
                    }
                ],
                BillingMode='PAY_PER_REQUEST'
            )
            
            print("Table created successfully!")
            
        except dynamodb.exceptions.ResourceInUseException:
            print("Table already exists")
            
    def _render_chart_with_values(self, df: pd.DataFrame, chart_type: str, chart_attrs: dict):
        """Render chart with optional color using plotly."""
        try:
            # Handle both 2-column and 3-column data
            if len(df.columns) < 2:
                st.error(f"Insufficient columns in data. Found columns: {df.columns.tolist()}")
                return

            # Get column names
            category_col = df.columns[0]
            
            # Common label configuration
            labels = {
                category_col: chart_attrs.get('xAxis', category_col)
            }
            
            if len(df.columns) == 2:
                # 2-column format
                value_col = df.columns[1]
                df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
                labels[value_col] = chart_attrs.get('yAxis', value_col)
                
                # Create visualization based on chart type
                if chart_type == "pie":
                    fig = px.pie(
                        df,
                        values=value_col,
                        names=category_col,
                        title=chart_attrs.get('title', ''),
                        labels=labels,
                        hole=0  # No hole for regular pie
                    )
                    fig.update_traces(
                        textposition='inside',
                        textinfo='percent+label+value',
                        texttemplate='%{label}<br>%{value:.2f}<br>(%{percent:.1%})'
                    )
                
                elif chart_type == "donut":
                    fig = px.pie(
                        df,
                        values=value_col,
                        names=category_col,
                        title=chart_attrs.get('title', ''),
                        labels=labels,
                        hole=0.4  # Create donut hole
                    )
                    fig.update_traces(
                        textposition='inside',
                        textinfo='percent+label+value',
                        texttemplate='%{label}<br>%{value:.2f}<br>(%{percent:.1%})'
                    )
                
                elif chart_type == "funnel":
                    fig = go.Figure(go.Funnel(
                        y=df[category_col],
                        x=df[value_col],
                        textposition="inside",
                        textinfo="value+percent initial",
                        texttemplate='%{value:.2f}<br>(%{percentInitial:.1%})'
                    ))
                    fig.update_layout(title=chart_attrs.get('title', ''))
                
                elif chart_type == "treemap":
                    fig = px.treemap(
                        df,
                        path=[category_col],
                        values=value_col,
                        title=chart_attrs.get('title', '')
                    )
                    fig.update_traces(
                        textinfo="label+value+percent parent",
                        texttemplate='%{label}<br>%{value:.2f}<br>(%{percentParent:.1%})'
                    )
                
                elif chart_type == "bubble":
                    fig = px.scatter(
                        df,
                        x=category_col,
                        y=value_col,
                        size=value_col,
                        title=chart_attrs.get('title', ''),
                        labels=labels
                    )
                    fig.update_traces(
                        texttemplate='%{y:.2f}',
                        textposition='top center'
                    )
                
                elif chart_type == "bar":
                    fig = px.bar(
                        df,
                        x=category_col,
                        y=value_col,
                        title=chart_attrs.get('title', ''),
                        labels=labels
                    )
                    fig.update_traces(
                        texttemplate='%{y:.2f}',
                        textposition='outside'
                    )
                
                elif chart_type == "line":
                    fig = px.line(
                        df,
                        x=category_col,
                        y=value_col,
                        title=chart_attrs.get('title', ''),
                        markers=True,
                        labels=labels
                    )
                    fig.update_traces(
                        texttemplate='%{y:.2f}',
                        textposition='top center'
                    )
                
                elif chart_type == "area":
                    fig = px.area(
                        df,
                        x=category_col,
                        y=value_col,
                        title=chart_attrs.get('title', ''),
                        labels=labels
                    )
                
                elif chart_type == "scatter":
                    fig = px.scatter(
                        df,
                        x=category_col,
                        y=value_col,
                        title=chart_attrs.get('title', ''),
                        labels=labels
                    )
                    fig.update_traces(
                        texttemplate='%{y:.2f}',
                        textposition='top center'
                    )
                
                elif chart_type == "horizontal_bar":
                    fig = px.bar(
                        df,
                        y=category_col,
                        x=value_col,
                        title=chart_attrs.get('title', ''),
                        labels=labels,
                        orientation='h'
                    )
                    fig.update_traces(
                        texttemplate='%{x:.2f}',
                        textposition='outside'
                    )
            
            else:
                # 3-column format
                group_col = df.columns[1]
                value_col = df.columns[2]
                
                # Convert data types
                df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
                df[group_col] = df[group_col].astype(str)
                
                # Update labels for 3-column format
                labels.update({
                    value_col: chart_attrs.get('yAxis', value_col),
                    group_col: chart_attrs.get('color', group_col)
                })
                
                if chart_type == "pie":
                    # For 3-column data, create a sunburst chart instead of pie
                    fig = px.sunburst(
                        df,
                        path=[category_col, group_col],
                        values=value_col,
                        title=chart_attrs.get('title', '')
                    )
                    fig.update_traces(
                        textinfo="label+value+percent parent",
                        texttemplate='%{label}<br>%{value:.2f}<br>(%{percentParent:.1%})'
                    )
                
                elif chart_type == "donut":
                    # For 3-column data, create a multi-level donut chart
                    fig = px.sunburst(
                        df,
                        path=[category_col, group_col],
                        values=value_col,
                        title=chart_attrs.get('title', ''),
                    )
                    fig.update_traces(
                        textinfo="label+value+percent parent",
                        texttemplate='%{label}<br>%{value:.2f}<br>(%{percentParent:.1%})'
                    )
                
                elif chart_type == "treemap":
                    fig = px.treemap(
                        df,
                        path=[category_col, group_col],
                        values=value_col,
                        title=chart_attrs.get('title', '')
                    )
                    fig.update_traces(
                        textinfo="label+value+percent parent",
                        texttemplate='%{label}<br>%{value:.2f}<br>(%{percentParent:.1%})'
                    )
                
                elif chart_type == "bubble":
                    fig = px.scatter(
                        df,
                        x=category_col,
                        y=value_col,
                        size=value_col,
                        color=group_col,
                        title=chart_attrs.get('title', ''),
                        labels=labels
                    )
                    fig.update_traces(
                        texttemplate='%{y:.2f}',
                        textposition='top center'
                    )
                
                elif chart_type == "bar":
                    fig = px.bar(
                        df,
                        x=category_col,
                        y=value_col,
                        color=group_col,
                        title=chart_attrs.get('title', ''),
                        barmode='group',
                        labels=labels
                    )
                    fig.update_traces(
                        texttemplate='%{y:.2f}',
                        textposition='outside'
                    )
                
                elif chart_type == "line":
                    fig = px.line(
                        df,
                        x=category_col,
                        y=value_col,
                        color=group_col,
                        title=chart_attrs.get('title', ''),
                        markers=True,
                        labels=labels
                    )
                    fig.update_traces(
                        texttemplate='%{y:.2f}',
                        textposition='top center'
                    )
                
                elif chart_type == "area":
                    fig = px.area(
                        df,
                        x=category_col,
                        y=value_col,
                        color=group_col,
                        title=chart_attrs.get('title', ''),
                        labels=labels
                    )
                
                elif chart_type == "scatter":
                    fig = px.scatter(
                        df,
                        x=category_col,
                        y=value_col,
                        color=group_col,
                        title=chart_attrs.get('title', ''),
                        labels=labels
                    )
                    fig.update_traces(
                        texttemplate='%{y:.2f}',
                        textposition='top center'
                    )
                
                elif chart_type == "horizontal_bar":
                    fig = px.bar(
                        df,
                        y=category_col,
                        x=value_col,
                        color=group_col,
                        title=chart_attrs.get('title', ''),
                        labels=labels,
                        orientation='h',
                        barmode='group'
                    )
                    fig.update_traces(
                        texttemplate='%{x:.2f}',
                        textposition='outside'
                    )

            # Calculate dynamic height based on data and chart type
            num_categories = len(df[category_col].unique())
            min_height = 400  # Minimum height
            height_per_category = 50  # Height per category
            
            # Calculate base height
            if chart_type in ['horizontal_bar', 'treemap', 'funnel']:
                # These charts need more height per category
                height = max(min_height, num_categories * height_per_category)
            elif chart_type in ['pie', 'donut']:
                # Pie and donut charts work well with square dimensions
                height = 800 if num_categories > 8 else 600
            else:
                # For other charts, use a more moderate scaling
                height = max(min_height, min(800, num_categories * 30))

            # Common layout updates
            fig.update_layout(
                width=1200,
                height=700,  # Dynamic height
                template='plotly_white',
                showlegend=True,
                legend=dict(
                    yanchor="top",    # Anchor to top
                    y=1.0,            # Position at top
                    xanchor="left",   # Anchor to left
                    x=1.02,           # Move slightly right of the chart
                    orientation="v",   # Vertical orientation
                    font=dict(size=12, family='Arial Black'),
                    bgcolor="rgba(255, 255, 255, 0.8)",  # Semi-transparent background
                    bordercolor="rgba(0, 0, 0, 0.2)",    # Light border
                    borderwidth=1
                ),
                margin=dict(
                    l=50, 
                    r=120,  # Increased right margin to accommodate legend
                    t=80,   # Top margin for title
                    b=100 if chart_type not in ['pie', 'donut', 'treemap'] else 50
                ),
                title=dict(
                    text=chart_attrs.get('title', ''),
                    font=dict(size=16, family='Arial Black')
                )
            )

            # Add grid only for charts that need it
            if chart_type not in ['pie', 'donut', 'treemap', 'funnel']:
                fig.update_layout(
                    xaxis=dict(
                        title=dict(
                            text=chart_attrs.get('xAxis', ''),
                            font=dict(size=14, family='Arial Black')
                        ),
                        tickangle=-45,
                        showgrid=True,
                        tickfont=dict(size=12, family='Arial Black')
                    ),
                    yaxis=dict(
                        title=dict(
                            text=chart_attrs.get('yAxis', ''),
                            font=dict(size=14, family='Arial Black')
                        ),
                        showgrid=True,
                        tickfont=dict(size=12, family='Arial Black')
                    )
                )

            # Render the chart
            st.plotly_chart(fig, use_container_width=False)

        except Exception as e:
            st.error(f"Error rendering chart: {str(e)}")
            print("Debug - Error details:", e)
            print("Debug - DataFrame info:", df.info())

    def _handle_chart_selection(self, chart_type: str, dataset_key: str):
        """Handle chart selection and update state."""
        # Initialize selection if not exists
        if 'chart_selections' not in st.session_state:
            st.session_state.chart_selections = {}
        
        # Update selection
        st.session_state.chart_selections[dataset_key] = chart_type

    def _render_chat_interface(self, document_path: str):
        """Render chat interface for document interaction."""
        # Initialize session states
        if "doc_assistant_messages" not in st.session_state:
            st.session_state.doc_assistant_messages = []
        if "selected_document" not in st.session_state:
            st.session_state.selected_document = document_path
        if "chart_selections" not in st.session_state:
            st.session_state.chart_selections = {}
        if 'checkbox_states' not in st.session_state:
            st.session_state.checkbox_states = {}

        # Check if document changed
        if st.session_state.selected_document != document_path:
            st.session_state.selected_document = document_path
            st.session_state.doc_assistant_messages = []
            st.session_state.chart_selections = {}
                
        # Custom CSS for message colors
        st.markdown(
            """
            <style>
            .user-message {
                background-color: #d1e7dd;
                border-radius: 8px;
                padding: 8px;
            }
            .assistant-message {
                background-color: #e7e7ff;
                border-radius: 8px;
                padding: 8px;
            }
            .stButton > button {
                width: 100px !important;
                min-width: 100px;
                white-space: nowrap;
                padding: 0.5rem 1rem;
                text-align: center;
                background-color: #E7E7E7 !important;
                color: black !important;
                transition: all 0.3s ease;
                border: none !important;
            }
            .stButton > button:hover {
                background-color: #CCCCCC !important;
                color: black !important;
                border: none !important;
            }
            /* Primary button styling */
            .stButton > button[kind="primary"] {
                background-color: #0051A2 !important;
                color: white !important;
                border: none !important;
            }
            .stButton > button[kind="primary"]:hover {
                background-color: #003D82 !important;
                color: white !important;
            }
            .chart-type-label {
                margin-bottom: 0.0rem !important;
                padding-bottom: 0 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )        

        # Display existing messages
        for msg_idx, msg in enumerate(st.session_state.doc_assistant_messages):
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    try:
                        # Parse response data
                        response_data = json.loads(msg["content"])

                        # Display answer text
                        if "answer" in response_data:
                            user_answer = response_data["answer"]
                            color_class = "user-message" if msg["role"] == "user" else "assistant-message"
                            st.markdown(f"<div class='{color_class}'>{user_answer}</div>", unsafe_allow_html=True)

                        # Handle chart data display
                        if "chart_data" in response_data:
                            chart_data = response_data["chart_data"]
                            chart_attrs = response_data.get("chart_attributes", {})
                            
                            # For historical messages (within the message history loop):
                            for chart_idx, chart in enumerate(chart_data):
                                if "header" in chart and "rows" in chart:
                                    try:
                                        # Create DataFrame
                                        df = pd.DataFrame(chart["rows"], columns=chart["header"])
                                        
                                        # Show data table
                                        table_label = f"Data Table {chart_idx + 1}" if len(chart_data) > 1 else "Data Table"
                                        with st.expander(table_label):
                                            numeric_col = df.columns[-1]
                                            st.dataframe(df.style.format({numeric_col: "{:.2f}"}), use_container_width=True)
                                        
                                        st.write("")
                                        
                                        # Available chart types
                                        available_charts = [
                                            "bar", "horizontal_bar", "line", "area", "scatter",
                                            "pie", "donut", "treemap", "funnel", "bubble"
                                        ]
                                        
                                        # Chart selection header
                                        if len(chart_data) > 1:
                                            #st.write(f"Select chart type for dataset {chart_idx + 1}:")
                                            st.markdown(f'<p class="chart-type-label">Select chart type for dataset {chart_idx + 1}:</p>', unsafe_allow_html=True)
                                        else:
                                            st.markdown('<p class="chart-type-label">Select chart type:</p>', unsafe_allow_html=True)
                                            
                                        cols = st.columns(len(available_charts))
                                        
                                        # Generate unique key for this dataset
                                        dataset_key = f"history_{msg_idx}_{chart_idx}"

                                        # Handle chart selections
                                        dataset_key = f"history_{msg_idx}_{chart_idx}"
                                        current_selection = st.session_state.chart_selections.get(dataset_key)

                                        # Create uniform columns for the buttons
                                        num_buttons = len(available_charts)
                                        cols = st.columns(num_buttons)
                                        for idx, chart_type in enumerate(available_charts):
                                            with cols[idx]:
                                                button_label = chart_type.replace("_", " ").title()
                                                # Center-align the button in its column
                                                st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
                                                if st.button(
                                                    button_label,
                                                    type="secondary" if current_selection != chart_type else "primary",
                                                    key=f"history_chart_{msg_idx}_{chart_idx}_{chart_type}"
                                                ):
                                                    if current_selection == chart_type:
                                                        # Deselect if clicking the same type
                                                        st.session_state.chart_selections[dataset_key] = None
                                                    else:
                                                        # Select new chart type
                                                        self._handle_chart_selection(chart_type, dataset_key)
                                                    st.rerun()

                                        # Render chart if selected
                                        if current_selection:
                                            try:
                                                self._render_chart_with_values(df, current_selection, chart_attrs)
                                                st.write("")
                                            except Exception as e:
                                                st.error(f"Error rendering chart: {str(e)}")
                                                
                                    except Exception as e:
                                        st.error(f"Error displaying dataset {chart_idx + 1}: {str(e)}")
                                        st.write("Raw chart data:", chart)

                    except json.JSONDecodeError:
                        st.markdown(msg["content"])
                else:
                    st.markdown(msg["content"])

        # Chat input and message handling
        if prompt := st.chat_input("Ask about the document..."):
            # Add user message
            st.session_state.doc_assistant_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(f"<div class='user-message'>{prompt}</div>", unsafe_allow_html=True)

            # Generate and display assistant response
            with st.spinner("Analyzing document..."):
                response = self._get_document_response(prompt, document_path)

                st.session_state.doc_assistant_messages.append({
                    "role": "assistant",
                    "content": response
                })

                with st.chat_message("assistant"):
                    try:
                        response_data = json.loads(response)
                        
                        # Display answer text
                        if "answer" in response_data:
                            user_answer1 = response_data["answer"]
                            st.markdown(f"<div class='assistant-message'>{user_answer1}</div>", unsafe_allow_html=True)

                        # Handle chart data display
                        if "chart_data" in response_data:
                            chart_data = response_data["chart_data"]
                            chart_attrs = response_data.get("chart_attributes", {})
                            
                            # Handle multiple chart datasets
                            for chart_idx, chart in enumerate(chart_data):
                                if "header" in chart and "rows" in chart:
                                    try:
                                        # Create DataFrame
                                        df = pd.DataFrame(chart["rows"], columns=chart["header"])
                                        
                                        # Show data table with index for multiple charts
                                        if len(chart_data) > 1:
                                            with st.expander(f"Data Table {chart_idx + 1}"):
                                                numeric_col = df.columns[-1]
                                                st.dataframe(df.style.format({numeric_col: "{:.2f}"}), use_container_width=True)
                                        else:
                                            with st.expander(f"Data Table"):
                                                numeric_col = df.columns[-1]
                                                st.dataframe(df.style.format({numeric_col: "{:.2f}"}), use_container_width=True)
                                        
                                        # Add some spacing
                                        st.write("")
                                        
                                        # Create list of available chart types
                                        available_charts = [
                                            "bar", "horizontal_bar", "line", "area", "scatter",
                                            "pie", "donut", "treemap", "funnel", "bubble"
                                        ]
                                        
                                        # Create horizontal chart type selection
                                        if len(chart_data) > 1:
                                            #st.write(f"Select chart type for dataset {chart_idx + 1}:")
                                            st.markdown(f'<p class="chart-type-label">Select chart type for dataset {chart_idx + 1}:</p>', unsafe_allow_html=True)
                                        else:
                                            #st.write("Select chart type:")
                                            st.markdown('<p class="chart-type-label">Select chart type:</p>', unsafe_allow_html=True)
                                            
                                        cols = st.columns(len(available_charts))
                                        
                                        # For new messages (within the new message handling section):
                                        # Use the same logic but with different keys:

                                        dataset_key = f"new_{len(st.session_state.doc_assistant_messages)}_{chart_idx}"
                                        current_selection = st.session_state.chart_selections.get(dataset_key)

                                        # Create uniform columns for the buttons
                                        num_buttons = len(available_charts)
                                        cols = st.columns(num_buttons)
                                        for idx, chart_type in enumerate(available_charts):
                                            with cols[idx]:
                                                button_label = chart_type.replace("_", " ").title()
                                                # Center-align the button in its column
                                                st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
                                                if st.button(
                                                    button_label,
                                                    type="secondary" if current_selection != chart_type else "primary",
                                                    key=f"new_chart_{len(st.session_state.doc_assistant_messages)}_{chart_idx}_{chart_type}"
                                                ):
                                                    if current_selection == chart_type:
                                                        # Deselect if clicking the same type
                                                        st.session_state.chart_selections[dataset_key] = None
                                                    else:
                                                        # Select new chart type
                                                        self._handle_chart_selection(chart_type, dataset_key)
                                                    st.rerun()

                                        # Render chart if selected
                                        if current_selection:
                                            try:
                                                self._render_chart_with_values(df, current_selection, chart_attrs)
                                                st.write("")
                                            except Exception as e:
                                                st.error(f"Error rendering chart: {str(e)}")

                                    except Exception as e:
                                        st.error(f"Error displaying dataset {chart_idx + 1}: {str(e)}")
                                        st.write("Raw chart data:", chart)

                    except json.JSONDecodeError:
                        st.markdown(response)

    def _process_document(self, file, bank: str, year: str, period: str):
        """Process uploaded document with validation and error handling."""
        try:
            # Validate file
            validation_results = self._validate_file(file)
            if not validation_results['is_valid']:
                raise ValueError("\n".join(validation_results['errors']))

            # Read file content once
            file.seek(0)
            file_content = file.read()

            # Upload to S3
            s3_file = io.BytesIO(file_content)
            s3_file.name = file.name
            s3_path = self.s3_service.upload_file(s3_file, bank, year, period)

            # Process document
            process_file = io.BytesIO(file_content)
            process_file.name = file.name
            chunks = self.doc_processor.process_document(
                file=process_file,
                filename=file.name
            )

            # Index in Knowledge Base
            self.kb_service.index_document(
                chunks,
                s3_path,
                metadata={
                    "bank": bank,
                    "year": year,
                    "period": period,
                    "filename": file.name,
                    "file_type": os.path.splitext(file.name)[1].lower(),
                    "upload_time": datetime.utcnow().isoformat()
                }
            )

        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")
        finally:
            # Explicitly close file objects
            if 's3_file' in locals():
                s3_file.close()
            if 'process_file' in locals():
                process_file.close()

    def _validate_file(self, file) -> Dict[str, Any]:
        """Validate uploaded file."""
        try:
            validation_results = {
                'is_valid': True,
                'errors': [],
                'warnings': []
            }
            
            # Check file size
            file.seek(0, 2)  # Seek to end
            file_size = file.tell()
            file.seek(0)  # Reset position
            
            max_size = self.config['document_processing'].get('max_file_size_mb', 100) * 1024 * 1024
            if file_size > max_size:
                validation_results['is_valid'] = False
                validation_results['errors'].append(
                    f"File size ({file_size/1024/1024:.1f}MB) exceeds maximum allowed size ({max_size/1024/1024}MB)"
                )
            
            # Check file type
            file_extension = os.path.splitext(file.name)[1].lower()
            supported_formats = []
            for formats in self.config['document_processing']['supported_formats'].values():
                supported_formats.extend(formats)
                
            if file_extension not in supported_formats:
                validation_results['is_valid'] = False
                validation_results['errors'].append(
                    f"Unsupported file type: {file_extension}. Supported types: {', '.join(supported_formats)}"
                )
            
            return validation_results
            
        except Exception as e:
            return {
                'is_valid': False,
                'errors': [str(e)],
                'warnings': []
            }

    def _get_document_response(self, prompt: str, document_path: str) -> str:
        """Generate AI response based on document content with guardrails."""
        try:
            # Validate prompt using guardrails
            is_valid, error_message, redacted_prompt = self.guardrails_service.validate_prompt(prompt)
            
            if not is_valid:
                return json.dumps({
                    "answer": f"Security Alert: {error_message}. Please revise your question.",
                    "chart_data": []
                })

            # Use redacted prompt if PII was detected and redacted
            safe_prompt = redacted_prompt if redacted_prompt else prompt

            # Get response from Knowledge Base service
            response = self.kb_service.search_documents(
                query=safe_prompt,
                document_path=document_path,
                chat_history=st.session_state.doc_assistant_messages
            )

            if not response:
                return json.dumps({
                    "answer": "I couldn't find relevant information in the document to answer your question.",
                    "chart_data": []
                })

            # Validate response using guardrails
            is_valid, error_message, redacted_response = self.guardrails_service.validate_response(response)
            
            if not is_valid:
                return json.dumps({
                    "answer": f"Security Alert: {error_message}. Please try a different question.",
                    "chart_data": []
                })

            # Use redacted response if PII was detected and redacted
            safe_response = redacted_response if redacted_response else response

            # Log successful interaction
            self.guardrails_service.log_security_event(
                "successful_interaction",
                {
                    "document_path": document_path,
                    "prompt_length": len(prompt),
                    "response_length": len(response),
                    "pii_redacted": bool(redacted_prompt or redacted_response)
                }
            )

            return safe_response

        except Exception as e:
            error_msg = f"Error processing your question: {str(e)}"
            self.guardrails_service.log_security_event(
                "error",
                {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "document_path": document_path
                }
            )
            return json.dumps({
                "answer": error_msg,
                "chart_data": []
            })

    def _format_chunks_for_context(self, chunks: List[Dict]) -> str:
        """Format document chunks for context."""
        context_parts = []
        for chunk in chunks:
            source = os.path.basename(chunk['source'])
            page = chunk.get('page_number', 'N/A')
            context_parts.append(
                f"""Document: {source}
                Page: {page}
                Content: {chunk['content']}
                ---"""
            )
        return "\n\n".join(context_parts)

    def _format_response_with_citations(self, response: str, chunks: List[Dict]) -> str:
        """Format response with citations."""
        formatted_response = response
        
        # Add source summary
        sources = set()
        for chunk in chunks:
            source = os.path.basename(chunk['source'])
            page = chunk.get('page_number', 'N/A')
            sources.add(f"{source} (Page {page})")
        
        formatted_response += "\n\n**Sources:**\n"
        for source in sorted(sources):
            formatted_response += f"- {source}\n"
        
        return formatted_response

    def _get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        formats = []
        for format_list in self.config['document_processing']['supported_formats'].values():
            formats.extend(format_list)
        return formats