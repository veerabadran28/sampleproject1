import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
from datetime import datetime
import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import pdfkit
from typing import Dict, Tuple, Optional, List
from io import StringIO
import numpy as np
import math
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from io import BytesIO
import pandas as pd

class ReportLabDocumentGenerator:
    """Handles PDF generation using ReportLab."""
    
    def __init__(self):
        """Initialize the document generator with default styles."""
        self.styles = getSampleStyleSheet()
        self.custom_styles = self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom paragraph styles for the document."""
        custom_styles = {
            'Title': ParagraphStyle(
                'CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=TA_CENTER
            ),
            'Heading1': ParagraphStyle(
                'CustomHeading1',
                parent=self.styles['Heading1'],
                fontSize=18,
                spaceAfter=12,
                spaceBefore=12
            ),
            'Heading2': ParagraphStyle(
                'CustomHeading2',
                parent=self.styles['Heading2'],
                fontSize=16,
                spaceAfter=10,
                spaceBefore=10
            ),
            'Heading3': ParagraphStyle(
                'CustomHeading3',
                parent=self.styles['Heading3'],
                fontSize=14,
                spaceAfter=8,
                spaceBefore=8
            ),
            'Normal': ParagraphStyle(
                'CustomNormal',
                parent=self.styles['Normal'],
                fontSize=11,
                spaceAfter=8,
                leading=14
            ),
            'Table': ParagraphStyle(
                'CustomTable',
                parent=self.styles['Normal'],
                fontSize=10,
                leading=12
            )
        }
        return custom_styles
    
    def generate_line_chart(self, data: pd.DataFrame, title: str, y_label: str) -> Optional[go.Figure]:
        """Generate a simplified line chart."""
        try:
            print(f"Generating line chart: {title}")
            fig = go.Figure()
            
            # Define colors
            colors = {
                'BHC Base': '#000000',
                'BHC Stress': '#0000FF',
                'Sup Base': '#006400',
                'Sup Sev Adv': '#800080'
            }
            
            # Get time columns
            time_cols = [col for col in data.columns if col.startswith('20')]
            time_cols.sort()
            
            # Add traces
            for scenario in colors.keys():
                if scenario in data['Scenario ($bn)'].values:
                    scenario_data = data[data['Scenario ($bn)'] == scenario]
                    
                    y_values = [scenario_data[col].iloc[0] for col in time_cols]
                    
                    fig.add_trace(go.Scatter(
                        x=time_cols,
                        y=y_values,
                        name=scenario,
                        line=dict(color=colors[scenario])
                    ))
            
            # Simple layout
            fig.update_layout(
                title=title,
                showlegend=True,
                width=800,
                height=400,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            return fig
            
        except Exception as e:
            print(f"Error in chart generation: {str(e)}")
            return None

    def generate_pdf(self, cached_data: dict, config: dict) -> BytesIO:
        """Generate PDF document from cached data and configuration.
        
        Args:
            cached_data: Dictionary containing document content
            config: Document configuration dictionary
            
        Returns:
            BytesIO: PDF document as a bytes buffer
        """
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        elements = []
        
        # Add title page
        self._add_title_page(elements, config['document_info'])
        elements.append(PageBreak())
        
        # Add business sections
        if 'business_sections' in cached_data:
            for section_key, section_data in cached_data['business_sections'].items():
                self._add_section(elements, section_key, section_data)
                elements.append(PageBreak())
        
        # Build the PDF
        doc.build(elements)
        buffer.seek(0)
        return buffer

    def _add_title_page(self, elements: list, doc_info: dict):
        """Add title page elements."""
        # Add title
        elements.append(Paragraph(
            doc_info['title'],
            self.custom_styles['Title']
        ))
        elements.append(Spacer(1, 30))
        
        # Add metadata
        if 'metadata' in doc_info:
            for key, value in doc_info['metadata'].items():
                elements.append(Paragraph(
                    f"{key}: {value}",
                    self.custom_styles['Normal']
                ))
                elements.append(Spacer(1, 12))

    def _add_section(self, elements: list, section_key: str, section_data: dict):
        """Add a business section to the document."""
        # Add section title
        elements.append(Paragraph(
            section_data.get('title', 'Untitled Section'),
            self.custom_styles['Heading1']
        ))
        elements.append(Spacer(1, 12))
        
        # Add topics
        if 'topics' in section_data:
            for topic_key, topic_data in section_data['topics'].items():
                self._add_topic(elements, topic_data)

    def _add_topic(self, elements: list, topic_data: dict):
        """Add a topic to the document."""
        # Add topic title
        elements.append(Paragraph(
            topic_data.get('title', 'Untitled Topic'),
            self.custom_styles['Heading2']
        ))
        elements.append(Spacer(1, 10))
        
        # Add driver information
        if 'driver_info' in topic_data:
            elements.append(Paragraph(
                'Driver Information',
                self.custom_styles['Heading3']
            ))
            self._add_driver_table(elements, topic_data['driver_info'])
            elements.append(Spacer(1, 12))
        
        # Add projections
        if 'projections' in topic_data:
            self._add_projections(elements, topic_data['projections'])
        
        # Add narratives
        if 'narratives' in topic_data:
            self._add_narratives(elements, topic_data['narratives'])

    def _add_driver_table(self, elements: list, driver_info: dict):
        """Add driver information table."""
        # Create table data
        data = [
            ['Dependent Variable', 'Independent Variable', 'Lag', 'Direction'],
            [
                driver_info.get('dependent_var', ''),
                driver_info.get('independent_var', ''),
                str(driver_info.get('lag', '')),
                driver_info.get('direction', '')
            ]
        ]
        
        # Create table
        table = Table(data, colWidths=[2*inch, 2*inch, 1*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BOX', (0, 0), (-1, -1), 2, colors.black),
        ]))
        
        elements.append(table)

    def _add_projections(self, elements: list, projections: dict):
        """Add projection tables and charts."""
        if 'assets' in projections:
            elements.append(Paragraph(
                'Assets Projection',
                self.custom_styles['Heading3']
            ))
            self._add_projection_table(elements, projections['assets'])
            elements.append(Spacer(1, 12))
            
            # Add assets chart
            chart = self.generate_line_chart(
                pd.DataFrame(projections['assets']),
                'Assets Projection Over Time',
                'Assets ($bn)'
            )
            if chart:
                img_buffer = BytesIO()
                chart.write_image(img_buffer, format='png')
                img_buffer.seek(0)
                elements.append(Image(img_buffer))
                elements.append(Spacer(1, 12))
        
        if 'liabilities' in projections:
            elements.append(Paragraph(
                'Liabilities Projection',
                self.custom_styles['Heading3']
            ))
            self._add_projection_table(elements, projections['liabilities'])
            elements.append(Spacer(1, 12))
            
            # Add liabilities chart
            chart = self.generate_line_chart(
                pd.DataFrame(projections['liabilities']),
                'Liabilities Projection Over Time',
                'Liabilities ($bn)'
            )
            if chart:
                img_buffer = BytesIO()
                chart.write_image(img_buffer, format='png')
                img_buffer.seek(0)
                elements.append(Image(img_buffer))
                elements.append(Spacer(1, 12))

    def _add_projection_table(self, elements: list, projection_data: list):
        """Add a projection data table."""
        if not projection_data:
            return
        
        # Convert data to DataFrame for easier handling
        df = pd.DataFrame(projection_data)
        
        # Create table data
        data = [df.columns.tolist()]  # Headers
        data.extend(df.values.tolist())  # Data rows
        
        # Calculate column widths (distribute available space)
        page_width = A4[0] - 1.5*inch  # Account for margins
        col_width = page_width / len(df.columns)
        
        # Create table
        table = Table(data, colWidths=[col_width]*len(df.columns))
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BOX', (0, 0), (-1, -1), 2, colors.black),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        
        elements.append(table)

    def _add_narratives(self, elements: list, narratives: dict):
        """Add narrative sections."""
        if 'overall' in narratives:
            elements.append(Paragraph(
                'Overall Analysis',
                self.custom_styles['Heading3']
            ))
            elements.append(Paragraph(
                narratives['overall'],
                self.custom_styles['Normal']
            ))
            elements.append(Spacer(1, 12))
        
        if 'baseline' in narratives:
            elements.append(Paragraph(
                'Baseline Scenario Analysis',
                self.custom_styles['Heading3']
            ))
            elements.append(Paragraph(
                narratives['baseline'],
                self.custom_styles['Normal']
            ))
            elements.append(Spacer(1, 12))
        
        if 'stress' in narratives:
            elements.append(Paragraph(
                'Stress Scenarios Analysis',
                self.custom_styles['Heading3']
            ))
            elements.append(Paragraph(
                narratives['stress'],
                self.custom_styles['Normal']
            ))
            elements.append(Spacer(1, 12))
            
class DocumentGenerator:
    def __init__(self, common_config: Dict, s3_service, bedrock_service, doc_type: str = None):
        """Initialize document generator.
        
        Args:
            common_config: Common application configuration
            s3_service: S3 service instance
            bedrock_service: Bedrock service instance
            doc_type: Type of document to generate
        """
        self.common_config = common_config
        self.s3_service = s3_service
        self.bedrock_service = bedrock_service
        self.doc_type = doc_type
        self.config = self._load_config(doc_type)
        self.cached_data = {}

    #@st.cache_data(ttl=3600)
    def _load_config(_self, doc_type: str = None) -> Dict:
        """Load document generation configuration.
        
        Args:
            doc_type: Type of document to generate (e.g., 'ppnr')
            
        Returns:
            Dict containing merged configuration
        """
        try:
            # Load common config
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            #print(f"Base Path:{base_path}")
            config_dir = os.path.join(base_path, 'config')
            
            common_config_path = os.path.join(config_dir, 'doc_gen_config.json')
            if not os.path.exists(common_config_path):
                raise FileNotFoundError(f"Common config file not found at: {common_config_path}")
                
            with open(common_config_path, 'r', encoding='utf-8') as f:
                common_config = json.load(f)
                
            # If no doc_type specified, return only common config
            if not doc_type:
                return common_config
                
            # Validate document type
            if doc_type not in common_config.get('document_types', {}):
                raise ValueError(f"Invalid document type: {doc_type}")
                
            # Get document-specific config file name
            doc_config_file = common_config['document_types'][doc_type]['config_file']
            doc_config_path = os.path.join(config_dir, doc_config_file)
            
            if not os.path.exists(doc_config_path):
                raise FileNotFoundError(f"Document config file not found at: {doc_config_path}")
                
            # Load document-specific config
            with open(doc_config_path, 'r', encoding='utf-8') as f:
                doc_config = json.load(f)
                
            # Merge configurations
            # Common settings from doc_gen_config.json
            merged_config = {
                'document_type': doc_type,
                'input_settings': common_config['input_settings'],
                'output_settings': common_config['output_settings'],
                'chart_settings': common_config['chart_settings'],
                'document_settings': common_config['document_settings']
            }
            
            # Document specific settings override common settings
            merged_config.update({
                'document_info': doc_config['document_info'],
                'input_settings': {
                    **merged_config['input_settings'],
                    **doc_config['input_settings']
                },
                'output_settings': {
                    **merged_config['output_settings'],
                    **doc_config['output_settings']
                },
                'executive_summary': doc_config['executive_summary'],
                'narrative_templates': doc_config['narrative_templates'],
                'business_sections': doc_config['business_sections']
            })
            
            return merged_config
            
        except Exception as e:
            st.error(f"Error loading configuration: {str(e)}")
            st.error(f"Current working directory: {os.getcwd()}")
            return {}

    def _process_csv_data(self, business_line: str, data_files: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process CSV data for a specific business line.
        
        Args:
            business_line: Name of the business line to process
            data_files: Dictionary of scenario names and their file paths
            
        Returns:
            Tuple of DataFrames (assets_data, liabilities_data)
        """
        try:
            assets_data = []
            liabilities_data = []
            
            # Process base period (Q42023) data
            #print(data_files['base_period'])
            base_content = self.s3_service.get_document_content(data_files['base_period'])
            # Decode the content from bytes to a string
            try:
                base_content = base_content.decode('utf-8')
                #print(f"Here before base_content type: {type(base_content)}")
            except Exception as decode_error:
                print(f"Error decoding base_content: {decode_error}")
                base_content = None

            # Check if base_content is not empty
            if base_content:
                #print("Inside base_content")
                try:
                    # Use StringIO to simulate a file for pandas
                    csv_buffer = StringIO(base_content)
                    base_df = pd.read_csv(csv_buffer, delimiter='|')  # Adjust the delimiter if needed
                    #print("Before base_df")
                    #print(base_df.head())  # Display the DataFrame for debugging
                    # Filter for the specific business line
                    print(f"business_line:[{business_line}]")
                    base_df = base_df[base_df['business'] == business_line]
                    #print("After base_df")
                    # Get 2023Q4 values
                    q4_2023_assets = round(base_df[base_df['level_2'] == 'U10000000']['balance'].sum() / 100000000, 1)  # Convert to billions
                    q4_2023_liabilities = round(base_df[base_df['level_2'] == 'U20000000']['balance'].sum() / 100000000, 1)
                    #print(f"q4_2023_assets:[{q4_2023_assets}], q4_2023_liabilities:[{q4_2023_liabilities}]")
                    
                    # Create initial rows for each scenario
                    scenarios = ['BHC Base', 'BHC Stress', 'Sup Base', 'Sup Sev Adv']
                    for scenario in scenarios:
                        assets_data.append({'Scenario ($bn)': scenario, '2023Q4': q4_2023_assets})
                        liabilities_data.append({'Scenario ($bn)': scenario, '2023Q4': -q4_2023_liabilities})  # Negative for liabilities
                    
                    #print("assets_data")
                    #print(assets_data)
                    #print("liabilities_data")
                    #print(liabilities_data)
                except Exception as read_error:
                    print(f"Error read_csv: {str(read_error)}")
            else:
                print("base_content is empty or invalid")
                raise Exception
            
            # Process scenario data
            for scenario, file_path in data_files['scenarios'].items():
                content = self.s3_service.get_document_content(file_path)
                
                content = content.decode('utf-8')
                csv_buffer = StringIO(content)
                
                if content:
                    df = pd.read_csv(csv_buffer, delimiter='|')
                    df = df[df['business'] == business_line]
                    
                    # Process each quarter
                    quarters = ['2024Q1', '2024Q2', '2024Q3', '2024Q4', 
                              '2025Q1', '2025Q2', '2025Q3', '2025Q4', '2026Q1']
                    
                    for quarter in quarters:
                        #print(f"scenario data for:{scenario}")
                        #print(df.head())
                        quarter_df = df[df['period_id'] == quarter]
                        assets = round(quarter_df[quarter_df['level_2'] == 'U10000000']['balance'].sum() / 100000000, 1)
                        liabilities = round(quarter_df[quarter_df['level_2'] == 'U20000000']['balance'].sum() / 100000000, 1)
                        #print(f"assets:[{assets}], liabilities:[{liabilities}]")
                        # Update scenario rows
                        try:
                            assets_idx = next(i for i, d in enumerate(assets_data) if d['Scenario ($bn)'] == scenario)
                        except Exception as e:
                            print(f"Error assets_idx: {str(e)}")
                        #print(f"assets_idx:[{assets_idx}]")
                        liabilities_idx = next(i for i, d in enumerate(liabilities_data) if d['Scenario ($bn)'] == scenario)
                        #print(f"liabilities_idx:[{liabilities_idx}]")
                        
                        assets_data[assets_idx][quarter] = assets
                        liabilities_data[liabilities_idx][quarter] = -liabilities  # Negative for liabilities
                        #print("assets_data")
                        #print(assets_data)
                        #print("liabilities_data")
                        #print(liabilities_data)
            
            # Convert to DataFrames
            assets_df = pd.DataFrame(assets_data)
            liabilities_df = pd.DataFrame(liabilities_data)
            
            # Calculate 9Q Average
            quarter_cols = ['2024Q1', '2024Q2', '2024Q3', '2024Q4', 
                          '2025Q1', '2025Q2', '2025Q3', '2025Q4', '2026Q1']
            
            assets_df['9Q Avg'] = assets_df[quarter_cols].mean(axis=1)
            liabilities_df['9Q Avg'] = liabilities_df[quarter_cols].mean(axis=1)
            
            #print("assets_df1")
            #print(assets_df)
            #print("liabilities_df1")
            #print(liabilities_df)
            
            # Calculate Dev. from BHC Base
            bhc_base_assets = assets_df[assets_df['Scenario ($bn)'] == 'BHC Base']['9Q Avg'].values[0]
            bhc_base_liabilities = liabilities_df[liabilities_df['Scenario ($bn)'] == 'BHC Base']['9Q Avg'].values[0]
            
            def calc_deviation(row, base_value):
                if row['Scenario ($bn)'] == 'BHC Base':
                    return None
                return ((row['9Q Avg'] / base_value) - 1) * 100 if base_value != 0 else None
            
            assets_df['Dev. from BHC Base'] = assets_df.apply(lambda row: calc_deviation(row, bhc_base_assets), axis=1)
            liabilities_df['Dev. from BHC Base'] = liabilities_df.apply(lambda row: calc_deviation(row, bhc_base_liabilities), axis=1)
            
            #print("assets_df2")
            #print(assets_df)
            #print("liabilities_df2")
            #print(liabilities_df)
            
            # Format the deviation values
            def format_deviation(val):
                if pd.isna(val):
                    return ''
                return f"({abs(val):.1f})%" if val < 0 else f"{val:.1f}%"
            
            assets_df['Dev. from BHC Base'] = assets_df['Dev. from BHC Base'].apply(
                lambda x: format_deviation(x) if pd.notnull(x) else '')
            liabilities_df['Dev. from BHC Base'] = liabilities_df['Dev. from BHC Base'].apply(
                lambda x: format_deviation(x) if pd.notnull(x) else '')
            
            #print("assets_df3")
            #print(assets_df)
            #print("liabilities_df3")
            #print(liabilities_df)
            
            return assets_df, liabilities_df

        except Exception as e:
            st.error(f"Error processing CSV data: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()

    def save_cache(self) -> bool:
        """Save the current cache to S3."""
        try:
            if not self.cached_data:
                return False
                
            # Define json filename
            json_filename = "2024_ppnr_methodology_and_process_overview.json"
            
            # Save locally first
            with open(json_filename, 'w') as f:
                json.dump(self.cached_data, f, indent=2)
                
            # Upload to S3
            s3_path = os.path.join(
                self.config['output_settings']['s3_output_path'],
                json_filename
            ).replace('\\', '/')
            
            success = self.s3_service.upload_document(json_filename, s3_path)
            
            # Clean up local file
            try:
                os.remove(json_filename)
            except:
                pass
                
            return success
            
        except Exception as e:
            st.error(f"Error saving cache: {str(e)}")
            return False
    
    def generate_driver_table(self, driver_config: Dict) -> go.Figure:
        """Generate driver information table."""
        try:
            # Create DataFrame from driver configuration
            df = pd.DataFrame([{
                'Dependent Variable': driver_config['dependent_var'],
                'Independent Variable': driver_config['independent_var'],
                'Lag': driver_config['lag'],
                'Direction': driver_config['direction']
            }])

            # Create Plotly table
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=list(df.columns),
                    fill_color='blue',
                    font=dict(color='white'),
                    align='center'
                ),
                cells=dict(
                    values=[df[col] for col in df.columns],
                    fill_color='white',
                    align='center'
                )
            )])

            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            return fig

        except Exception as e:
            st.error(f"Error generating driver table: {str(e)}")
            return None

    def generate_line_chart(self, data: pd.DataFrame, title: str, y_label: str) -> Optional[go.Figure]:
        """Generate line chart for data visualization."""
        try:
            print("Starting chart generation...")
            print(f"Input data shape: {data.shape}")
            print(f"Columns: {data.columns.tolist()}")
            
            fig = go.Figure()
            
            # Define consistent colors and order for scenarios
            scenario_colors = {
                'BHC Base': '#000000',    # Black
                'BHC Stress': '#0000FF',  # Blue
                'Sup Base': '#006400',    # Dark Green
                'Sup Sev Adv': '#800080'  # Purple
            }
            
            # Get quarters and sort them
            quarters = [col for col in data.columns if col.startswith(('20')) 
                    and col not in ['Scenario ($bn)', '9Q Avg', 'Dev. from BHC Base']]
            quarters.sort()
            print(f"Quarters found: {quarters}")
            
            # Add traces in specific order
            for scenario in scenario_colors.keys():
                print(f"Processing scenario: {scenario}")
                if scenario in data['Scenario ($bn)'].values:
                    scenario_data = data[data['Scenario ($bn)'] == scenario]
                    print(f"Found data for scenario: {scenario}")
                    
                    y_values = []
                    for q in quarters:
                        try:
                            val = scenario_data[q].values[0]
                            val = round(float(val), 1)
                            y_values.append(val)
                        except Exception as val_error:
                            print(f"Error processing value for {q}: {str(val_error)}")
                            y_values.append(None)
                    
                    print(f"Y values for {scenario}: {y_values}")
                    
                    fig.add_trace(go.Scatter(
                        x=quarters,
                        y=y_values,
                        name=scenario,
                        line=dict(
                            color=scenario_colors[scenario],
                            width=1.5
                        ),
                        mode='lines+markers',
                        marker=dict(
                            size=5,
                            symbol='circle'
                        )
                    ))
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,
                    y=0.95,
                    xanchor='center',
                    yanchor='top',
                    font=dict(size=14)
                ),
                xaxis=dict(
                    title="",
                    showgrid=True,
                    gridcolor='lightgray',
                    gridwidth=1,
                    tickmode='array',
                    ticktext=[q.replace('20', '') for q in quarters],
                    tickvals=quarters,
                    tickangle=0,
                    showline=True,
                    linewidth=1,
                    linecolor='black'
                ),
                yaxis=dict(
                    title=dict(
                        text=y_label,
                        font=dict(size=12)
                    ),
                    showgrid=True,
                    gridcolor='lightgray',
                    gridwidth=1,
                    showline=True,
                    linewidth=1,
                    linecolor='black'
                ),
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99,
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    bordercolor='black',
                    borderwidth=1
                ),
                plot_bgcolor='white',
                width=800,
                height=400,
                margin=dict(l=50, r=50, t=50, b=50, pad=4)
            )
            
            print("Chart generated successfully")
            return fig
            
        except Exception as e:
            print(f"Error generating line chart: {str(e)}")
            print(f"Data columns: {data.columns}")
            st.error(f"Error generating line chart: {str(e)}")
            return None

    def _format_data_for_narrative(self, assets_df: pd.DataFrame, liabilities_df: pd.DataFrame) -> str:
        """Format data for narrative generation."""
        try:
            formatted_data = "Assets Projections:\n"
            formatted_data += assets_df.to_string()
            formatted_data += "\n\nLiabilities Projections:\n"
            formatted_data += liabilities_df.to_string()
            return formatted_data
        except Exception as e:
            st.error(f"Error formatting data for narrative: {str(e)}")
            return ""
        
    def generate_narratives(self, section_key: str, data: Dict[str, pd.DataFrame], driver_data: Dict) -> Dict[str, str]:
        """Generate narratives using the Nova model."""
        try:
            narratives = {}
            
            # Get narrative templates from config
            narrative_templates = self.config['narrative_templates']
            
            # Format data for prompt
            formatted_data = self._format_data_for_narrative(
                data['assets'],
                data['liabilities']
            )
            print("formatted_data")
            print(formatted_data)
            # Generate overall narrative
            overall_prompt = self._create_narrative_prompt(
                narrative_templates['overall'],
                formatted_data,
                section_key,
                'overall',
                driver_data
            )
            print("overall_prompt")
            print(overall_prompt)
            narratives['overall'] = self.bedrock_service.invoke_nova_model(overall_prompt)
            
            # Generate baseline narrative
            baseline_prompt = self._create_narrative_prompt(
                narrative_templates['baseline'],
                formatted_data,
                section_key,
                'baseline',
                driver_data
            )
            narratives['baseline'] = self.bedrock_service.invoke_nova_model(baseline_prompt)
            
            # Generate stress narrative
            stress_prompt = self._create_narrative_prompt(
                narrative_templates['stress'],
                formatted_data,
                section_key,
                'stress',
                driver_data
            )
            narratives['stress'] = self.bedrock_service.invoke_nova_model(stress_prompt)
            
            return narratives

        except Exception as e:
            st.error(f"Error generating narratives: {str(e)}")
            return {}

    def _create_narrative_prompt(self, template: Dict, data: str, section_key: str, narrative_type: str, driver_data: pd.DataFrame) -> str:
        """Create specific narrative prompt based on type."""
        try:
            prompt = f"""Please analyze the provided financial projections and driver information to generate detailed CCAR model narratives as per the given example output.

Data to Analyze:
{data}

Drivers: The following driver information is relevant for scenario analysis and narrative generation.
{driver_data}

Example Output provided below: This is how the CCAR users write narratives for the overview business section and baseline, and stress scenario sections. Carefully review and adhere to the same format provided in the example with no deviation.
{template['example']}

Additional Guidelines:
1. Strictly follow the exact formatting shown in the example output with no deviation

Strictly provide only the {narrative_type} narrative following exactly the format shown in the Example Output above."""

            return prompt

        except Exception as e:
            st.error(f"Error creating narrative prompt: {str(e)}")
            return ""    
            
    def generate_document(self) -> Tuple[Optional[str], Optional[str]]:
        """Generate document with all content."""
        try:
            # Define filenames
            base_filename = "2024_ppnr_methodology_and_process_overview"
            docx_filename = f"{base_filename}.docx"
            pdf_filename = f"{base_filename}.pdf"
            json_filename = f"{base_filename}.json"
            
            # Get base path from config
            s3_base_path = self.config['output_settings']['s3_output_path']

            # Generate content if not already cached
            if not self.cached_data:
                self.cached_data = self._generate_complete_cache()

            # Create Word document
            doc = Document()
            self._add_document_content(doc)

            # Save files locally first
            with open(json_filename, 'w') as f:
                json.dump(self.cached_data, f, indent=2)
                
            doc.save(docx_filename)
            
            # Generate PDF using ReportLab
            pdf_generator = ReportLabDocumentGenerator()
            pdf_buffer = pdf_generator.generate_pdf(self.cached_data, self.config)
            
            with open(pdf_filename, 'wb') as pdf_file:
                pdf_file.write(pdf_buffer.getvalue())

            # Upload to S3
            try:
                for filename in [docx_filename, pdf_filename, json_filename]:
                    full_path = os.path.join(s3_base_path, filename).replace('\\', '/')
                    self.s3_service.upload_document(filename, full_path)
                        
            except Exception as e:
                st.error(f"Error uploading files to S3: {str(e)}")
                return None, None
            finally:
                # Clean up local files
                for filename in [docx_filename, pdf_filename, json_filename]:
                    try:
                        os.remove(filename)
                    except:
                        pass

            # Return the full S3 paths
            docx_path = os.path.join(s3_base_path, docx_filename).replace('\\', '/')
            pdf_path = os.path.join(s3_base_path, pdf_filename).replace('\\', '/')
            return docx_path, pdf_path

        except Exception as e:
            st.error(f"Error generating document: {str(e)}")
            return None, None

    def load_latest_cache(self) -> bool:
        """Load the cached data from S3."""
        try:
            # Construct the full S3 path
            json_file = "2024_ppnr_methodology_and_process_overview.json"
            s3_path = os.path.join(
                self.config['output_settings']['s3_output_path'],
                json_file
            ).replace('\\', '/')
            
            # Check if file exists before trying to load it
            if not self.s3_service.file_exists(s3_path):
                return False
            
            # Load the content
            content = self.s3_service.get_document_content(s3_path)
            if content:
                self.cached_data = json.loads(content.decode('utf-8'))
                return True
                
            return False
            
        except Exception as e:
            st.error(f"Error loading cached data: {str(e)}")
            return False

    def get_latest_documents(self) -> Optional[Dict]:
        """Get the existing document versions from S3."""
        try:
            # Get S3 base path from config
            s3_base_path = self.config['output_settings']['s3_output_path']
            docs = {}
            
            base_filename = "2024_ppnr_methodology_and_process_overview"
            for ext in ['.docx', '.pdf', '.json']:
                filename = f"{base_filename}{ext}"
                full_path = os.path.join(s3_base_path, filename).replace('\\', '/')
                
                # Check if file exists
                if self.s3_service.file_exists(full_path):
                    docs[ext[1:]] = {
                        'filename': filename,
                        'url': full_path
                    }
            
            return docs if docs else None

        except Exception as e:
            st.error(f"Error retrieving documents: {str(e)}")
            return None
    
    def _validate_sections(self, cached_data: Dict) -> Tuple[bool, List[str]]:
        """Validate that all required sections are present and have content.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list of missing/incomplete sections)
        """
        missing_sections = []
        
        try:
            # Check executive summary
            if 'executive_summary' not in cached_data:
                missing_sections.append("Executive Summary section is missing")
            elif not cached_data['executive_summary'].get('sections'):
                missing_sections.append("Executive Summary content is incomplete")
                
            # Check business sections
            if 'business_sections' not in cached_data:
                missing_sections.append("Business sections are missing")
                return False, missing_sections
                
            # For each business section, verify required components
            for section_key, section_data in cached_data['business_sections'].items():
                # Check section has topics
                if 'topics' not in section_data:
                    missing_sections.append(f"Section '{section_key}' is missing topics")
                    continue
                    
                for topic_key, topic_data in section_data['topics'].items():
                    # Check required topic components
                    missing_components = []
                    for required in ['title', 'projections', 'narratives']:
                        if required not in topic_data:
                            missing_components.append(required)
                    
                    if missing_components:
                        missing_sections.append(
                            f"Topic '{topic_key}' in section '{section_key}' is missing: {', '.join(missing_components)}"
                        )
                        
                    # Verify projections data exists
                    projections = topic_data.get('projections', {})
                    if not projections:
                        missing_sections.append(f"No projections data for '{section_key}/{topic_key}'")
                    else:
                        for proj_type in ['assets', 'liabilities']:
                            if proj_type not in projections:
                                missing_sections.append(
                                    f"Missing {proj_type} projections for '{section_key}/{topic_key}'"
                                )
                        
                    # Verify narratives exist
                    narratives = topic_data.get('narratives', {})
                    if not narratives:
                        missing_sections.append(f"No narratives for '{section_key}/{topic_key}'")
                    else:
                        for narr_type in ['overall', 'baseline', 'stress']:
                            if narr_type not in narratives:
                                missing_sections.append(
                                    f"Missing {narr_type} narrative for '{section_key}/{topic_key}'"
                                )
            
            is_valid = len(missing_sections) == 0
            return is_valid, missing_sections
            
        except Exception as e:
            missing_sections.append(f"Error during validation: {str(e)}")
            return False, missing_sections
        
    def _add_document_content(self, doc: Document):
        """Add content to the Word document."""
        try:
            print("Adding document content...")
            # Add title
            title = doc.add_heading(self.config['document_info']['title'], 0)
            title.alignment = WD_ALIGN_PARAGRAPH.CENTER

            # Add Executive Summary
            if 'executive_summary' in self.config:
                print("Adding executive summary...")
                doc.add_heading('Executive Summary', level=1)
                for section in self.config['executive_summary']['sections']:
                    doc.add_heading(section['title'], level=2)
                    doc.add_paragraph(section['content'])

            # Add business sections
            if 'business_sections' in self.config:
                print("Adding business sections...")
                for section_key, section_data in self.config['business_sections'].items():
                    print(f"Processing section: {section_key}")
                    self._add_section_content(doc, section_key, section_data)
                    doc.add_page_break()

        except Exception as e:
            print(f"Error in _add_document_content: {str(e)}")
            st.error(f"Error adding document content: {str(e)}")
            raise

    def _add_section_content(self, doc: Document, section_key: str, section_data: Dict):
        """Add section content to document."""
        try:
            print(f"Adding content for section: {section_key}")
            # Add section heading
            doc.add_heading(section_data['title'], level=1)

            # Process topics
            if 'topics' in section_data:
                for topic_key, topic_data in section_data['topics'].items():
                    print(f"Processing topic: {topic_key}")
                    # Add topic heading
                    doc.add_heading(topic_data['title'], level=2)

                    # Add driver information if available
                    self._add_driver_info(doc, topic_data)
                    
                    # Get cached data
                    cached_topic = self.cached_data.get('business_sections', {}).get(section_key, {}).get('topics', {}).get(topic_key, {})

                    # Add projections
                    self._add_projections_content(doc, topic_data, cached_topic)

                    # Add narratives
                    self._add_narratives_content(doc, cached_topic)

        except Exception as e:
            print(f"Error in _add_section_content: {str(e)}")
            st.error(f"Error adding section content: {str(e)}")
            raise

    def _add_driver_info(self, doc: Document, topic_data: Dict):
        """Add driver information table to document."""
        try:
            if 'driver_info' in topic_data:
                print("Adding driver information...")
                doc.add_heading('Driver Information', level=2)
                driver_table = doc.add_table(rows=2, cols=4)
                driver_info = topic_data['driver_info']['table']
                
                # Add headers
                headers = ['Dependent Variable', 'Independent Variable', 'Lag', 'Direction']
                header_row = driver_table.rows[0]
                for i, header in enumerate(headers):
                    cell = header_row.cells[i]
                    cell.text = header
                    cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                
                # Add data
                data_row = driver_table.rows[1]
                data = [
                    driver_info['dependent_var'],
                    driver_info['independent_var'],
                    str(driver_info['lag']),
                    driver_info['direction']
                ]
                
                for i, value in enumerate(data):
                    cell = data_row.cells[i]
                    cell.text = value
                    cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                
                doc.add_paragraph()  # Add spacing
                print("Driver information added successfully")
        except Exception as e:
            print(f"Error adding driver information: {str(e)}")

    def _add_projections_content(self, doc: Document, topic_data: Dict, cached_topic: Dict):
        """Add projections content including tables and charts."""
        try:
            if 'projections' in topic_data and 'projections' in cached_topic:
                print("Adding projections...")
                
                # Handle assets projections
                self._add_projection_type(
                    doc,
                    topic_data,
                    cached_topic,
                    'assets',
                    'Assets ($bn)'
                )
                
                # Handle liabilities projections
                self._add_projection_type(
                    doc,
                    topic_data,
                    cached_topic,
                    'liabilities',
                    'Liabilities ($bn)'
                )
                print("Projections added successfully")
        except Exception as e:
            print(f"Error adding projections content: {str(e)}")

    def _add_projection_type(self, doc: Document, topic_data: Dict, cached_topic: Dict, proj_type: str, y_label: str):
        """Add specific projection type (assets or liabilities) content."""
        try:
            if proj_type in topic_data['projections']:
                print(f"Processing {proj_type} projections...")
                doc.add_heading(topic_data['projections'][proj_type]['title'], level=3)
                
                projection_data = cached_topic['projections'][proj_type]
                if projection_data:
                    print(f"Creating {proj_type} table...")
                    df = pd.DataFrame(projection_data)
                    
                    # Add table
                    table = doc.add_table(rows=len(df)+1, cols=len(df.columns))
                    
                    # Add headers
                    for j, col in enumerate(df.columns):
                        cell = table.cell(0, j)
                        cell.text = str(col)
                        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                    
                    # Add data
                    for i, row in df.iterrows():
                        for j, value in enumerate(row):
                            cell = table.cell(i+1, j)
                            cell.text = str(value)
                            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                    
                    doc.add_paragraph()  # Add spacing
                    print(f"{proj_type} table added")

                    # Add chart
                    print(f"Adding {proj_type} chart...")
                    chart_title = topic_data['projections'][proj_type]['chart_title']
                    fig = self.generate_line_chart(df, chart_title, y_label)
                    
                    if fig:
                        img_bytes = self._create_chart_image(fig)
                        if img_bytes:
                            try:
                                img_stream = BytesIO(img_bytes)
                                doc.add_picture(img_stream, width=Inches(6))
                                doc.add_paragraph()
                                print(f"{proj_type} chart added successfully")
                            except Exception as chart_error:
                                print(f"Error adding {proj_type} chart: {str(chart_error)}")
                                doc.add_paragraph(f"[Chart: {chart_title}]")
                        else:
                            print(f"Failed to create {proj_type} chart image")
                            doc.add_paragraph(f"[Chart: {chart_title}]")
        except Exception as e:
            print(f"Error adding {proj_type} projection: {str(e)}")

    def _add_narratives_content(self, doc: Document, cached_topic: Dict):
        """Add narrative sections to document."""
        try:
            if 'narratives' in cached_topic:
                print("Adding narratives...")
                narratives = cached_topic['narratives']
                
                if 'overall' in narratives:
                    doc.add_heading('Overall Analysis', level=3)
                    doc.add_paragraph(narratives['overall'])
                
                if 'baseline' in narratives:
                    doc.add_heading('Baseline Scenario Analysis', level=3)
                    doc.add_paragraph(narratives['baseline'])
                
                if 'stress' in narratives:
                    doc.add_heading('Stress Scenarios Analysis', level=3)
                    doc.add_paragraph(narratives['stress'])
                    
                print("Narratives added successfully")
        except Exception as e:
            print(f"Error adding narratives: {str(e)}")
    
    def _save_chart_to_document(self, doc: Document, fig: go.Figure, title: str) -> bool:
        """Save chart to document using a simpler approach."""
        try:
            print("Starting chart save process...")
            
            # First convert to PNG bytes
            print("Converting to PNG...")
            try:
                img_bytes = fig.to_image(
                    format="png",
                    width=800,
                    height=400,
                    scale=1.0,
                    engine="kaleido"
                )
                print("Successfully converted to PNG")
            except Exception as png_error:
                print(f"Error converting to PNG: {str(png_error)}")
                # Try with lower quality
                try:
                    print("Trying with lower quality...")
                    img_bytes = fig.to_image(
                        format="png",
                        width=400,
                        height=300,
                        scale=0.5,
                        engine="kaleido"
                    )
                except Exception as retry_error:
                    print(f"Error on retry: {str(retry_error)}")
                    return False

            # Create BytesIO object
            print("Creating BytesIO object...")
            img_stream = BytesIO(img_bytes)
            print("BytesIO object created")

            # Add to document
            print("Adding to document...")
            try:
                doc.add_picture(img_stream, width=Inches(6))
                doc.add_paragraph()  # Add spacing
                print("Added to document successfully")
                return True
            except Exception as doc_error:
                print(f"Error adding to document: {str(doc_error)}")
                return False

        except Exception as e:
            print(f"General error in chart save: {str(e)}")
            return False

    def _create_chart_image(self, fig: go.Figure) -> Optional[bytes]:
        """Create chart image bytes with simplified settings."""
        try:
            # Use minimal layout settings
            fig.update_layout(
                width=800,
                height=400,
                margin=dict(l=50, r=50, t=50, b=50),
                plot_bgcolor='white',
                showlegend=True
            )
            
            # Convert to image with basic settings
            img_bytes = fig.to_image(
                format="png",
                engine="kaleido",
                width=800,
                height=400
            )
            
            return img_bytes
        except Exception as e:
            print(f"Error creating chart image: {str(e)}")
            return None
        
    def _add_driver_table_content(self, table: object, driver_info: Dict):
        """Add content to driver information table."""
        try:
            # Set header row
            header_cells = table.rows[0].cells
            headers = ['Dependent Variable', 'Independent Variable', 'Lag', 'Direction']
            for i, header in enumerate(headers):
                header_cells[i].text = header
                #self._apply_cell_style(header_cells[i], is_header=True)

            # Set data row
            data_cells = table.rows[1].cells
            data_cells[0].text = driver_info['dependent_var']
            data_cells[1].text = driver_info['independent_var']
            data_cells[2].text = str(driver_info['lag'])
            data_cells[3].text = driver_info['direction']

            #for cell in data_cells:
                #self._apply_cell_style(cell, is_header=False)

        except Exception as e:
            st.error(f"Error adding driver table content: {str(e)}")

    def _apply_cell_style(self, cell: object, is_header: bool):
        """Apply style to table cell."""
        try:
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = paragraph.runs[0] if paragraph.runs else paragraph.add_run()
            font = run.font
            font.name = 'Arial'
            font.size = Pt(11)
            
            if is_header:
                font.bold = True
                font.color.rgb = RGBColor(255, 255, 255)
                cell._tc.get_or_add_tcPr().append(
                    parse_xml(r'<w:shd xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:fill="0000FF"/>')
                )
            else:
                font.bold = False
                font.color.rgb = RGBColor(0, 0, 0)

        except Exception as e:
            st.error(f"Error applying cell style: {str(e)}")

    def load_cached_data(self, json_url: str) -> bool:
        """Load cached data from S3 JSON file."""
        try:
            content = self.s3_service.get_document_content(json_url)
            if content:
                self.cached_data = json.loads(content.decode('utf-8'))
                return True
            return False

        except Exception as e:
            st.error(f"Error loading cached data: {str(e)}")
            return False