import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg
from io import BytesIO
import json
import os
from datetime import datetime
import base64
from src.ui.document_generation import DocumentGenerator
from typing import Optional, Dict, List, Tuple

@st.cache_data(ttl=3600)
def load_bank_data_cached(_s3_service, file_pattern: str) -> dict:
    """Cached function to load bank data.
    
    Args:
        _s3_service: S3Service instance (underscore prefix to prevent hashing)
        file_pattern: Pattern for the file to load
        
    Returns:
        dict: Loaded bank data or None if error occurs
    """
    try:
        #print(f"file pattern:{file_pattern}")
        content = _s3_service.get_document_content(file_pattern)
        if content:
            return json.loads(content.decode('utf-8'))
        return None
    except Exception as e:
        st.error(f"Error loading bank data: {str(e)}")
        return None

class ReportGenerator:
    def __init__(self, common_config: dict, s3_service, bedrock_service):
        """Initialize report generator with configurations and services."""
        try:
            self.common_config = common_config
            self.s3_service = s3_service
            
            # Load report configuration
            self.report_config = self._load_report_config()
            if not self.report_config:
                st.error("Failed to load report configuration")
                return
                
            self.driver_values = st.session_state.get('driver_values', {})
            self.nova_analyzer = NovaProAnalyzer(bedrock_service)

            # Initialize session state
            if 'report_data' not in st.session_state:
                st.session_state.report_data = {}
            if 'selected_section' not in st.session_state:
                st.session_state.selected_section = 'executive_summary'
                
        except Exception as e:
            st.error(f"Error initializing report generator: {str(e)}")
            raise

    @staticmethod
    @st.cache_data(ttl=3600)
    def _load_report_config_cached() -> dict:
        """Cached function to load report configuration."""
        try:
            # Get the correct path to config directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(current_dir)),  # go up two levels
                'config',
                'report_config.json'
            )
            
            # Debug log for path verification
            #print(f"Loading report config from: {config_path}")
            
            # Check if file exists
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Report config file not found at: {config_path}")
                
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                #print("config_data")
                #print(config_data)
                return config_data.get('report_config', {})
        except Exception as e:
            st.error(f"Error loading report config: {str(e)}")
            return {}

    def _load_report_config(self) -> dict:
        """Load report configuration using cached function."""
        return self._load_report_config_cached()    

    def _load_bank_data(self, bank_config: dict) -> dict:
        """Load bank data using cached function."""
        try:
            file_pattern = bank_config['file_pattern'].format(
                year=st.session_state.driver_year,
                period=st.session_state.driver_period
            )
            #print(f"File Pattern inside _load_bank_data:{file_pattern}")
            return load_bank_data_cached(self.s3_service, file_pattern)
        except Exception as e:
            st.error(f"Error loading bank data for pattern {file_pattern}: {str(e)}")
            return None

    def _load_all_bank_data(self) -> dict:
        """Load data for all banks."""
        try:
            bank_data = {}
            with st.spinner("Loading bank data..."):
                for bank_type in ['retail', 'group']:
                    for bank_id, bank_config in self.report_config['banks'][bank_type].items():
                        #print(bank_config)
                        data = self._load_bank_data(bank_config)
                        if data:
                            bank_data[f"{bank_type}_{bank_id}"] = {
                                **data,
                                'name': bank_config['name'],
                                'color': bank_config['color']
                            }
            
            if not bank_data:
                st.warning("No bank data was loaded. Please check the data sources.")
                
            return bank_data
            
        except Exception as e:
            st.error(f"Error loading all bank data: {str(e)}")
            return {}
    
    def _create_cover_page(self):
        """Create the report cover page."""
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(
                self.report_config['metadata']['logo_url'],
                width=200
            )
            st.markdown(
                f"<h1 style='text-align: center; color: {self.report_config['formatting']['colors']['primary']}'>"
                f"{self.report_config['metadata']['title']}</h1>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<h2 style='text-align: center; color: {self.report_config['formatting']['colors']['text']}'>"
                f"Banking Industry - {self.driver_values.get('Half Year Period', '')}</h2>",
                unsafe_allow_html=True
            )
            st.markdown("---")
            st.markdown(f"**Date:** {datetime.now().strftime('%B %d, %Y')}")
            st.markdown(f"**Department:** {self.report_config['metadata']['department']}")
            st.markdown(
                f"<p style='color: red;'>{self.report_config['metadata']['confidentiality']}</p>",
                unsafe_allow_html=True
            )

    def _create_executive_summary(self, bank_data_dict):
        """Create the executive summary section using Nova Pro."""
        st.markdown("## Executive Summary")
        
        with st.spinner("Generating executive summary..."):
            summary = self.summary_generator.generate_executive_summary(
                bank_data_dict,
                self.driver_values
            )
            st.markdown(summary)
            
            # Add key metrics visualization
            st.markdown("### Key Metrics Overview")
            cols = st.columns(len(self.report_config['sections']['executive_summary']['key_metrics']))
            for col, metric in zip(cols, self.report_config['sections']['executive_summary']['key_metrics']):
                with col:
                    if 'barclays' in bank_data_dict:
                        value = self._extract_metric_value(bank_data_dict['barclays'], metric)
                        st.metric(
                            label=metric,
                            value=self._format_metric_value(value, metric),
                            delta=self._calculate_metric_change(bank_data_dict['barclays'], metric)
                        )

    @staticmethod
    def _format_metric_value(value: float, metric_type: str) -> str:
        """Format metric value based on type."""
        if metric_type == 'currency':
            return f"£{value:,.0f}m"
        elif metric_type == 'percentage':
            return f"{value:.2f}%"
        return f"{value:,.2f}"
    
    @staticmethod
    def _calculate_change(current: float, prior: float) -> float:
        """Calculate percentage change between values."""
        if prior and prior != 0:
            return ((current - prior) / prior) * 100
        return 0.0

    def _calculate_metric_change(self, bank_data, metric):
        """Calculate metric change between periods."""
        current_value = self._extract_metric_value(bank_data, metric, prior=False)
        prior_value = self._extract_metric_value(bank_data, metric, prior=True)
        
        if prior_value != 0:
            return ((current_value - prior_value) / prior_value) * 100
        return 0

    def _create_pdf_report(self, bank_data_dict):
        """Create PDF report using ReportLab."""
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        styles = getSampleStyleSheet()
        elements = []
        
        # Add cover page
        elements.append(Image(
            self.report_config['metadata']['logo_url'],
            width=200,
            height=100
        ))
        elements.append(Spacer(1, 30))
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            textColor=colors.HexColor(self.report_config['formatting']['colors']['primary']),
            fontSize=24,
            alignment=1
        )
        
        elements.append(Paragraph(
            self.report_config['metadata']['title'],
            title_style
        ))
        elements.append(Spacer(1, 20))
        
        # Add executive summary
        elements.append(Paragraph(
            "Executive Summary",
            styles['Heading1']
        ))
        elements.append(Spacer(1, 12))
        
        elements.append(Paragraph(
            self.report_config['sections']['executive_summary']['template'].format(
                period=self.driver_values.get('Half Year Period', '')
            ),
            styles['Normal']
        ))
        elements.append(Spacer(1, 20))
        
        # Add charts
        for chart_id, chart_config in self.report_config['sections']['comparative_analysis']['charts'].items():
            elements.append(Paragraph(
                chart_config['title'],
                styles['Heading2']
            ))
            elements.append(Spacer(1, 12))
            
            if chart_config['type'] == 'bar':
                fig = self._create_bar_chart(bank_data_dict.values(), chart_config)
            elif chart_config['type'] == 'pie':
                fig = self._create_pie_chart(bank_data_dict.values(), chart_config)
            elif chart_config['type'] == 'line':
                fig = self._create_line_chart(bank_data_dict.values(), chart_config)
            
            # Save chart as image
            img_buffer = BytesIO()
            fig.write_image(img_buffer, format='png')
            img_buffer.seek(0)
            
            elements.append(Image(img_buffer, width=400, height=300))
            elements.append(Spacer(1, 20))
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        return buffer

    def generate_report(self):
        """Generate the complete report."""
        try:
            # Load data if not in session state
            if not st.session_state.report_data:
                with st.spinner("Loading bank data..."):
                    st.session_state.report_data = self._load_all_bank_data()

            # Navigation sidebar
            self._render_navigation()

            # Render selected section
            if st.session_state.selected_section == 'executive_summary':
                self._render_executive_summary()
            elif st.session_state.selected_section == 'competitor_analysis':
                self._render_competitor_analysis()
            elif st.session_state.selected_section == 'market_analysis':
                self._render_market_analysis()
            elif st.session_state.selected_section == 'comparative_analysis':
                self._render_comparative_analysis()

            # Download options
            self._render_download_options()

        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
            st.error("Please check the data and try again.")

    def _render_navigation(self):
        """Render navigation sidebar."""
        st.sidebar.title("Report Navigation")
        
        sections = {
            "executive_summary": "Executive Summary",
            "competitor_analysis": "Competitor Analysis",
            "market_analysis": "Market Analysis",
            "comparative_analysis": "Comparative Analysis"
        }
        
        selected = st.sidebar.radio(
            "Go to Section:",
            list(sections.keys()),
            format_func=lambda x: sections[x]
        )
        
        if selected != st.session_state.selected_section:
            st.session_state.selected_section = selected
            st.rerun()

    @staticmethod
    @st.cache_data(ttl=3600)
    def _create_chart_cached(chart_type: str, data: dict, config: dict, driver_values: dict) -> go.Figure:
        """Cached function to create charts."""
        try:
            if chart_type == "bar":
                return create_bar_chart(data, config, driver_values)
            elif chart_type == "pie":
                return create_pie_chart(data, config, driver_values)
            elif chart_type == "line":
                return create_line_chart(data, config, driver_values)
            return None
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            return None

    def _create_chart(self, chart_type: str, data: dict, config: dict) -> go.Figure:
        """Create chart using cached function."""
        return self._create_chart_cached(chart_type, data, config, self.driver_values)
    
    def _render_key_metrics(self):
        """Render key metrics dashboard section."""
        metrics = self.report_config['sections']['executive_summary']['key_metrics']
        st.subheader("Key Performance Metrics")
        
        # Create columns for metrics
        cols = st.columns(len(metrics))
        
        for col, metric in zip(cols, metrics):
            with col:
                # Get Barclays data (reference bank)
                barclays_data = next(
                    (data for data in st.session_state.report_data.values()
                     if data.get('name') == 'Barclays'),
                    None
                )
                
                if barclays_data:
                    current_value = extract_metric_value(
                        barclays_data,
                        metric,
                        self.driver_values.get('Half Year Period', '')
                    )
                    prior_value = extract_metric_value(
                        barclays_data,
                        metric,
                        self.driver_values.get('Prior Half Year Period', '')
                    )
                    
                    # Calculate change
                    if prior_value != 0:
                        change = ((current_value - prior_value) / prior_value) * 100
                    else:
                        change = 0
                    
                    # Display metric
                    st.metric(
                        label=metric,
                        value=format_value(current_value, metric),
                        delta=f"{change:+.1f}%"
                    )

    def _render_executive_summary(self):
        """Render executive summary section."""
        st.title("Executive Summary")
        
        with st.spinner("Generating executive summary..."):            
            # Add key metrics dashboard
            self._render_key_metrics()
            
            # Add divider
            st.divider()
            
            summary = self.nova_analyzer.generate_analysis(
                'executive_summary',
                st.session_state.report_data,
                self.driver_values
            )
            st.markdown(summary)
            
            

    def _render_competitor_analysis(self):
        """Render competitor analysis section."""
        st.title("Competitor Analysis")
        
        # Bank selector
        bank_names = [data.get('name') for data in st.session_state.report_data.values()]
        selected_bank = st.selectbox("Select Bank for Analysis:", bank_names)
        
        selected_data = next(
            (data for data in st.session_state.report_data.values() 
             if data.get('name') == selected_bank),
            None
        )
        
        if selected_data:
            # Performance metrics
            metrics_cols = st.columns(2)
            with metrics_cols[0]:
                self._render_bank_metrics(selected_data)
            
            with metrics_cols[1]:
                self._render_bank_charts(selected_data)
            
            # Bank analysis
            with st.spinner(f"Analyzing {selected_bank}..."):
                analysis = self.nova_analyzer.generate_analysis(
                    'competitor_analysis',
                    {selected_bank: selected_data},
                    self.driver_values
                )
                st.markdown(analysis)

    def _render_bank_metrics(self, bank_data: dict):
        """Render metrics for a specific bank."""
        st.subheader("Key Metrics")
        metrics = ['Total Income', 'NIM', 'CIR', 'RoTE']
        
        for metric in metrics:
            current_value = extract_metric_value(
                bank_data,
                metric,
                self.driver_values.get('Half Year Period', '')
            )
            prior_value = extract_metric_value(
                bank_data,
                metric,
                self.driver_values.get('Prior Half Year Period', '')
            )
            
            # Calculate change
            if prior_value != 0:
                change = ((current_value - prior_value) / prior_value) * 100
            else:
                change = 0
            
            # Display metric
            st.metric(
                label=metric,
                value=format_value(current_value, metric),
                delta=f"{change:+.1f}%"
            )

    def _render_bank_charts(self, bank_data: dict):
        """Render charts for a specific bank."""
        st.subheader("Performance Trends")
        
        # Create trend chart
        metric = st.selectbox(
            "Select Metric:",
            ['Total Income', 'NIM', 'CIR', 'RoTE']
        )
        
        chart_config = {
            'type': 'line',
            'title': f'{metric} Trend Analysis',
            'metric': metric,
            'height': 300
        }
        
        fig = create_line_chart(
            {bank_data['name']: bank_data},
            chart_config,
            self.driver_values
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
    def _render_market_analysis(self):
        """Render market analysis section."""
        st.title("Market Analysis")
        
        # Market metrics
        st.subheader("Market Overview")
        
        # Market share pie chart
        chart_config = {
            'type': 'pie',
            'title': 'Market Share Analysis',
            'metric': 'Total Income',
            'height': 600
        }
        fig = create_pie_chart(
            st.session_state.report_data,
            chart_config,
            self.driver_values
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Industry trend analysis
        with st.spinner("Analyzing market trends..."):
            trends = self.nova_analyzer.generate_analysis(
                'market_trends',
                st.session_state.report_data,
                self.driver_values
            )
            st.markdown(trends)

    def _render_comparative_analysis(self):
        """Render comparative analysis section."""
        st.title("Comparative Analysis")
        
        # Metric selector
        selected_metric = st.selectbox(
            "Select Metric for Comparison:",
            ['Total Income', 'NIM', 'CIR', 'RoTE']
        )
        
        # Bar chart comparison
        chart_config = {
            'type': 'bar',
            'title': f'{selected_metric} Comparison',
            'metric': selected_metric,
            'height': 400
        }
        
        fig = create_bar_chart(
            st.session_state.report_data,
            chart_config,
            self.driver_values
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Comparative analysis text
        with st.spinner("Generating comparative analysis..."):
            analysis = self.nova_analyzer.generate_analysis(
                'comparative_analysis',
                st.session_state.report_data,
                self.driver_values
            )
            st.markdown(analysis)
    
    def _render_download_options(self):
        """Render download options."""
        st.sidebar.markdown("---")
        st.sidebar.subheader("Download Report")
        
        if st.sidebar.button("Generate PDF Report"):
            with st.spinner("Generating PDF report..."):
                try:
                    pdf_buffer = self._generate_pdf_report()
                    if pdf_buffer:
                        st.sidebar.download_button(
                            "Download PDF",
                            data=pdf_buffer.getvalue(),
                            file_name=f"competitor_analysis_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )
                        st.sidebar.success("PDF generated successfully!")
                    else:
                        st.sidebar.error("Failed to generate PDF. Please check the errors above.")
                except Exception as e:
                    st.sidebar.error(f"Error generating PDF: {str(e)}")
                    
    def _generate_chart_for_pdf(self, chart_type: str, config: dict, data: dict) -> Image:
        """Generate a chart image for PDF using static plotting with controlled dimensions."""
        try:
            import matplotlib.pyplot as plt
            import io
            import numpy as np
            
            # Set figure size to match PDF width (accounting for margins)
            plt.figure(figsize=(8, 5))  # Reduced size to fit PDF
            
            if chart_type == 'bar':
                # Extract data
                current_period = self.driver_values.get('Half Year Period', '')
                prior_period = self.driver_values.get('Prior Half Year Period', '')
                
                banks = []
                current_values = []
                prior_values = []
                
                for bank_data in data.values():
                    if not bank_data or 'data' not in bank_data:
                        continue
                    
                    banks.append(bank_data.get('name', ''))
                    current_val = extract_metric_value(bank_data, config['metric'], current_period)
                    prior_val = extract_metric_value(bank_data, config['metric'], prior_period)
                    current_values.append(current_val)
                    prior_values.append(prior_val)
                
                # Create bar chart
                x = np.arange(len(banks))
                width = 0.35
                
                plt.bar(x - width/2, current_values, width, label=current_period)
                plt.bar(x + width/2, prior_values, width, label=prior_period)
                
                plt.xlabel('Banks')
                plt.ylabel(config['metric'])
                plt.title(config['title'])
                plt.xticks(x, banks, rotation=45, ha='right')
                plt.legend()
                
            elif chart_type == 'pie':
                # Extract data for pie chart
                current_period = self.driver_values.get('Half Year Period', '')
                labels = []
                values = []
                
                total = 0
                for bank_data in data.values():
                    if bank_data and 'data' in bank_data:
                        value = extract_metric_value(bank_data, config['metric'], current_period)
                        total += value
                
                for bank_data in data.values():
                    if not bank_data or 'data' not in bank_data:
                        continue
                    
                    value = extract_metric_value(bank_data, config['metric'], current_period)
                    market_share = (value / total * 100) if total > 0 else 0
                    
                    labels.append(bank_data.get('name', ''))
                    values.append(market_share)
                
                plt.pie(values, labels=labels, autopct='%1.1f%%')
                plt.title(config['title'])
                
            elif chart_type == 'line':
                # Get period sequence
                periods = [
                    self.driver_values.get('Prior 6 Quarter', ''),
                    self.driver_values.get('Prior 5 Quarter', ''),
                    self.driver_values.get('Prior 4 Quarter', ''),
                    self.driver_values.get('Prior 3 Quarter', ''),
                    self.driver_values.get('Prior 2 Quarter', ''),
                    self.driver_values.get('Prior 1 Quarter', ''),
                    self.driver_values.get('Current Quarter End Period', '')
                ]
                
                for bank_data in data.values():
                    if not bank_data or 'data' not in bank_data:
                        continue
                    
                    values = []
                    for period in periods:
                        value = extract_metric_value(bank_data, config['metric'], period)
                        values.append(value)
                    
                    plt.plot(periods, values, marker='o', label=bank_data.get('name', ''))
                
                plt.xlabel('Period')
                plt.ylabel(config['metric'])
                plt.title(config['title'])
                plt.xticks(rotation=45, ha='right')
                plt.legend()
            
            # Adjust layout
            plt.tight_layout()
            
            # Save to BytesIO with controlled DPI
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            # Create reportlab Image with explicit size control
            img = Image(img_buffer)
            # Set maximum width and height for the image in the PDF
            max_width = 400  # points
            max_height = 300  # points
            
            # Calculate aspect ratio
            aspect = img.imageHeight / float(img.imageWidth)
            
            # Determine final dimensions while maintaining aspect ratio
            if img.imageWidth > max_width:
                img.drawWidth = max_width
                img.drawHeight = max_width * aspect
            
            if img.drawHeight > max_height:
                img.drawHeight = max_height
                img.drawWidth = max_height / aspect
                
            return img
            
        except Exception as e:
            print(f"Error generating chart image: {str(e)}")
            return None

    def _clean_text_for_pdf(self, text: str) -> str:
        """Clean and format text for PDF display."""
        if not text:
            return ""
        
        # Remove markdown headers
        text = text.replace('###', '')
        text = text.replace('####', '')
        
        # Remove asterisks
        text = text.replace('**', '')
        
        # Clean up extra spaces
        text = ' '.join(text.split())
        
        # Fix bullet points if present
        text = text.replace('- ', '\n• ')
        
        return text.strip()

    def _format_section_text(self, text: str, styles) -> list:
        """Format section text into properly styled paragraphs with correct heading levels."""
        elements = []
        
        # Split text into paragraphs
        paragraphs = text.split('\n')
        current_text = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                # Process accumulated text if any
                if current_text:
                    elements.append(Paragraph(
                        ' '.join(current_text),
                        styles['Normal']
                    ))
                    current_text = []
                continue
            
            # Handle different heading levels
            if para.startswith('### '):
                # Process accumulated text if any
                if current_text:
                    elements.append(Paragraph(
                        ' '.join(current_text),
                        styles['Normal']
                    ))
                    current_text = []
                
                heading_text = para.replace('### ', '')
                elements.append(Paragraph(
                    heading_text,
                    styles['Heading1']
                ))
                
            elif para.startswith('#### '):
                # Process accumulated text if any
                if current_text:
                    elements.append(Paragraph(
                        ' '.join(current_text),
                        styles['Normal']
                    ))
                    current_text = []
                
                heading_text = para.replace('#### ', '')
                elements.append(Paragraph(
                    heading_text,
                    styles['Heading2']
                ))
                
            elif para.startswith('# '):
                # Process accumulated text if any
                if current_text:
                    elements.append(Paragraph(
                        ' '.join(current_text),
                        styles['Normal']
                    ))
                    current_text = []
                
                heading_text = para.replace('# ', '')
                elements.append(Paragraph(
                    heading_text,
                    styles['Heading3']
                ))
                
            elif para.startswith('- ') or para.startswith('• '):
                # Process accumulated text if any
                if current_text:
                    elements.append(Paragraph(
                        ' '.join(current_text),
                        styles['Normal']
                    ))
                    current_text = []
                
                bullet_text = para.replace('- ', '').replace('• ', '')
                # Remove markdown bold markers
                bullet_text = bullet_text.replace('**', '')
                elements.append(Paragraph(
                    '• ' + bullet_text,
                    styles['Bullet']
                ))
                
            else:
                # Handle regular paragraphs and accumulate text
                # Remove markdown bold markers
                para = para.replace('**', '')
                current_text.append(para)
        
        # Process any remaining text
        if current_text:
            elements.append(Paragraph(
                ' '.join(current_text),
                styles['Normal']
            ))
        
        return elements

    def _create_custom_styles(self):
        """Create custom styles for PDF document."""
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        
        styles = getSampleStyleSheet()
        
        # Create custom styles
        custom_styles = {
            'Title': ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                textColor=colors.HexColor(self.report_config['formatting']['colors']['primary']),
                fontSize=24,
                alignment=1,
                spaceAfter=30
            ),
            'Heading1': ParagraphStyle(
                'CustomHeading1',
                parent=styles['Heading1'],
                fontSize=16,
                textColor=colors.HexColor('#000000'),
                spaceAfter=12,
                spaceBefore=12
            ),
            'Heading2': ParagraphStyle(
                'CustomHeading2',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#000000'),
                spaceAfter=10,
                spaceBefore=10
            ),
            'Heading3': ParagraphStyle(
                'CustomHeading3',
                parent=styles['Heading3'],
                fontSize=12,
                textColor=colors.HexColor('#000000'),
                spaceAfter=8,
                spaceBefore=8,
                fontName='Helvetica-Bold'
            ),
            'Normal': ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=8,
                leading=14,
                alignment=TA_LEFT
            ),
            'Bullet': ParagraphStyle(
                'CustomBullet',
                parent=styles['Normal'],
                fontSize=11,
                leftIndent=20,
                firstLineIndent=0,
                spaceAfter=8,
                spaceBefore=0,
                leading=14,
                bulletIndent=10,
                alignment=TA_LEFT
            )
        }
        
        return custom_styles

    def _generate_pdf_report(self) -> BytesIO:
        """Generate comprehensive PDF report with charts and analysis."""
        print("Starting PDF generation...")
        try:
            from reportlab.lib.units import inch
            from reportlab.platypus import PageBreak
            
            buffer = BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=0.75*inch,
                leftMargin=0.75*inch,
                topMargin=0.75*inch,
                bottomMargin=0.75*inch
            )
            
            # Get custom styles
            styles = self._create_custom_styles()
            elements = []
            
            # Cover page
            print("Creating cover page...")
            try:
                elements.append(Paragraph(
                    self.report_config['metadata']['title'],
                    styles['Title']
                ))
                
                elements.append(Paragraph(
                    f"Banking Industry Analysis - {self.driver_values.get('Half Year Period', '')}",
                    styles['Heading1']
                ))
                
                elements.append(Paragraph(
                    f"Generated on: {datetime.now().strftime('%B %d, %Y')}",
                    styles['Normal']
                ))
                elements.append(Paragraph(
                    f"Department: {self.report_config['metadata']['department']}",
                    styles['Normal']
                ))
                elements.append(Paragraph(
                    self.report_config['metadata']['confidentiality'],
                    ParagraphStyle(
                        'Confidential',
                        parent=styles['Normal'],
                        textColor=colors.red
                    )
                ))
                
                elements.append(PageBreak())
                print("Cover page created successfully")
            except Exception as e:
                print(f"Error in cover page creation: {str(e)}")
                raise

            print("Creating analysis sections with integrated charts...")
            # Analysis Sections with their corresponding charts
            section_chart_mapping = {
                'executive_summary': [
                    ('market_share', {
                        'type': 'pie',
                        'title': 'Market Share Analysis',
                        'metric': 'Total Income'
                    })
                ],
                'market_trends': [
                    ('efficiency_trends', {
                        'type': 'line',
                        'title': 'Cost Income Ratio Trends',
                        'metric': 'CIR'
                    })
                ],
                'comparative_analysis': [
                    ('income_comparison', {
                        'type': 'bar',
                        'title': 'Total Income Comparison',
                        'metric': 'Total Income'
                    }),
                    ('profitability', {
                        'type': 'bar',
                        'title': 'Return on Tangible Equity',
                        'metric': 'RoTE'
                    })
                ]
            }

            analysis_sections = [
                ('Executive Summary', 'executive_summary'),
                ('Market Analysis', 'market_trends'),
                ('Comparative Analysis', 'comparative_analysis'),
                ('Strategic Recommendations', 'strategic_recommendations')
            ]

            for section_title, analysis_type in analysis_sections:
                print(f"Processing section: {section_title}")
                elements.append(Paragraph(section_title, styles['Heading1']))
                
                # Add relevant charts for this section
                if analysis_type in section_chart_mapping:
                    print(f"Adding charts for {section_title}")
                    for chart_id, chart_config in section_chart_mapping[analysis_type]:
                        try:
                            elements.append(Paragraph(chart_config['title'], styles['Heading3']))
                            
                            chart_img = self._generate_chart_for_pdf(
                                chart_config['type'],
                                chart_config,
                                st.session_state.report_data
                            )
                            
                            if chart_img:
                                elements.append(chart_img)
                                elements.append(Spacer(1, 20))
                                print(f"Chart {chart_config['title']} added successfully")
                        except Exception as chart_error:
                            print(f"Error processing chart {chart_config['title']}: {str(chart_error)}")
                            elements.append(Paragraph(
                                f"Error generating chart: {chart_config['title']}",
                                styles['Normal']
                            ))
                
                # Add analysis text
                analysis = self.nova_analyzer.generate_analysis(
                    analysis_type,
                    st.session_state.report_data,
                    self.driver_values
                )
                
                if analysis:
                    # Format the section text
                    section_elements = self._format_section_text(analysis, styles)
                    elements.extend(section_elements)
                else:
                    print(f"No analysis generated for {section_title}")
                    elements.append(Paragraph(
                        f"Analysis not available for {section_title}",
                        styles['Normal']
                    ))
                
                elements.append(PageBreak())
                print(f"Section {section_title} completed")
            
            print("Building final PDF...")
            # Build PDF with error handling
            try:
                doc.build(elements)
                buffer.seek(0)
                print("PDF built successfully")
                return buffer
            except Exception as build_error:
                print(f"Error building PDF: {str(build_error)}")
                st.error(f"Error building PDF: {str(build_error)}")
                return None
                
        except Exception as e:
            print(f"Error in PDF generation: {str(e)}")
            st.error(f"Error generating PDF report: {str(e)}")
            return None
            
@st.cache_data(ttl=3600)
def create_bar_chart(data: dict, config: dict, driver_values: dict) -> go.Figure:
    """Create a bar chart with current vs prior period comparison."""
    try:
        current_period = driver_values.get('Half Year Period', '')
        prior_period = driver_values.get('Prior Half Year Period', '')
        
        banks = []
        current_values = []
        prior_values = []
        colors = []
        
        for bank_id, bank_data in data.items():
            if not bank_data or 'data' not in bank_data:
                continue
                
            banks.append(bank_data.get('name', ''))
            colors.append(bank_data.get('color', '#000000'))
            
            # Extract values for current and prior periods
            current_val = extract_metric_value(bank_data, config['metric'], current_period)
            prior_val = extract_metric_value(bank_data, config['metric'], prior_period)
            
            current_values.append(current_val)
            prior_values.append(prior_val)

        # Create figure
        fig = go.Figure()
        
        # Add current period bars
        fig.add_trace(go.Bar(
            name=current_period,
            x=banks,
            y=current_values,
            marker_color=config.get('colors', {}).get('current', '#0051A2'),
            text=[format_value(val, config['metric']) for val in current_values],
            textposition='auto'
        ))
        
        # Add prior period bars
        fig.add_trace(go.Bar(
            name=prior_period,
            x=banks,
            y=prior_values,
            marker_color=config.get('colors', {}).get('prior', '#006A4D'),
            text=[format_value(val, config['metric']) for val in prior_values],
            textposition='auto'
        ))
        
        # Update layout
        fig.update_layout(
            title=config.get('title', ''),
            barmode='group',
            height=config.get('height', 400),
            xaxis_title="Banks",
            yaxis_title=config['metric'],
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating bar chart: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def create_pie_chart(data: dict, config: dict, driver_values: dict) -> go.Figure:
    """Create a pie chart showing market share."""
    try:
        current_period = driver_values.get('Half Year Period', '')
        
        labels = []
        values = []
        colors = []
        
        # Calculate total for market share
        total = 0
        for bank_data in data.values():
            if bank_data and 'data' in bank_data:
                value = extract_metric_value(bank_data, config['metric'], current_period)
                total += value
        
        # Calculate individual shares
        for bank_data in data.values():
            if not bank_data or 'data' not in bank_data:
                continue
                
            value = extract_metric_value(bank_data, config['metric'], current_period)
            market_share = (value / total * 100) if total > 0 else 0
            
            labels.append(bank_data.get('name', ''))
            values.append(market_share)
            colors.append(bank_data.get('color', '#000000'))
        
        # Create figure
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            text=[f"{v:.1f}%" for v in values],
            textinfo='label+text',
            hole=0.4
        )])
        
        # Update layout
        fig.update_layout(
            title=config.get('title', ''),
            height=config.get('height', 400),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating pie chart: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def create_line_chart(data: dict, config: dict, driver_values: dict) -> go.Figure:
    """Create a line chart showing metric trends."""
    try:
        # Get period sequence
        periods = [
            driver_values.get('Prior 6 Quarter', ''),
            driver_values.get('Prior 5 Quarter', ''),
            driver_values.get('Prior 4 Quarter', ''),
            driver_values.get('Prior 3 Quarter', ''),
            driver_values.get('Prior 2 Quarter', ''),
            driver_values.get('Prior 1 Quarter', ''),
            driver_values.get('Current Quarter End Period', '')
        ]
        
        fig = go.Figure()
        
        for bank_data in data.values():
            if not bank_data or 'data' not in bank_data:
                continue
            
            # Extract values for each period
            values = []
            for period in periods:
                value = extract_metric_value(bank_data, config['metric'], period)
                values.append(value)
            
            # Add line trace for each bank
            fig.add_trace(go.Scatter(
                x=periods,
                y=values,
                name=bank_data.get('name', ''),
                mode='lines+markers',
                line=dict(
                    color=bank_data.get('color', '#000000'),
                    width=2
                ),
                marker=dict(size=8),
                text=[format_value(val, config['metric']) for val in values],
                hovertemplate="%{text}<br>%{x}<extra></extra>"
            ))
        
        # Update layout
        fig.update_layout(
            title=config.get('title', ''),
            height=config.get('height', 400),
            xaxis_title="Period",
            yaxis_title=config['metric'],
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating line chart: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def extract_metric_value(bank_data: dict, metric: str, period: str) -> float:
    """Extract metric value from bank data for a specific period."""
    try:
        if 'data' in bank_data:
            for row in bank_data['data'][1:]:  # Skip header row
                if row[0].strip() == metric:
                    for i, col in enumerate(bank_data['data'][0]):  # Check header row
                        if col.strip() == period:
                            return convert_to_number(row[i])
        return 0
    except Exception:
        return 0

@st.cache_data(ttl=3600)
def convert_to_number(value: any) -> float:
    """Convert string value to number, handling parentheses for negatives."""
    try:
        if isinstance(value, (int, float)):
            return float(value)
        
        if not value or value == '':
            return 0.0
            
        value = str(value).replace(',', '')
        if '(' in value and ')' in value:
            return -float(value.replace('(', '').replace(')', '').replace('%', ''))
        return float(value.replace('%', ''))
    except:
        return 0.0

@st.cache_data(ttl=3600)
def format_value(value: float, metric: str) -> str:
    """Format value based on metric type."""
    if 'Income' in metric or 'Expenses' in metric:
        return f"£{value:,.0f}m"
    elif metric in ['NIM', 'CIR', 'RoTE']:
        return f"{value:.2f}%"
    return f"{value:,.2f}"

class ReportUI:
    def __init__(self, common_config, s3_service, bedrock_service):
        """Initialize the report UI."""
        self.common_config = common_config
        self.s3_service = s3_service
        self.bedrock_service = bedrock_service
        self.doc_generator = None
        self.report_generator = None
    
    def render(self):
        """Render the report interface."""
        st.markdown("""
            <style>
            .stButton > button {
                background-color: #0051A2;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                cursor: pointer;
            }
            .section-header {
                background-color: #006A4D;
                color: white;
                padding: 10px 15px;
                margin-bottom: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            .dataframe th {
                background-color: #006A4D !important;
                color: white !important;
                text-align: center !important;
            }
            .dataframe td {
                padding: 8px !important;
            }
            .stTabs [data-baseweb="tab"] {
                background-color: #f8f9fa;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            .stTabs [data-baseweb="tab"][aria-selected="true"] {
                background-color: #0051A2 !important;
                color: white !important;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Render header
        col1, col2, col3 = st.columns([0.1, 10.5, 1])
        with col2:
            st.markdown("### Report Generation")
        with col3:
            if st.button("← Back", key="summary_back", use_container_width=True):
                st.session_state.current_view = 'main'
                st.rerun()
        
        # Document type selection - exactly two options as per requirements
        doc_type = st.selectbox(
            "Select Document Type:",
            ["", "PPNR Methodology and Process Overview", "Finance Benchmarking"],
            key="doc_type_selector"
        )

        if doc_type == "PPNR Methodology and Process Overview":
            self._render_document()
        elif doc_type == "Finance Benchmarking":
            self._render_finance_benchmarking()
    
    #@st.cache_resource
    def _initialize_doc_generator(self):
        """Initialize document generator if not already initialized."""
        if self.doc_generator is None:
            self.doc_generator = DocumentGenerator(
                self.common_config,
                self.s3_service,
                self.bedrock_service,
                "ppnr"
            )
        return self.doc_generator

    def _render_document(self):
        """Render methodology document generation interface."""
        try:
            # Initialize document generator
            generator = self._initialize_doc_generator()
            
            # Try to load cached data
            has_cache = generator.load_latest_cache()
            
            # Create containers for status and buttons
            header_container = st.container()
            status_container = st.empty()
            validation_container = st.empty()
            
            # Get existing documents
            existing_docs = generator.get_latest_documents()
            
            with header_container:
                # Create three columns: status/info, analysis button, document button
                col1, col2, col3 = st.columns([2, 6, 2])
                
                with col1:
                    if existing_docs:
                        # Show download options for existing documents
                        st.markdown("#### Download Existing Documents")
                        download_col1, download_col2 = st.columns(2)
                        with download_col1:
                            if 'docx' in existing_docs:
                                st.markdown(f"[📄 Download DOCX]({existing_docs['docx']['url']})")
                        with download_col2:
                            if 'pdf' in existing_docs:
                                st.markdown(f"[📑 Download PDF]({existing_docs['pdf']['url']})")
                
                # Analysis Generation Button
                with col2:
                    if has_cache:
                        if st.button("Re-Generate Analysis", key="regenerate_analysis"):
                            with st.spinner("Regenerating analysis..."):
                                # Clear existing cache
                                generator.cached_data = {}
                                # Render sections with new analysis
                                rendered = self._render_document_sections(generator)
                                if rendered:
                                    st.success("Analysis regenerated successfully!")
                                    # Save new cache
                                    generator.save_cache()
                                    st.rerun()
                    else:
                        if st.button("Generate Analysis", key="generate_analysis"):
                            with st.spinner("Generating analysis..."):
                                rendered = self._render_document_sections(generator)
                                if rendered:
                                    st.success("Analysis generated successfully!")
                                    # Save cache
                                    generator.save_cache()
                                    st.rerun()
                
                # Document Generation Button
                with col3:
                    is_valid, missing_items = generator._validate_sections(generator.cached_data) if has_cache else (False, ["No analysis generated"])
                    document_button_label = "Create Document" if not existing_docs else "Re-Create Document"
                    
                    if st.button(document_button_label, key="generate_doc", disabled=not is_valid):
                        with st.spinner("Generating document..."):
                            try:
                                docx_file, pdf_file = generator.generate_document()
                                if docx_file and pdf_file:
                                    st.success("Documents generated successfully!")
                                    st.experimental_rerun()
                                else:
                                    st.error("Failed to generate documents")
                            except Exception as e:
                                st.error(f"Error generating documents: {str(e)}")

            # Show status message
            if has_cache:
                status_container.info("Displaying existing analysis")
            else:
                status_container.warning("No analysis found. Please generate new analysis.")

            # Show content and validation status
            if has_cache:
                self._render_document_sections(generator)
                
                # Show validation status
                if not is_valid:
                    validation_message = "⚠️ The following sections are incomplete or missing:\n\n"
                    for item in missing_items:
                        validation_message += f"- {item}\n"
                    validation_container.error(validation_message)
                else:
                    validation_container.success("✅ All sections are complete and valid")

            # Add a divider before the main content
            st.markdown("---")

        except Exception as e:
            st.error(f"Error rendering methodology document: {str(e)}")    
    
    def _render_document_sections(self, generator):
        """Render all document sections as expanders."""
        try:
            if not generator.config:
                st.error("Document configuration not loaded")
                return
                
            # First render executive summary
            if 'executive_summary' in generator.config:
                st.header("Executive Summary")
                for section in generator.config['executive_summary']['sections']:
                    st.subheader(section['title'])
                    st.write(section['content'])

            # Initialize section data if no cache exists
            if not generator.cached_data:
                generator.cached_data = {
                    'executive_summary': generator.config['executive_summary'],
                    'business_sections': {}
                }
                
            # Handle business sections
            for section_key, section_config in generator.config['business_sections'].items():
                print(f"section_key:{section_key}")
                print(f"Generating for: {section_config['title']}")
                with st.expander(section_config['title']):
                    try:
                        # Ensure section exists in cache
                        if section_key not in generator.cached_data['business_sections']:
                            generator.cached_data['business_sections'][section_key] = {
                                'title': section_config['title'],
                                'topics': {}
                            }
                            
                        # Process each topic
                        for topic_key, topic_config in section_config['topics'].items():
                            st.subheader(topic_config['title'])
                            
                            # Initialize topic data
                            if topic_key not in generator.cached_data['business_sections'][section_key]['topics']:
                                generator.cached_data['business_sections'][section_key]['topics'][topic_key] = {
                                    'title': topic_config['title'],
                                    'driver_info': topic_config.get('driver_info', {}),
                                    'projections': {},
                                    'narratives': {}
                                }
                            
                            # Render driver information
                            if 'driver_info' in topic_config:
                                st.markdown("#### Driver Information")
                                driver_info = topic_config['driver_info']['table']
                                #print(f"Driver info type:{type(driver_info)}")
                                #print(driver_info)
                                driver_table = generator.generate_driver_table(topic_config['driver_info']['table'])
                                if driver_table:
                                    st.plotly_chart(driver_table)
                                    
                            # Process and render data
                            try:
                                data = generator._process_csv_data(
                                    business_line=section_config['title'],
                                    data_files={
                                        'base_period': generator.config['input_settings']['base_period_file'],
                                        'scenarios': generator.config['input_settings']['scenario_files']
                                    }
                                )
                                
                                if data and len(data) == 2:
                                    assets_df, liabilities_df = data
                                    
                                    # Store and render projections
                                    if 'projections' in topic_config:
                                        topic_data = generator.cached_data['business_sections'][section_key]['topics'][topic_key]
                                        
                                        # Handle assets projection
                                        if 'assets' in topic_config['projections']:
                                            st.markdown(f"#### {topic_config['projections']['assets']['title']}")
                                            st.dataframe(assets_df)
                                            topic_data['projections']['assets'] = assets_df.to_dict('records')
                                            
                                            print("Before plotting chart")
                                            print(assets_df)
                                            # Create assets chart
                                            assets_chart = generator.generate_line_chart(
                                                assets_df,
                                                topic_config['projections']['assets']['chart_title'],
                                                "Assets ($bn)"
                                            )
                                            if assets_chart:
                                                st.plotly_chart(assets_chart)
                                        
                                        # Handle liabilities projection
                                        if 'liabilities' in topic_config['projections']:
                                            st.markdown(f"#### {topic_config['projections']['liabilities']['title']}")
                                            st.dataframe(liabilities_df)
                                            topic_data['projections']['liabilities'] = liabilities_df.to_dict('records')
                                            
                                            # Create liabilities chart
                                            liabilities_chart = generator.generate_line_chart(
                                                liabilities_df,
                                                topic_config['projections']['liabilities']['chart_title'],
                                                "Liabilities ($bn)"
                                            )
                                            if liabilities_chart:
                                                st.plotly_chart(liabilities_chart)
                                                
                                    # Generate and store narratives
                                    narratives = generator.generate_narratives(
                                        section_key,
                                        {'assets': assets_df, 'liabilities': liabilities_df},
                                        driver_info
                                    )
                                    
                                    if narratives:
                                        st.subheader("Analysis")
                                        if 'overall' in narratives:
                                            st.markdown(narratives['overall'])
                                        
                                        st.subheader("Baseline Scenario")
                                        if 'baseline' in narratives:
                                            st.markdown(narratives['baseline'])
                                        
                                        st.subheader("Stress Scenarios")
                                        if 'stress' in narratives:
                                            st.markdown(narratives['stress'])
                                            
                                        # Store narratives in cache
                                        generator.cached_data['business_sections'][section_key]['topics'][topic_key]['narratives'] = narratives
                                        
                            except Exception as data_error:
                                st.error(f"Error processing data for {section_key}/{topic_key}: {str(data_error)}")
                                
                    except Exception as section_error:
                        st.error(f"Error rendering section {section_key}: {str(section_error)}")

            # Return True if all sections were rendered successfully
            return True

        except Exception as e:
            st.error(f"Error rendering document sections: {str(e)}")
            st.error(f"Config keys available: {list(generator.config.keys())}")
            return False    
    
    def _render_finance_benchmarking(self):
        """Render existing finance benchmarking report interface."""
        try:
            if self.report_generator is None:
                self.report_generator = ReportGenerator(
                    self.common_config,
                    self.s3_service,
                    self.bedrock_service
                )

            # Using existing report generation functionality
            self.report_generator.generate_report()
            
        except Exception as e:
            st.error(f"Error generating finance benchmarking report: {str(e)}")
            st.error("Please check the data and try again.")

@st.cache_data(ttl=3600)
def generate_cached_analysis(_bedrock_service, analysis_type: str, data: dict, driver_values: dict) -> str:
    """Generate analysis using Nova Pro with caching.
    
    Args:
        _bedrock_service: Bedrock service instance (underscore prefix to prevent hashing)
        analysis_type: Type of analysis to generate
        data: Bank data for analysis
        driver_values: Driver values from session state
        
    Returns:
        str: Generated analysis text
    """
    try:
        prompts = get_analysis_prompts()
        if analysis_type not in prompts:
            return "Invalid analysis type"
            
        data_summary = prepare_data_summary(data, driver_values)
        prompt = prompts[analysis_type].format(
            period=driver_values.get('Half Year Period'),
            prior_period=driver_values.get('Prior Half Year Period'),
            data_summary=data_summary
        )
        
        return _bedrock_service.invoke_nova_model(prompt)
        
    except Exception as e:
        st.error(f"Error generating analysis: {str(e)}")
        return "Error generating analysis. Please try again."

@st.cache_data(ttl=3600)
def prepare_data_summary(data: dict, driver_values: dict) -> str:
    """Prepare data summary for prompts with caching."""
    try:
        summary = []
        metrics = ['Total Income', 'NIM', 'CIR', 'RoTE']
        current_period = driver_values.get('Half Year Period', '')
        prior_period = driver_values.get('Prior Half Year Period', '')
        
        for bank_id, bank_data in data.items():
            bank_summary = [f"\n{bank_data.get('name', 'Unknown Bank')}:"]
            
            for metric in metrics:
                values = extract_metric_values(bank_data, metric, current_period, prior_period)
                if values:
                    bank_summary.append(f"{metric}: {values['current']} (vs {values['prior']})")
            
            summary.extend(bank_summary)
        
        return "\n".join(summary)
        
    except Exception as e:
        st.error(f"Error preparing data summary: {str(e)}")
        return ""

@st.cache_data(ttl=3600)
def extract_metric_values(bank_data: dict, metric: str, current_period: str, prior_period: str) -> dict:
    """Extract metric values from bank data with caching."""
    try:
        values = {'current': None, 'prior': None}
        
        if 'data' in bank_data and len(bank_data['data']) > 1:
            headers = bank_data['data'][0]
            for row in bank_data['data'][1:]:
                if row[0].strip() == metric:
                    for i, header in enumerate(headers):
                        if header.strip() == current_period:
                            values['current'] = row[i]
                        elif header.strip() == prior_period:
                            values['prior'] = row[i]
        
        return values if values['current'] and values['prior'] else None
        
    except Exception as e:
        st.error(f"Error extracting metric values: {str(e)}")
        return None

def get_analysis_prompts() -> dict:
    """Get analysis prompts for different report sections."""
    return {
        'executive_summary': """
        As a senior financial analyst at Barclays, create a comprehensive executive summary for our competitor analysis report.
        
        Analysis Period: 
        - Current Period: {period}
        - Previous Period: {prior_period}
        
        Bank Performance Data:
        {data_summary}

        Provide a detailed analysis covering:
        1. Overall Industry Performance:
            - Industry-wide revenue and profitability trends
            - Key shifts in competitive dynamics
            - Notable market developments

        2. Competitive Positioning:
            - Market share analysis
            - Leadership in key performance metrics
            - Competitive advantages/disadvantages
            - Notable market position changes

        3. Critical Metrics Analysis:
            - Total Income trends and drivers
            - Net Interest Margin (NIM) dynamics and pressure points
            - Cost Income Ratio (CIR) efficiency trends
            - Return on Tangible Equity (RoTE) performance
            - Key variance explanations

        4. Strategic Implications:
            - Emerging opportunities and challenges
            - Key risk factors
            - Areas requiring strategic focus
            - Competitive response recommendations

        Format the response in clear, professional paragraphs with specific data points.
        Focus on actionable insights and strategic implications rather than just numbers.
        """,

        'competitor_analysis': """
        Analyze the competitive performance for {period} based on the following data:
        
        {data_summary}

        Provide a detailed analysis covering:
        1. Performance Overview:
            - Key metric comparisons
            - Notable outperformers/underperformers
            - Significant performance shifts
            - Market share dynamics

        2. Strategic Focus Areas:
            - Revenue growth strategies
            - Cost management initiatives
            - Balance sheet optimization
            - Digital transformation progress

        3. Business Model Analysis:
            - Income diversification
            - Cost efficiency measures
            - Risk management approach
            - Market positioning

        4. Competitive Dynamics:
            - Key differentiators
            - Market share battles
            - Product innovation
            - Customer acquisition strategies

        Support analysis with specific data points and trend observations.
        Highlight implications for competitive positioning.
        """,

        'market_trends': """
        Based on the performance data from {period}, analyze market trends in the UK banking sector.

        Bank Performance Data:
        {data_summary}

        Provide comprehensive analysis covering:
        1. Market Evolution:
            - Industry structural changes
            - Competitive landscape shifts
            - Regulatory impact analysis
            - Technology disruption effects

        2. Performance Trends:
            - Income and profitability patterns
            - Margin pressure analysis
            - Cost efficiency trends
            - Asset quality developments

        3. Strategic Focus:
            - Common strategic priorities
            - Investment patterns
            - Digital transformation progress
            - Customer engagement approaches

        4. Forward-Looking Analysis:
            - Growth opportunities
            - Emerging challenges
            - Risk factors
            - Success determinants

        Include specific metrics and year-over-year comparisons.
        Focus on implications for competitive strategy.
        """,

        'swot_analysis': """
        Conduct a detailed SWOT analysis based on the following performance data:

        {data_summary}

        Provide a comprehensive SWOT analysis covering:
        1. Strengths:
            - Market leadership positions
            - Competitive advantages
            - Strong performance areas
            - Core capabilities

        2. Weaknesses:
            - Performance gaps
            - Operational challenges
            - Market share losses
            - Efficiency issues

        3. Opportunities:
            - Growth potential
            - Market gaps
            - Innovation possibilities
            - Expansion areas

        4. Threats:
            - Competitive pressures
            - Market challenges
            - Regulatory changes
            - Economic factors

        Support each point with specific data.
        Focus on actionable insights for strategic planning.
        """,

        'key_metrics': """
        Analyze the key performance metrics for {period} based on:

        {data_summary}

        Provide detailed analysis of:
        1. Total Income Performance:
            - Growth rates and trends
            - Income mix analysis
            - Market share implications
            - Key drivers of change

        2. Profitability Metrics:
            - NIM trends and pressure points
            - RoTE performance drivers
            - Efficiency ratio analysis
            - Profitability sustainability

        3. Efficiency Analysis:
            - CIR trends and drivers
            - Cost management effectiveness
            - Operational efficiency
            - Investment impact

        4. Comparative Position:
            - Relative performance
            - Market share changes
            - Competitive advantages
            - Areas for improvement

        Include specific data points and comparative analysis.
        Highlight key performance drivers and implications.
        """,

        'strategic_recommendations': """
        Based on the analysis of {period} performance:

        {data_summary}

        Provide strategic recommendations covering:
        1. Immediate Priorities:
            - Critical focus areas
            - Quick wins
            - Risk mitigation
            - Performance improvement

        2. Medium-term Strategy:
            - Growth initiatives
            - Efficiency programs
            - Digital transformation
            - Market positioning

        3. Long-term Positioning:
            - Sustainable advantages
            - Market leadership
            - Innovation focus
            - Strategic partnerships

        4. Implementation Considerations:
            - Resource requirements
            - Timeline recommendations
            - Risk factors
            - Success metrics

        Provide specific, actionable recommendations.
        Include implementation guidance and expected benefits.
        """,
        
        'comparative_analysis': """
        Analyze the comparative performance across banks for {period} vs {prior_period}:

        {data_summary}

        Provide a detailed comparative analysis covering:

        1. Total Income Analysis:
           - Relative market positions
           - Growth rate comparisons
           - Market share dynamics
           - Key drivers of outperformance/underperformance

        2. Efficiency & Profitability:
           - Cost-Income Ratio (CIR) benchmarking
           - NIM spread analysis
           - Return metrics comparison (RoTE)
           - Operational efficiency insights

        3. Competitive Strengths:
           - Areas of competitive advantage
           - Performance differentiators
           - Strategic positioning
           - Notable success factors

        4. Growth and Scale:
           - Relative growth rates
           - Scale advantages/disadvantages
           - Geographic presence impact
           - Customer base comparisons

        5. Business Mix:
           - Revenue stream diversity
           - Product portfolio comparison
           - Customer segment focus
           - Business model effectiveness

        Provide specific data points to support the analysis.
        Focus on key differentiators and strategic implications.
        Identify clear patterns and competitive dynamics.
        """
    }
        
class NovaProAnalyzer:
    def __init__(self, bedrock_service):
        """Initialize the Nova Pro analyzer."""
        self.bedrock_service = bedrock_service

    def generate_analysis(self, analysis_type: str, data: dict, driver_values: dict) -> str:
        """Generate analysis using cached function."""
        try:
            response = generate_cached_analysis(
                self.bedrock_service, 
                analysis_type, 
                data, 
                driver_values
            )
            return response if response else "No analysis generated."
        except Exception as e:
            return f"Error generating analysis: {str(e)}"

    def generate_executive_summary(self, bank_data_dict, driver_values):
        """Generate comprehensive executive summary using Nova Pro."""
        data_summary = self._prepare_data_summary(bank_data_dict, driver_values)
        
        prompt = f"""
        As a senior financial analyst at Barclays, create a comprehensive executive summary for our competitor analysis report.
        
        Analysis Period: 
        - Current Period: {driver_values.get('Half Year Period')}
        - Previous Period: {driver_values.get('Prior Half Year Period')}
        
        Bank Performance Data:
        {data_summary}

        Provide a detailed analysis covering:
        1. Overall Industry Performance:
           - Revenue trends across the banking sector
           - Key shifts in market dynamics
           - Notable industry-wide developments

        2. Competitive Positioning:
           - Market share changes
           - Leadership in key metrics
           - Areas of competitive advantage/disadvantage

        3. Critical Metrics Analysis:
           - Total Income trends and implications
           - NIM (Net Interest Margin) dynamics
           - Cost Income Ratio (CIR) efficiency analysis
           - Return on Tangible Equity (RoTE) performance

        4. Strategic Implications:
           - Emerging opportunities
           - Potential risks
           - Areas requiring strategic focus

        Format the response in clear, professional paragraphs with specific data points and insights.
        Focus on actionable insights and strategic implications rather than just numbers.
        """

        return self.bedrock_service.invoke_nova_model(prompt)

    def generate_swot_analysis(self, bank_data, competitors_data, driver_values):
        """Generate SWOT analysis using Nova Pro."""
        bank_summary = self._prepare_bank_summary(bank_data, driver_values)
        competitor_summary = self._prepare_data_summary(competitors_data, driver_values)

        prompt = f"""
        Conduct a detailed SWOT analysis for {bank_data.get('name', 'Barclays')} based on the following data:

        Bank Performance:
        {bank_summary}

        Competitor Landscape:
        {competitor_summary}

        Provide a comprehensive SWOT analysis including:

        1. Strengths:
           - Key competitive advantages
           - Areas of market leadership
           - Strong performance metrics

        2. Weaknesses:
           - Performance gaps
           - Operational challenges
           - Areas needing improvement

        3. Opportunities:
           - Market gaps
           - Growth potential
           - Strategic possibilities

        4. Threats:
           - Competitive pressures
           - Market challenges
           - Potential risks

        For each point, provide specific evidence from the data and clear strategic implications.
        Focus on actionable insights that can inform strategic planning.
        """

        return self.bedrock_service.invoke_nova_model(prompt)

    def generate_ceo_commentary_analysis(self, bank_data_dict, driver_values):
        """Analyze CEO commentary and extract key themes using Nova Pro."""
        prompt = f"""
        Analyze the CEO commentary and results presentations for the following banks during {driver_values.get('Half Year Period')}:

        {self._prepare_ceo_commentary_data(bank_data_dict)}

        Provide analysis of:
        1. Key Themes:
           - Common strategic priorities
           - Shared market observations
           - Divergent views on market conditions

        2. Strategic Focus Areas:
           - Investment priorities
           - Digital transformation
           - Customer initiatives
           - Cost management

        3. Market Outlook:
           - Growth expectations
           - Risk assessment
           - Strategic opportunities

        4. Comparative Analysis:
           - Areas of consensus
           - Points of differentiation
           - Unique strategic positions

        Highlight specific quotes where relevant and provide strategic implications of the commentary.
        """

        return self.bedrock_service.invoke_nova_model(prompt)

    def generate_recommendations(self, bank_data_dict, driver_values):
        """Generate strategic recommendations using Nova Pro."""
        data_summary = self._prepare_data_summary(bank_data_dict, driver_values)

        prompt = f"""
        Based on the comprehensive analysis of the UK banking sector for {driver_values.get('Half Year Period')}, 
        generate strategic recommendations.

        Market Context:
        {data_summary}

        Provide detailed recommendations covering:

        1. Strategic Priorities:
           - Key areas requiring immediate focus
           - Medium-term strategic initiatives
           - Long-term positioning goals

        2. Competitive Response:
           - Actions to address competitive threats
           - Opportunities to gain market share
           - Defensive strategies

        3. Performance Enhancement:
           - Revenue growth initiatives
           - Efficiency improvements
           - Risk management strategies

        4. Innovation and Technology:
           - Digital transformation priorities
           - Customer experience enhancements
           - Operational efficiency opportunities

        5. Market Positioning:
           - Brand and product strategy
           - Customer segment focus
           - Geographic expansion opportunities

        For each recommendation:
        - Provide clear rationale based on data
        - Include specific implementation considerations
        - Highlight expected benefits and potential risks
        """

        return self.bedrock_service.invoke_nova_model(prompt)

    def generate_market_outlook(self, bank_data_dict, driver_values):
        """Generate market outlook analysis using Nova Pro."""
        prompt = f"""
        Based on the performance data and trends from {driver_values.get('Half Year Period')}, 
        provide a forward-looking analysis of the UK banking sector.

        Historical Performance:
        {self._prepare_trend_summary(bank_data_dict, driver_values)}

        Analyze and provide insights on:

        1. Market Evolution:
           - Expected structural changes
           - Emerging competitive dynamics
           - Regulatory considerations

        2. Growth Opportunities:
           - High-potential segments
           - Product innovation areas
           - Market expansion possibilities

        3. Risk Landscape:
           - Economic factors
           - Competitive threats
           - Regulatory changes
           - Technology disruption

        4. Success Factors:
           - Critical capabilities
           - Required investments
           - Strategic partnerships

        Provide specific, data-backed insights and clear strategic implications.
        """

        return self.bedrock_service.invoke_nova_model(prompt)

    def _prepare_ceo_commentary_data(self, bank_data_dict):
        """Prepare CEO commentary data from results presentations."""
        # Implementation depends on how CEO commentary is stored in your data
        # This is a placeholder for the actual implementation
        commentary_data = []
        for bank_id, bank_data in bank_data_dict.items():
            if 'ceo_commentary' in bank_data:
                commentary_data.append(f"\n{bank_data['name']} CEO Commentary:\n{bank_data['ceo_commentary']}")
        return "\n".join(commentary_data)

    def generate_competitor_insights(self, bank_data, driver_values):
        """Generate competitor-specific insights using Nova Pro."""
        try:
            data_summary = self._prepare_bank_summary(bank_data, driver_values)
            
            prompt = f"""
            Analyze the following bank data and provide strategic insights:
            
            Bank: {bank_data.get('name', 'Unknown Bank')}
            Period: {driver_values.get('Half Year Period')} vs {driver_values.get('Prior Half Year Period')}
            
            Data:
            {data_summary}
            
            Provide analysis covering:
            1. Key performance highlights
            2. Areas of strength and concern
            3. Strategic positioning
            4. Notable trends
            
            Focus on actionable insights and competitive implications.
            """

            response = self.bedrock_service.invoke_nova_model(prompt)
            return response if response else "Error generating competitor insights."
            
        except Exception as e:
            return f"Error generating competitor insights: {str(e)}"

    def generate_market_trends(self, bank_data_dict, driver_values):
        """Generate market trends analysis using Nova Pro."""
        try:
            data_summary = self._prepare_trend_summary(bank_data_dict, driver_values)
            
            prompt = f"""
            Analyze the following market trend data for the UK banking sector:
            
            Period: {driver_values.get('Half Year Period')}
            Previous Period: {driver_values.get('Prior Half Year Period')}
            
            Data:
            {data_summary}
            
            Please provide a comprehensive analysis of:
            1. Industry-wide trends in key metrics
            2. Competitive dynamics
            3. Notable shifts in market positioning
            4. Forward-looking implications
            
            Include specific data points to support the analysis.
            Focus on meaningful patterns and their strategic significance.
            """

            response = self.bedrock_service.invoke_nova_model(prompt)
            return response if response else "Error generating market trends analysis."
            
        except Exception as e:
            return f"Error generating market trends: {str(e)}"

    @st.cache_data(ttl=3600)
    def _prepare_data_summary(self, data: dict, driver_values: dict) -> str:
        """Prepare data summary for prompts with caching."""
        summary = []
        metrics = ['Total Income', 'NIM', 'CIR', 'RoTE']
        
        for bank_id, bank_data in data.items():
            bank_summary = [f"\n{bank_data.get('name', 'Unknown Bank')}:"]
            current_period = driver_values.get('Half Year Period', '')
            prior_period = driver_values.get('Prior Half Year Period', '')
            
            for metric in metrics:
                values = self._extract_metric_values(bank_data, metric, current_period, prior_period)
                if values:
                    bank_summary.append(f"{metric}: {values['current']} (vs {values['prior']})")
            
            summary.extend(bank_summary)
        
        return "\n".join(summary)

    @staticmethod
    @st.cache_data(ttl=3600)
    def _extract_metric_values(bank_data: dict, metric: str, current_period: str, prior_period: str) -> dict:
        """Extract metric values from bank data with caching."""
        values = {'current': None, 'prior': None}
        
        if 'data' in bank_data and len(bank_data['data']) > 1:
            headers = bank_data['data'][0]
            for row in bank_data['data'][1:]:
                if row[0].strip() == metric:
                    for i, header in enumerate(headers):
                        if header.strip() == current_period:
                            values['current'] = row[i]
                        elif header.strip() == prior_period:
                            values['prior'] = row[i]
        
        return values if values['current'] and values['prior'] else None
    
    def _prepare_bank_summary(self, bank_data, driver_values):
        """Prepare single bank summary for prompt."""
        summary = []
        current_period = driver_values.get('Half Year Period', '')
        prior_period = driver_values.get('Prior Half Year Period', '')
        
        if 'data' in bank_data and len(bank_data['data']) > 1:
            headers = bank_data['data'][0]
            for row in bank_data['data'][1:]:
                metric = row[0].strip()
                current_value = None
                prior_value = None
                
                for i, header in enumerate(headers):
                    if header.strip() == current_period:
                        current_value = row[i]
                    elif header.strip() == prior_period:
                        prior_value = row[i]
                
                if current_value and prior_value:
                    summary.append(f"{metric}: {current_value} (vs {prior_value})")
        
        return "\n".join(summary)

    def _prepare_trend_summary(self, bank_data_dict, driver_values):
        """Prepare trend summary for prompt."""
        summary = []
        periods = [
            driver_values.get('Prior 6 Quarter', ''),
            driver_values.get('Prior 5 Quarter', ''),
            driver_values.get('Prior 4 Quarter', ''),
            driver_values.get('Prior 3 Quarter', ''),
            driver_values.get('Prior 2 Quarter', ''),
            driver_values.get('Prior 1 Quarter', ''),
            driver_values.get('Current Quarter End Period', '')
        ]
        
        metrics = ['Total Income', 'NIM', 'CIR']
        
        for metric in metrics:
            summary.append(f"\n{metric} Trends:")
            for bank_id, bank_data in bank_data_dict.items():
                trend_values = []
                if 'data' in bank_data:
                    for row in bank_data['data'][1:]:
                        if row[0].strip() == metric:
                            for period in periods:
                                for i, header in enumerate(bank_data['data'][0]):
                                    if header.strip() == period:
                                        trend_values.append(row[i])
                
                if trend_values:
                    summary.append(f"{bank_data.get('name', 'Unknown')}: {' -> '.join(trend_values)}")
        
        return "\n".join(summary)
