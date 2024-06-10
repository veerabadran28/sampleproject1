import streamlit as st
import boto3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import BytesIO
import json
import re
from collections import Counter
import spacy
from textblob import TextBlob

# Helper function to call the Claude-3 Sonnet API
def call_claude3_api(prompt, data=None):
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000
    }

    if data:
        body["messages"][0]["content"].append({"type": "data", "data": data})

    response = bedrock_runtime.invoke_model(
        modelId='anthropic.claude-3-sonnet-20240229-v1:0',
        body=json.dumps(body)
    )

    inference_result = json.loads(response['body'].read()).get("content")[0].get("text")
    return inference_result

# Helper function to extract keywords from the response
def extract_keywords(response):
    # Remove punctuation and convert to lowercase
    cleaned_response = re.sub(r'[^\w\s]', '', response.lower())

    # Split the response into words
    words = cleaned_response.split()

    # Count word frequencies
    word_counts = Counter(words)

    # Extract keywords based on frequency or other criteria
    keywords = [word for word, count in word_counts.items() if count >= 2]  # Example: Words occurring at least twice

    return keywords

# Determine the appropriate chart type based on the response and entities
def determine_chart_type(response, entities):
    # Use NLP and heuristics to determine the chart type
    # based on keywords, entity types, and context
    # ...
    # Extract keywords and entity types from the response
    keywords = extract_keywords(response)
    entity_types = [entity['Type'] for entity in entities]

    # Check for common chart type keywords
    if any(keyword in ['bar', 'bars', 'column'] for keyword in keywords) or 'QUANTITY' in entity_types:
        chart_type = 'bar'
    elif any(keyword in ['line', 'trend', 'over time'] for keyword in keywords) or 'DATE' in entity_types:
        chart_type = 'line'
    elif any(keyword in ['scatter', 'correlation', 'relationship'] for keyword in keywords):
        chart_type = 'scatter'
    elif any(keyword in ['histogram', 'distribution', 'frequency'] for keyword in keywords):
        chart_type = 'histogram'
    elif any(keyword in ['pie', 'proportion', 'percentage'] for keyword in keywords):
        chart_type = 'pie'
    elif any(keyword in ['box', 'quartile', 'outlier'] for keyword in keywords):
        chart_type = 'box'
    elif any(keyword in ['heatmap', 'matrix', 'correlation matrix'] for keyword in keywords):
        chart_type = 'heatmap'
    elif any(keyword in ['violin', 'distribution', 'density'] for keyword in keywords):
        chart_type = 'violin'
    elif any(keyword in ['map', 'geographic', 'location'] for keyword in keywords) or 'LOCATION' in entity_types:
        chart_type = 'map'
    else:
        chart_type = 'bar'  # Default to bar chart if no specific chart type is detected
        
    return chart_type

# Helper function to autocorrect user input
def autocorrect_input(user_input):
    blob = TextBlob(user_input)
    corrected_input = str(blob.correct())
    return corrected_input

# Helper function to extract relevant attributes from the response and entities
def extract_attributes(data, response, entities, chart_type):
    attributes = {}

    # Process the response using spaCy
    doc = nlp(response)

    # Extract attributes based on chart type and named entities
    if chart_type in ['bar', 'line', 'scatter', 'box', 'violin']:
        # Look for potential x and y attributes
        x_attr = None
        y_attr = None

        for ent in doc.ents:
            if ent.label_ == 'DATE':
                x_attr = ent.text
            elif ent.label_ == 'QUANTITY':
                y_attr = ent.text

        if not x_attr:
            potential_x_attrs = [token.text for token in doc if token.text.lower() in data.columns]
            x_attr = potential_x_attrs[0] if potential_x_attrs else None

        if not y_attr:
            potential_y_attrs = [token.text for token in doc if token.text.lower() in data.columns and token.text.lower() != x_attr.lower()]
            y_attr = potential_y_attrs[0] if potential_y_attrs else None

        attributes['x'] = x_attr
        attributes['y'] = y_attr

    elif chart_type == 'histogram':
        # Look for potential x attribute
        potential_x_attrs = [token.text for token in doc if token.text.lower() in data.columns]
        x_attr = potential_x_attrs[0] if potential_x_attrs else None
        attributes['x'] = x_attr

    elif chart_type == 'pie':
        # Look for potential label and value attributes
        potential_label_attrs = [token.text for token in doc if token.text.lower() in data.columns]
        potential_value_attrs = [token.text for token in doc if token.text.lower() in data.columns and token.text.lower() != potential_label_attrs[0].lower()]

        labels_attr = potential_label_attrs[0] if potential_label_attrs else None
        values_attr = potential_value_attrs[0] if potential_value_attrs else None

        attributes['labels'] = labels_attr
        attributes['values'] = values_attr

    elif chart_type == 'heatmap':
        # Look for potential x, y, and z attributes
        potential_attrs = [token.text for token in doc if token.text.lower() in data.columns]
        x_attr = potential_attrs[0] if potential_attrs else None
        y_attr = potential_attrs[1] if len(potential_attrs) > 1 else None
        z_attr = potential_attrs[2] if len(potential_attrs) > 2 else None

        attributes['x'] = x_attr
        attributes['y'] = y_attr
        attributes['z'] = z_attr

    elif chart_type == 'map':
        # Look for potential location and value attributes
        potential_location_attrs = [token.text for token in doc if token.text.lower() in data.columns and 'location' in token.text.lower()]
        potential_value_attrs = [token.text for token in doc if token.text.lower() in data.columns and token.text.lower() != potential_location_attrs[0].lower()]

        locations_attr = potential_location_attrs[0] if potential_location_attrs else None
        values_attr = potential_value_attrs[0] if potential_value_attrs else None

        attributes['locations'] = locations_attr
        attributes['values'] = values_attr

    return attributes
        
# Generate the chart using the determined chart type and relevant data
def generate_chart(data, chart_type):
    # Extract relevant attributes from the response and entities
    attributes = extract_attributes(data, response, entities, chart_type)

    # Generate the chart using libraries like Matplotlib or Plotly
    if chart_type == 'bar':
        x = attributes['x']
        y = attributes['y']
        chart = px.bar(data, x=x, y=y)
    elif chart_type == 'line':
        x = attributes['x']
        y = attributes['y']
        chart = px.line(data, x=x, y=y)
    elif chart_type == 'scatter':
        x = attributes['x']
        y = attributes['y']
        chart = px.scatter(data, x=x, y=y)
    elif chart_type == 'histogram':
        x = attributes['x']
        chart = px.histogram(data, x=x)
    elif chart_type == 'pie':
        labels = attributes['labels']
        values = attributes['values']
        chart = px.pie(data, values=values, names=labels)
    elif chart_type == 'box':
        x = attributes['x']
        y = attributes['y']
        chart = px.box(data, x=x, y=y)
    elif chart_type == 'heatmap':
        x = attributes['x']
        y = attributes['y']
        z = attributes['z']
        chart = px.imshow(z, x=x, y=y)
    elif chart_type == 'violin':
        x = attributes['x']
        y = attributes['y']
        chart = px.violin(data, x=x, y=y)
    elif chart_type == 'map':
        locations = attributes['locations']
        values = attributes['values']
        chart = px.scatter_mapbox(data, lat=locations.apply(lambda loc: loc.split(',')[0]),
                                  lon=locations.apply(lambda loc: loc.split(',')[1]),
                                  color=values)
    else:
        chart = None  # Handle case where no chart type is detected

    return chart

# Initialize AWS services
s3 = boto3.client('s3')
comprehend = boto3.client('comprehend')
bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

# Set up Streamlit app
st.set_page_config(page_title="Data Analysis Chatbot", layout="wide")
st.title("Data Analysis Chatbot")

# List files from S3 bucket
bucket_name = 'avengers-ba007'
file_list = s3.list_objects_v2(Bucket=bucket_name)['Contents']
file_names = [obj['Key'] for obj in file_list]

# Display file names in Streamlit dropdown
selected_file = st.selectbox('Select a file', file_names)

# Load and process selected file
if selected_file:
    file_obj = s3.get_object(Bucket=bucket_name, Key=selected_file)
    file_content = file_obj['Body'].read()

    # Process the file based on its format (e.g., CSV, Excel, JSON, Parquet)
    if selected_file.endswith('.csv'):
        data = pd.read_csv(BytesIO(file_content))
    elif selected_file.endswith('.xlsx'):
        data = pd.read_excel(BytesIO(file_content))
    elif selected_file.endswith('.json'):
        data = pd.read_json(BytesIO(file_content))
    elif selected_file.endswith('.parquet'):
        data = pd.read_parquet(BytesIO(file_content))

    # Perform data cleaning and preprocessing
    # ...

    # Generate data summary (Using Claude-3 Sonnet and AWS Comprehend)
    data_summary_prompt = f"Provide a summary of the following data: {data.to_dict(orient='records')}"
    data_summary_response = call_claude3_api(data_summary_prompt)
    comprehend_output = comprehend.detect_entities(Text=data_summary_response, LanguageCode='en')
    entities = comprehend_output['Entities']

    # Display data summary and insights in Streamlit
    st.subheader('Data Summary')
    st.write(data_summary_response)
    st.subheader('Key Entities')
    for entity in entities:
        st.write(f"- {entity['Text']} ({entity['Type']})")

    # Initiate chat and process user input
    user_input = st.text_input('Enter your query:')

    if user_input:
        # Autocorrect user input
        corrected_input = autocorrect_input(user_input)

        # Use Claude-3 Sonnet model to understand the context and generate a response
        response = claude.generate(corrected_input, data.to_dict(orient='records'))

        # Use AWS Comprehend to analyze the user input and response
        comprehend_output = comprehend.detect_entities(Text=response, LanguageCode='en')
        entities = comprehend_output['Entities']

        # Determine the appropriate chart type based on the response and entities
        chart_type = determine_chart_type(response, entities)

        # Generate the chart using the determined chart type and relevant data
        chart = generate_chart(data, chart_type, response, entities)

        # Display the response and chart in Streamlit
        st.subheader('Response')
        st.write(response)
        st.subheader('Chart')
        st.pyplot(chart)


# Run the Streamlit app
if __name__ == "__main__":
    st.run()