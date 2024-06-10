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
from difflib import get_close_matches
from Levenshtein import distance as levenshtein_distance

# Load the spaCy English language model
nlp = spacy.load("en_core_web_sm")

# Helper functions...

# Function to autocorrect words using get_close_matches and Levenshtein distance
def autocorrect(word, possibilities):
    word = word.lower()
    closest_matches = get_close_matches(word, possibilities, n=1, cutoff=0.6)
    if closest_matches:
        return closest_matches[0]
    else:
        # If no close match is found, use Levenshtein distance
        closest_match = min(possibilities, key=lambda x: levenshtein_distance(word, x))
        return closest_match

# Function to combine tokens and match with columns
def combine_and_match(tokens, columns):
    combined_attributes = []
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens) + 1):
            combined = ''.join(tokens[i:j]).lower()
            if combined in columns:
                combined_attributes.append(combined)
    return combined_attributes

def extract_requested_columns(response_text):
    # Use regular expressions or other techniques to extract the requested columns
    # from the response text
    pattern = r"columns?\s*for\s*([\w\s,]+)\s*and\s*([\w\s,]+)"
    match = re.search(pattern, response_text, re.IGNORECASE)
    if match:
        column1 = match.group(1).replace(" ", "").split(",")
        column2 = match.group(2).replace(" ", "").split(",")
        requested_columns = column1 + column2
        return requested_columns
    return None

def format_data_for_model(data, requested_columns):
    # Prepare the data in the format requested by the model
    formatted_data = []
    for row in data.to_dict(orient='records'):
        formatted_row = {}
        for column in requested_columns:
            if column in row:
                formatted_row[column] = row[column]
        formatted_data.append(formatted_row)
    return formatted_data

def call_claude3_api(prompt, data=None, max_tokens=2000, chunk_size=1000):
    if data is not None and isinstance(data, pd.DataFrame):
        # Split the data into chunks
        data_chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        responses = []

        # Send a separate request for each data chunk
        for chunk in data_chunks:
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "data": chunk.to_dict(orient='records')
                    }
                ],
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens
            }

            response = st.session_state.bedrock_runtime.invoke_model(
                modelId='anthropic.claude-3-sonnet-20240229-v1:0',
                body=json.dumps(body)
            )

            inference_result = json.loads(response['body'].read()).get("content")[0].get("text")
            responses.append(inference_result)

            # Check if the model requests the data in a specific format
            if "need the data in a tabular format" in inference_result.lower():
                # Identify the requested columns or variables
                requested_columns = extract_requested_columns(inference_result)

                # Prepare the data in the requested format
                formatted_data = format_data_for_model(chunk, requested_columns)

                # Send a new request with the formatted data
                formatted_response = call_claude3_api(prompt, formatted_data, max_tokens, chunk_size)
                responses.append(formatted_response)
                break  # Exit the loop after sending the formatted data

        # Join the responses from each chunk
        combined_response = ' '.join(responses)
        return combined_response

    else:
        # Handle the case when no data or invalid data is provided
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens
        }

        response = st.session_state.bedrock_runtime.invoke_model(
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

def extract_attributes(data, response, entities, chart_type):
    attributes = {}

    # Convert user input and column names to lowercase
    response_lower = response.lower()
    column_names_lower = [col.lower() for col in data.columns]

    # Check if the user input explicitly mentions column names
    column_names_pattern = r"for\s+(\w+)\s+and\s+(\w+)"
    match = re.search(column_names_pattern, response_lower)

    if match:
        column1 = match.group(1).lower()
        column2 = match.group(2).lower()

        # Check if the column names exist in the data frame
        if column1 in column_names_lower and column2 in column_names_lower:
            if chart_type in ['bar', 'line', 'scatter', 'box', 'violin']:
                attributes['x'] = data.columns[column_names_lower.index(column1)]
                attributes['y'] = data.columns[column_names_lower.index(column2)]
            elif chart_type == 'pie':
                attributes['labels'] = data.columns[column_names_lower.index(column1)]
                attributes['values'] = data.columns[column_names_lower.index(column2)]
            elif chart_type == 'histogram':
                attributes['x'] = data.columns[column_names_lower.index(column1)]
            elif chart_type == 'heatmap':
                attributes['x'] = data.columns[column_names_lower.index(column1)]
                attributes['y'] = data.columns[column_names_lower.index(column2)]
                if len(column_names_lower) > 2:
                    column3 = match.group(3).lower()
                    if column3 in column_names_lower:
                        attributes['z'] = data.columns[column_names_lower.index(column3)]
            elif chart_type == 'map':
                attributes['locations'] = data.columns[column_names_lower.index(column1)]
                attributes['values'] = data.columns[column_names_lower.index(column2)]
            # Handle other chart types as needed
        else:
            # If column names are not found, automatically select columns based on chart type
            numeric_columns = data.select_dtypes(include=['number']).columns
            string_columns = data.select_dtypes(include=['object']).columns

            if len(numeric_columns) > 0 and len(string_columns) > 0:
                if chart_type in ['bar', 'line', 'scatter', 'box', 'violin']:
                    attributes['x'] = string_columns[0]
                    attributes['y'] = numeric_columns[0]
                elif chart_type == 'pie':
                    attributes['labels'] = string_columns[0]
                    attributes['values'] = numeric_columns[0]
                elif chart_type == 'histogram':
                    attributes['x'] = numeric_columns[0]
                elif chart_type == 'heatmap':
                    attributes['x'] = string_columns[0]
                    attributes['y'] = string_columns[1]
                    attributes['z'] = numeric_columns[0]
                elif chart_type == 'map':
                    attributes['locations'] = string_columns[0]
                    attributes['values'] = numeric_columns[0]
                # Handle other chart types as needed

    else:
        # Process the response using spaCy
        doc = nlp(response_lower)

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
                potential_x_attrs = [token.text for token in doc if token.text.lower() in column_names_lower and data[data.columns[column_names_lower.index(token.text.lower())]].dtype == 'object']
                x_attr = potential_x_attrs[0] if potential_x_attrs else None

            if not y_attr:
                potential_y_attrs = [token.text for token in doc if token.text.lower() in column_names_lower and data[data.columns[column_names_lower.index(token.text.lower())]].dtype != 'object' and token.text.lower() != x_attr.lower()]
                y_attr = potential_y_attrs[0] if potential_y_attrs else None

            if x_attr and y_attr:
                attributes['x'] = data.columns[column_names_lower.index(x_attr.lower())]
                attributes['y'] = data.columns[column_names_lower.index(y_attr.lower())]

        elif chart_type == 'histogram':
            # Look for potential x attribute
            potential_x_attrs = [token.text for token in doc if token.text.lower() in column_names_lower and data[data.columns[column_names_lower.index(token.text.lower())]].dtype != 'object']
            x_attr = potential_x_attrs[0] if potential_x_attrs else None
            if x_attr:
                attributes['x'] = data.columns[column_names_lower.index(x_attr.lower())]

        elif chart_type == 'pie':
            # Look for potential label and value attributes
            potential_label_attrs = [token.text for token in doc if token.text.lower() in column_names_lower and data[data.columns[column_names_lower.index(token.text.lower())]].dtype == 'object']
            potential_value_attrs = [token.text for token in doc if token.text.lower() in column_names_lower and data[data.columns[column_names_lower.index(token.text.lower())]].dtype != 'object' and token.text.lower() != potential_label_attrs[0].lower()]

            if potential_label_attrs and potential_value_attrs:
                attributes['labels'] = data.columns[column_names_lower.index(potential_label_attrs[0].lower())]
                attributes['values'] = data.columns[column_names_lower.index(potential_value_attrs[0].lower())]

        elif chart_type == 'heatmap':
            # Look for potential x, y, and z attributes
            potential_attrs = [token.text for token in doc if token.text.lower() in column_names_lower]
            if len(potential_attrs) >= 3:
                attributes['x'] = data.columns[column_names_lower.index(potential_attrs[0].lower())]
                attributes['y'] = data.columns[column_names_lower.index(potential_attrs[1].lower())]
                attributes['z'] = data.columns[column_names_lower.index(potential_attrs[2].lower())]

        elif chart_type == 'map':
            # Look for potential location and value attributes
            potential_location_attrs = [token.text for token in doc if token.text.lower() in column_names_lower and ('location' in token.text.lower() or 'city' in token.text.lower() or 'country' in token.text.lower())]
            potential_value_attrs = [token.text for token in doc if token.text.lower() in column_names_lower and token.text.lower() != potential_location_attrs[0].lower()]

            if potential_location_attrs and potential_value_attrs:
                attributes['locations'] = data.columns[column_names_lower.index(potential_location_attrs[0].lower())]
                attributes['values'] = data.columns[column_names_lower.index(potential_value_attrs[0].lower())]

    return attributes

# Generate the chart using the determined chart type and relevant data
def generate_chart(data, chart_type, response, entities, attributes):
    # Generate the chart using libraries like Matplotlib or Plotly
    if chart_type == 'bar':
        x = attributes['x']
        y = attributes['y']
        fig, ax = plt.subplots()
        ax.bar(data[x], data[y])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"Bar Chart for {x} and {y}")
        st.pyplot(fig)

    elif chart_type == 'line':
        x = attributes['x']
        y = attributes['y']
        fig, ax = plt.subplots()
        ax.plot(data[x], data[y])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"Line Chart for {x} and {y}")
        st.pyplot(fig)

    elif chart_type == 'scatter':
        x = attributes['x']
        y = attributes['y']
        fig, ax = plt.subplots()
        ax.scatter(data[x], data[y])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"Scatter Plot for {x} and {y}")
        st.pyplot(fig)

    elif chart_type == 'histogram':
        x = attributes['x']
        fig, ax = plt.subplots()
        ax.hist(data[x], bins=20)
        ax.set_xlabel(x)
        ax.set_ylabel('Frequency')
        ax.set_title(f"Histogram for {x}")
        st.pyplot(fig)

    elif chart_type == 'pie':
        labels = attributes['labels']
        values = attributes['values']
        fig, ax = plt.subplots()
        ax.pie(data[values], labels=data[labels], autopct='%1.1f%%')
        ax.axis('equal')  # To ensure the pie chart is circular
        ax.set_title(f"Pie Chart for {labels} and {values}")
        st.pyplot(fig)

    elif chart_type == 'box':
        x = attributes['x']
        y = attributes['y']
        fig, ax = plt.subplots()
        ax = sns.boxplot(x=x, y=y, data=data)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"Box Plot for {x} and {y}")
        st.pyplot(fig)

    elif chart_type == 'heatmap':
        x = attributes['x']
        y = attributes['y']
        z = attributes['z']
        fig, ax = plt.subplots()
        sns.heatmap(data.pivot(index=x, columns=y, values=z), annot=True, ax=ax)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"Heatmap for {x}, {y}, and {z}")
        st.pyplot(fig)

    elif chart_type == 'violin':
        x = attributes['x']
        y = attributes['y']
        fig, ax = plt.subplots()
        ax = sns.violinplot(x=x, y=y, data=data)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"Violin Plot for {x} and {y}")
        st.pyplot(fig)

    elif chart_type == 'map':
        locations = attributes['locations']
        values = attributes['values']
        fig = px.scatter_mapbox(data, lat=data[locations].apply(lambda loc: float(loc.split(',')[0])),
                                lon=data[locations].apply(lambda loc: float(loc.split(',')[1])),
                                color=values)
        st.plotly_chart(fig)

    else:
        st.write("Chart type not supported or attributes not found.")

# Function to generate overall summary of the data
def generate_summary(data):
    summary = data.describe(include='all').transpose()
    summary['missing_values'] = data.isnull().sum()
    summary['unique_values'] = data.nunique()
    return summary

# Function to generate descriptive summary of the data
def generate_descriptive_summary(data):
    numerical_columns = data.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = data.select_dtypes(exclude=['number']).columns.tolist()
    
    summary = "The dataset contains the following columns:\n\n"
    
    if numerical_columns:
        summary += "Numerical Columns:\n"
        for col in numerical_columns:
            summary += f"- **{col}**: {data[col].describe().to_dict()}\n"
    
    if categorical_columns:
        summary += "\nCategorical Columns:\n"
        for col in categorical_columns:
            summary += f"- **{col}**: {data[col].value_counts().to_dict()}\n"
    
    return summary

# Function to display summary with explanation
def display_summary(summary):
    with st.expander("Overall Data Summary"):
        st.write("This summary includes the following statistics for each column in the dataset:")
        st.write("- **count**: The number of non-null entries")
        st.write("- **mean**: The average of the column (for numerical columns)")
        st.write("- **std**: The standard deviation (for numerical columns)")
        st.write("- **min**: The minimum value (for numerical columns)")
        st.write("- **25%**: The 25th percentile (for numerical columns)")
        st.write("- **50%**: The median or 50th percentile (for numerical columns)")
        st.write("- **75%**: The 75th percentile (for numerical columns)")
        st.write("- **max**: The maximum value (for numerical columns)")
        st.write("- **missing_values**: The number of missing (null) values")
        st.write("- **unique_values**: The number of unique values")
        st.dataframe(summary)

# Initialize AWS services
if 's3' not in st.session_state:
    st.session_state.s3 = boto3.client('s3')

if 'comprehend' not in st.session_state:
    st.session_state.comprehend = boto3.client('comprehend')

if 'bedrock_runtime' not in st.session_state:
    st.session_state.bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

# Set up Streamlit app
st.set_page_config(page_title="Data Analysis Chatbot", layout="wide")
st.title("Data Analysis Chatbot")

# List files from S3 bucket
bucket_name = 'avengers-ba007'
file_list = st.session_state.s3.list_objects_v2(Bucket=bucket_name)['Contents']
file_names = [obj['Key'] for obj in file_list]

# Display file names in Streamlit dropdown
selected_file = st.selectbox('Select a file', file_names)

# Load and process selected file
if selected_file:
    file_obj = st.session_state.s3.get_object(Bucket=bucket_name, Key=selected_file)
    file_content = file_obj['Body'].read()

    # Process the file based on its format (e.g., CSV, Excel, JSON, Parquet)
    if selected_file.endswith('.csv'):
        data = pd.read_csv(BytesIO(file_content))
    elif selected_file.endswith('.xlsx'):
        data = pd.read_excel(BytesIO(file_content))
    elif selected_file.endswith('.json'):
        data = pd.read_json(BytesIO(file_content), orient='records')
    elif selected_file.endswith('.parquet'):
        data = pd.read_parquet(BytesIO(file_content))
    else:
        # Handle other file formats or unsupported formats
        data = None

	# Perform data cleaning and preprocessing
	# ...

    with st.expander("Data Preview"):
        st.write("Here's a preview of your data:")
        st.dataframe(data.head())  # Show a preview of the data

	# Generate data summary (Using Claude-3 Sonnet and AWS Comprehend)
    data_summary_prompt = f"Provide a summary of the following data: {data.to_dict(orient='records')}"
	#st.write(data_summary_prompt)
    data_summary_response = call_claude3_api(data_summary_prompt)
	#data_summary_response = call_titan_api(data_summary_prompt)
    comprehend_output = st.session_state.comprehend.detect_entities(Text=data_summary_response, LanguageCode='en')
    entities = comprehend_output['Entities']
	#st.write(entities)

	# Display data summary and insights in Streamlit
    st.subheader('Data Summary')
    st.write(data_summary_response)
	#st.subheader('Key Entities')
	#for entity in entities:
		#st.write(f"- {entity['Text']} ({entity['Type']})")

	# Automatically display overall summary of the data
    summary = generate_summary(data)
    display_summary(summary)
    
    descriptive_summary = generate_descriptive_summary(data)
    with st.expander("Statistics:"):
        st.write(descriptive_summary)

	# Initiate chat and process user input
    user_input = st.text_input('Enter your query:')

    if user_input:
		# Autocorrect user input
        corrected_input = autocorrect_input(user_input)
		
        st.write(corrected_input)

		# Use Claude-3 Sonnet model to understand the context and generate a response
		#response = call_claude3_api(corrected_input, data.to_dict(orient='records'))
		#response = call_titan_api(corrected_input, data.to_dict(orient='records'))
		# Use Claude-3 Sonnet model to understand the context and generate a response
        if isinstance(data, pd.DataFrame):
            st.write('Inside data')
            response = call_claude3_api(corrected_input, data)
        else:
            st.write('Outside data')
            response = call_claude3_api(corrected_input)

		# Use AWS Comprehend to analyze the user input and response
        comprehend_output = st.session_state.comprehend.detect_entities(Text=response, LanguageCode='en')
        entities = comprehend_output['Entities']
        st.write(entities)

		# Determine the appropriate chart type based on the response and entities
        chart_type = determine_chart_type(response, entities)

		# Extract attributes based on the chart type and response
        attributes = extract_attributes(data, response, entities, chart_type)
	
        st.write(attributes)

		# Display the response and chart in Streamlit
        st.subheader('Response')
        st.write(response)
        st.subheader('Chart')
		#st.pyplot(chart)
		# Generate the chart using the determined chart type and relevant data
        generate_chart(data, chart_type, response, entities, attributes)

