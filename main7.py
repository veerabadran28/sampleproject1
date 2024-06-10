import streamlit as st
import pandas as pd
import plotly.express as px
import boto3
import json
from io import BytesIO
from difflib import get_close_matches
from Levenshtein import distance as levenshtein_distance
from pygwalker.api.streamlit import StreamlitRenderer

# Initialize AWS services
if 's3' not in st.session_state:
    st.session_state.s3 = boto3.client('s3')

if 'comprehend' not in st.session_state:
    st.session_state.comprehend = boto3.client('comprehend')

if 'bedrock_runtime' not in st.session_state:
    st.session_state.bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

# Set the Streamlit page to wide mode
st.set_page_config(layout="wide")

# Function to load data based on file type
def load_data(file):
    if file.name.endswith('.csv'):
        data = pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        xls = pd.ExcelFile(file)
        sheet_name = st.selectbox('Select sheet', xls.sheet_names) if len(xls.sheet_names) > 1 else xls.sheet_names[0]
        data = pd.read_excel(file, sheet_name=sheet_name)
    elif file.name.endswith('.json'):
        data = pd.read_json(file)
    elif file.name.endswith('.parquet'):
        data = pd.read_parquet(file)
    else:
        data = pd.read_csv(file, delimiter=st.text_input('Enter delimiter', value=','))
    data.columns = data.columns.str.lower()  # Convert column names to lowercase
    return data

# Function to process user query using Claude-3 Sonnet
def process_query_claude3(query):
    prompt = f"Please inform user that the appropriate chart will be displayed below based on your input: {query}"
    #st.write(prompt)
    response = call_claude3_api(prompt)
    #st.write(response)
    comprehend_output = st.session_state.comprehend.detect_entities(Text=response, LanguageCode='en')
    #st.write(comprehend_output)
    entities = comprehend_output['Entities']
    return response, entities

# Function to call Claude-3 Sonnet API
def call_claude3_api(prompt, max_tokens=2000):
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

# Refined function to determine chart type based on query, entities, and Claude-3 Sonnet response
def determine_chart_type(query, entities, response):
    query_lower = query.lower()
    entity_types = [entity['Type'] for entity in entities]

    # Check for explicit chart types in the query
    explicit_chart_types = ["bar", "line", "scatter", "histogram", "pie", "area", "box", "heatmap", "violin", "map"]
    for chart_type in explicit_chart_types:
        if chart_type in query_lower:
            return chart_type

    # Check for explicit chart types in the Claude-3 Sonnet response
    for chart_type in explicit_chart_types:
        if chart_type in response.lower():
            return chart_type

    # Check for implicit chart types based on entities and keywords
    if "QUANTITY" in entity_types or any(keyword in query_lower for keyword in ["quantity", "sales", "amount", "count", "total"]):
        return "bar"
    elif "DATE" in entity_types or any(keyword in query_lower for keyword in ["date", "time", "trend", "over time"]):
        return "line"
    elif "LOCATION" in entity_types or any(keyword in query_lower for keyword in ["location", "map", "geographical"]):
        return "map"
    elif any(keyword in query_lower for keyword in ["distribution", "spread", "range"]):
        return "histogram"
    elif any(keyword in query_lower for keyword in ["proportion", "percentage", "ratio"]):
        return "pie"
    elif any(keyword in query_lower for keyword in ["compare", "difference", "variation"]):
        return "bar"

    # Fallback to bar chart if no specific chart type is detected
    return "bar"

# Function to autocorrect words using Levenshtein distance
def autocorrect(word, possibilities, cutoff=0.6):
    word = word.lower()
    closest_match = min(possibilities, key=lambda x: levenshtein_distance(word, x.lower()))
    
    # Calculate the Levenshtein distance ratio
    distance = levenshtein_distance(word, closest_match.lower())
    max_length = max(len(word), len(closest_match))
    ratio = distance / max_length
    
    # Return the closest match only if the ratio is below the cutoff
    if ratio < cutoff:
        return closest_match
    else:
        return word  # Return the original word if no close match is found

# Function to combine tokens and match with columns
def combine_and_match(tokens, columns):
    combined_attributes = []
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens) + 1):
            combined = ''.join(tokens[i:j]).lower()
            if any(combined == col.lower() for col in columns):
                combined_attributes.append(combined)
    return combined_attributes

# Process user query with AWS Comprehend and Claude-3 Sonnet
def process_query(query, columns):
    response, entities = process_query_claude3(query)

    #st.write('response:', response)
    #st.write('entities:', entities)
    # Determine chart type using Claude-3 Sonnet and AWS Comprehend
    chart_type = determine_chart_type(query, entities, response)
    #st.write("Chart Type: ", chart_type)

    # Autocorrect tokens
    tokens = query.split()
    corrected_tokens = list(set([autocorrect(token, columns) for token in tokens]))
    #st.write('corrected_tokens:', corrected_tokens)

    # Extract and match attributes
    combined_attributes = combine_and_match(corrected_tokens, columns)
    #st.write('combined_attributes:', combined_attributes)

    # Ensure relevant and unique attributes
    relevant_attributes = []
    for attr in combined_attributes:
        matched_col = next((col for col in columns if col.lower() == attr), None)
        if matched_col and matched_col not in relevant_attributes:
            relevant_attributes.append(matched_col)

    #st.write('relevant_attributes1:', relevant_attributes)
    
    # Further autocorrect to find the closest matching columns if needed
    if len(relevant_attributes) < 2:
        for token in tokens:
            possible_corrections = get_close_matches(token, [col.lower() for col in columns], n=1)
            if possible_corrections:
                matched_col = next((col for col in columns if col.lower() == possible_corrections[0]), None)
                if matched_col and matched_col not in relevant_attributes:
                    relevant_attributes.append(matched_col)

    #st.write('relevant_attributes2:', relevant_attributes)
    
    # Limit the number of attributes to 3
    if len(relevant_attributes) > 2:
        relevant_attributes = relevant_attributes[:2]

    # If no attributes found, intelligently identify one string column for x-axis and one numeric column for y-axis
    if not relevant_attributes:
        numeric_columns = data.select_dtypes(include=['number']).columns
        string_columns = data.select_dtypes(include=['object']).columns

        if len(numeric_columns) > 0 and len(string_columns) > 0:
            relevant_attributes = [string_columns[0], numeric_columns[0]]
        else:
            st.error("Unable to identify suitable columns for the chart.")

    #st.write('relevant_attributes3:', relevant_attributes)

    # Ensure string column is on the x-axis and numeric column is on the y-axis
    if relevant_attributes and data[relevant_attributes[0]].dtype == 'object' and data[relevant_attributes[1]].dtype != 'object':
        x_attr, y_attr = relevant_attributes
    else:
        y_attr, x_attr = relevant_attributes

    return chart_type, x_attr, y_attr

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

# Function to get static mapping for country and city names to latitude and longitude
def get_location_coordinates():
    return {
        'USA': {'latitude': 37.0902, 'longitude': -95.7129},
        'France': {'latitude': 46.603354, 'longitude': 1.888334},
        'Japan': {'latitude': 36.204824, 'longitude': 138.252924},
        'Italy': {'latitude': 41.87194, 'longitude': 12.56738},
        'Austria': {'latitude': 47.516231, 'longitude': 14.550072},
        'Australia': {'latitude': -25.274398, 'longitude': 133.775136},
        'Spain': {'latitude': 40.463667, 'longitude': -3.74922},
        'UK': {'latitude': 55.378051, 'longitude': -3.435973},
        'Germany': {'latitude': 51.165691, 'longitude': 10.451526},
        'Singapore': {'latitude': 1.352083, 'longitude': 103.819836},
        'Belgium': {'latitude': 50.503887, 'longitude': 4.469936},
        'Finland': {'latitude': 61.92411, 'longitude': 25.748151},
        'Canada': {'latitude': 56.130366, 'longitude': -106.346771},
        'Norway': {'latitude': 60.472024, 'longitude': 8.468946},
        'Denmark': {'latitude': 56.26392, 'longitude': 9.501785},
        'Switzerland': {'latitude': 46.818188, 'longitude': 8.227512},
        'Ireland': {'latitude': 53.41291, 'longitude': -8.24389},
        'Sweden': {'latitude': 60.128161, 'longitude': 18.643501},
        'Philippines': {'latitude': 12.879721, 'longitude': 121.774017},
        # Add more countries and cities as needed
        'New York': {'latitude': 40.712776, 'longitude': -74.005974},
        'Los Angeles': {'latitude': 34.052235, 'longitude': -118.243683},
        'Chicago': {'latitude': 41.878113, 'longitude': -87.629799},
        'Houston': {'latitude': 29.760427, 'longitude': -95.369804},
        'Phoenix': {'latitude': 33.448376, 'longitude': -112.074036},
        'Paris': {'latitude': 48.856613, 'longitude': 2.352222},
        'Tokyo': {'latitude': 35.689487, 'longitude': 139.691711},
        'Berlin': {'latitude': 52.520008, 'longitude': 13.404954},
        # Continue adding cities
    }

# Function to add latitude and longitude to the dataframe
def add_lat_lon_static(df, location_col):
    coordinates = get_location_coordinates()
    latitudes = []
    longitudes = []
    
    for location in df[location_col]:
        if location in coordinates:
            latitudes.append(coordinates[location]['latitude'])
            longitudes.append(coordinates[location]['longitude'])
        else:
            latitudes.append(None)
            longitudes.append(None)
    
    df['latitude'] = latitudes
    df['longitude'] = longitudes
    return df

# Generate plot with Plotly Express
def generate_plot(result, plot_type, x, y, title):
    if plot_type == 'bar':
        fig = px.bar(result, x=x, y=y, title=title, text=y)
    elif plot_type == 'line':
        fig = px.line(result, x=x, y=y, title=title, text=y)
    elif plot_type == 'scatter':
        fig = px.scatter(result, x=x, y=y, title=title, text=y)
    elif plot_type == 'histogram':
        fig = px.histogram(result, x=x, title=title, text=y)
    elif plot_type == 'pie':
        fig = px.pie(result, names=x, values=y, title=title)
    elif plot_type == 'area':
        fig = px.area(result, x=x, y=y, title=title, text=y)
    elif plot_type == 'box':
        fig = px.box(result, x=x, y=y, title=title, text=y)
    elif plot_type == 'heatmap':
        fig = px.imshow(result.corr(), title=title, text=y)
    elif plot_type == 'violin':
        fig = px.violin(result, x=x, y=y, title=title)
    elif plot_type == 'map':
        # Add latitude and longitude if the country or city column exists
        if 'country' in [x, y] or 'city' in [x, y]:
            if 'country' in [x, y]:
                result = add_lat_lon_static(result, 'country')
            if 'city' in [x, y]:
                result = add_lat_lon_static(result, 'city')
            if 'latitude' in result.columns and 'longitude' in result.columns:
                fig = px.scatter_mapbox(result, text=y, lat='latitude', lon='longitude', title=title, mapbox_style="carto-positron", size=y if result[y].dtype != 'object' else None)
                fig.update_layout(mapbox_zoom=3, mapbox_center={"lat": 37.0902, "lon": -95.7129})
            else:
                st.error("Failed to add latitude and longitude for the country or city names.")
                fig = None
        else:
            st.error("Map chart requires 'country' or 'city' as one of the attributes.")
            fig = None
    return fig

def st_radio_horizontal(*args, **kwargs):
    """Trick to have horizontal st radio to simulate tabs"""
    col, _ = st.columns(2)
    with col:
        st.write('<style> dev[data_testid=column] > div > div > div > div.stRadio > div{flex-direction: row;}</style>', unsafe_allow_html=True)
        return st.radio(*args, **kwargs)

# Streamlit application
st.title("Interactive Data Visualization")

# List files from S3 bucket
bucket_name = 'avengers-ba007'
file_list = st.session_state.s3.list_objects_v2(Bucket=bucket_name)['Contents']
file_names = [obj['Key'] for obj in file_list]
data = pd.DataFrame()

user_selection_option = st_radio_horizontal("Choose Option:", ["Explore My Data!", "Explore data from AWS S3!"])

if user_selection_option == "Explore data from AWS S3!":
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
        columns = data.columns.tolist()

if user_selection_option == "Explore My Data!":
    st.write("Upload your data file and enter your query below:")

    with st.expander("Upload Data File"):
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'json', 'parquet', 'txt'])

    if uploaded_file is not None:    
        data = load_data(uploaded_file)
        columns = data.columns.tolist()

if not data.empty:
    st.write("Data loaded successfully!")
    with st.expander("Data Preview"):
        st.write("Here's a preview of your data:")
        st.dataframe(data)  # Show a preview of the data
    
    data_summary_prompt = f"Provide a summary of the following data: {data.to_dict(orient='records')}"
    data_summary_response = call_claude3_api(data_summary_prompt)
    with st.expander("Data Explanation:"):
        st.write(data_summary_response)
        
    # with st.expander('Quick Insights'):
    #     pyg_app = StreamlitRenderer(data, spec_io_mode="rw")
    #     pyg_app.explorer()
        
    # Automatically display overall summary of the data
    summary = generate_summary(data)
    display_summary(summary)

    descriptive_summary = generate_descriptive_summary(data)
    with st.expander("Statistics:"):
        st.write(descriptive_summary)

    # Text input for user query
    user_query = st.text_input("Query", key='user_query')

    # Process the query only when the button is clicked
    if st.button('Submit'):
        if user_query:
            with st.spinner('Processing...'):
                chart_type, x_attr, y_attr = process_query(user_query, columns)

                if chart_type == "summary" or chart_type == "statistical" or chart_type == "statistics":
                    display_summary(summary)
                elif x_attr and y_attr:
                    # Log the extracted attributes for debugging
                    st.write('chart_type:', chart_type)
                    st.write(f"Extracted attributes: {x_attr}, {y_attr}")
                    try:
                        if "top" in user_query:
                            number = int(''.join(filter(str.isdigit, user_query)))
                            result = data.groupby(y_attr)[x_attr].sum().reset_index().sort_values(by=x_attr, ascending=False).head(number)
                            title = f'Top {x_attr} by {y_attr}'
                        elif "bottom" in user_query:
                            number = int(''.join(filter(str.isdigit, user_query)))
                            result = data.groupby(y_attr)[x_attr].sum().reset_index().sort_values(by=x_attr, ascending=True).head(number)
                            title = f'Bottom {x_attr} by {y_attr}'
                        elif "quantity" in user_query or "sales" in user_query:
                            result = data.groupby(x_attr)[y_attr].sum().reset_index()
                            title = f'{y_attr} vs {x_attr}'
                        elif "date" in user_query or "time" in user_query:
                            result = data.groupby(x_attr)[y_attr].sum().reset_index()
                            title = f'{y_attr} Over Time'
                        elif "map" in user_query:
                            result = data
                            title = f'Map of {x_attr}'
                        else:
                            result = data.groupby(x_attr)[y_attr].sum().reset_index()
                            title = f'{y_attr} by {x_attr}'
                        
                        if result is not None:
                            fig = generate_plot(result, chart_type, x_attr, y_attr, title)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("No data to display for the given query.")
                    except KeyError as e:
                        st.error(f"KeyError: {e}. Available columns: {', '.join(columns)}")
                    except IndexError as e:
                        st.error(f"IndexError: {e}. This might be due to an incorrect number of attributes extracted from the query.")
                else:
                    # Generate descriptive summary if the intent is detected as an overview or performance query
                    if "performance" in user_query or "overview" in user_query or "how is my" in user_query:
                        descriptive_summary = generate_descriptive_summary(data)
                        st.write("Data Summary:")
                        st.write(descriptive_summary)
                    else:
                        st.error("Query not understood. Please try again.")

            st.success('Done!')