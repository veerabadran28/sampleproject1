import streamlit as st
import pandas as pd
from streamlit.components.v1 import html
import plotly.express as px
import boto3
import json
from io import BytesIO, StringIO

# Initialize AWS services
if 's3' not in st.session_state:
    st.session_state.s3 = boto3.client('s3')

if 'comprehend' not in st.session_state:
    st.session_state.comprehend = boto3.client('comprehend')

if 'bedrock_runtime' not in st.session_state:
    st.session_state.bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

data = pd.DataFrame()

# Set the Streamlit page to wide mode
st.set_page_config(page_title="Interactive Data Visualization", page_icon=":chart_with_upwards_trend:", layout="wide")

# Function to load data based on file type
def load_data(file):
    if file.name.endswith('.csv'):
        data = pd.read_csv(file, encoding='latin1')
    elif file.name.endswith('.xlsx'):
        xls = pd.ExcelFile(file)
        sheet_name = st.selectbox('Select sheet', xls.sheet_names) if len(xls.sheet_names) > 1 else xls.sheet_names[0]
        data = pd.read_excel(file, sheet_name=sheet_name)
    elif file.name.endswith('.json'):
        data = pd.read_json(file)
    elif file.name.endswith('.parquet'):
        data = pd.read_parquet(file)
    else:
        data = pd.read_csv(file, delimiter=st.text_input('Enter delimiter', value=','), encoding='latin1')
    data.columns = data.columns.str.lower()  # Convert column names to lowercase
    return data

# Function to process user query using Claude-3 Sonnet
def process_query_claude3(query):
    prompt = f"Please inform user that the appropriate chart will be displayed below based on your input: {query}"
    response = call_claude3_api(prompt, max_chars=12000)  # Add max_chars parameter
    comprehend_output = st.session_state.comprehend.detect_entities(Text=response, LanguageCode='en')
    entities = comprehend_output['Entities']
    return response, entities

# Function to call Claude-3 Sonnet API
def call_claude3_api(prompt, max_tokens=2000, max_chars=12000):
    # Split the prompt into smaller chunks
    prompt_chunks = [prompt[i:i+max_chars] for i in range(0, len(prompt), max_chars)]
    
    inference_results = []
    for chunk in prompt_chunks:
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": chunk
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
        inference_results.append(inference_result)
    
    return ' '.join(inference_results)

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
    chart_type = determine_chart_type(query, entities, response)

    # Autocorrect tokens
    tokens = query.split()
    corrected_tokens = list(set([autocorrect(token, columns) for token in tokens]))

    # Extract and match attributes
    combined_attributes = combine_and_match(corrected_tokens, columns)

    # Ensure relevant and unique attributes
    relevant_attributes = []
    for attr in combined_attributes:
        matched_col = next((col for col in columns if col.lower() == attr), None)
        if matched_col and matched_col not in relevant_attributes:
            relevant_attributes.append(matched_col)
    
    # Further autocorrect to find the closest matching columns if needed
    if len(relevant_attributes) < 2:
        for token in tokens:
            possible_corrections = get_close_matches(token, [col.lower() for col in columns], n=1)
            if possible_corrections:
                matched_col = next((col for col in columns if col.lower() == possible_corrections[0]), None)
                if matched_col and matched_col not in relevant_attributes:
                    relevant_attributes.append(matched_col)
    
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
    title = title.title()
    x_label = x.replace('_', ' ').title()  # Capitalize x axis label
    y_label = y.replace('_', ' ').title()  # Capitalize y axis label

    numeric_cols = result.select_dtypes(include=['float64', 'int64']).columns

    if plot_type == 'bar':
        fig = px.bar(result, x=x, y=y, title=title, text=y, color=x)
    elif plot_type == 'line':
        fig = px.line(result, x=x, y=y, title=title, text=y)
    elif plot_type == 'scatter':
        fig = px.scatter(result, x=x, y=y, title=title, text=y, color_discrete_sequence=['#1f77b4'])
    elif plot_type == 'histogram':
        fig = px.histogram(result, x=x, title=title, color=x)
    elif plot_type == 'pie':
        fig = px.pie(result, names=x, values=y, title=title, color_discrete_sequence=px.colors.sequential.RdBu)
    elif plot_type == 'area':
        fig = px.area(result, x=x, y=y, title=title, text=y, color=x)
    elif plot_type == 'box':
        fig = px.box(result, x=x, y=y, title=title, color=x)
    elif plot_type == 'heatmap':
        numeric_data = result[numeric_cols]
        fig = px.imshow(numeric_data.corr().values, title=title, text=y, color_continuous_scale='Viridis')
    elif plot_type == 'violin':
        fig = px.violin(result, x=x, y=y, title=title, color=x)
    elif plot_type == 'map':
        # Add latitude and longitude if the country or city column exists
        if 'country' in [x, y] or 'city' in [x, y]:
            if 'country' in [x, y]:
                result = add_lat_lon_static(result, 'country')
            if 'city' in [x, y]:
                result = add_lat_lon_static(result, 'city')
            if 'latitude' in result.columns and 'longitude' in result.columns:
                fig = px.scatter_mapbox(result, lat='latitude', lon='longitude', hover_name=x, hover_data=[y], title=title, mapbox_style="carto-positron", size=y if result[y].dtype != 'object' else None, zoom=3, height=600, color_discrete_sequence=['#000000'])
                fig.update_layout(mapbox_zoom=3, mapbox_center={"lat": 37.0902, "lon": -95.7129})
            else:
                st.error("Failed to add latitude and longitude for the country or city names.")
                fig = None
        else:
            st.error("Map chart requires 'country' or 'city' as one of the attributes.")
            fig = None

    fig.update_layout(title_font_size=24, title_font_family="Arial", title_font_color="#333333", 
                    xaxis_title_font_size=18, xaxis_title_font_family="Arial", xaxis_title_font_color="#666666", xaxis_title_text=x_label,
                    yaxis_title_font_size=18, yaxis_title_font_family="Arial", yaxis_title_font_color="#666666", yaxis_title_text=y_label,
                    legend_title_font_size=16, legend_title_font_family="Arial", legend_title_font_color="#333333")
    return fig

# Function to detect programming language using AWS Comprehend
def detect_language_comprehend(text):
    try:
        response = st.session_state.comprehend.detect_dominant_language(Text=text)
        language_code = response.get('Languages', [{}])[0].get('LanguageCode', 'en')

        if language_code == 'py':
            return 'python'
        elif language_code == 'js':
            return 'javascript'
        elif language_code == 'java':
            return 'java'
        elif language_code == 'cpp':
            return 'cpp'
        else:
            return 'python'
    except Exception as e:
        st.error(f"Error detecting language: {e}")
        return 'unknown'
    
# Function to generate code based on language and task
def generate_code(task):
    prompt = f"{task}"
    generated_code = call_claude3_api(prompt, max_tokens=1000)

    # Automatically identify the programming language using AWS Comprehend
    language = detect_language_comprehend(generated_code)

    return generated_code, language

def download_file(content, filename, file_ext):
    binary = content.encode()
    b64 = base64.b64encode(binary).decode()
    href = f'<a href="data:file/{file_ext};base64,{b64}" download="{filename}">Download File</a>'
    return href

def chatbot_component():
    chatbot_html = """
    <div id="chatbot">
        <div id="conversation"></div>
        <input type="text" id="user_input" placeholder="Type your message here...">
    </div>
    <script>
        const conversation = document.getElementById('conversation');
        const userInput = document.getElementById('user_input');

        userInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                const userMessage = userInput.value;
                if (userMessage.trim() !== '') {
                    addMessage('user', userMessage);
                    userInput.value = '';
                    processMessage(userMessage);
                }
            }
        });

        function processMessage(message) {
            console.log('Sending message:', message);  // Log the message being sent
            const payload = {
                "message": message
            };

            const request = new XMLHttpRequest();
            request.open('POST', '', true);
            request.setRequestHeader('Content-Type', 'application/json');
            request.onload = function() {
                console.log('Response status:', request.status);  // Log the response status
                if (request.status === 200) {
                    const response = JSON.parse(request.responseText);
                    console.log('Response:', response);  // Log the response object
                    addMessage('bot', response.reply);
                } else {
                    console.log('Error:', request.responseText);  // Log the error message
                    addMessage('bot', 'Something went wrong. Please try again later.');
                }
            };
            request.onerror = function() {
                console.log('Error:', request.statusText);  // Log the error message
                addMessage('bot', 'Something went wrong. Please try again later.');
            };
            request.send(JSON.stringify(payload));
        }

        function addMessage(sender, message) {
            const messageElement = document.createElement('div');
            messageElement.classList.add(sender);
            messageElement.innerText = message;
            conversation.appendChild(messageElement);
            conversation.scrollTop = conversation.scrollHeight;
        }
    </script>
    <style>
        #chatbot {
            height: 500px;
            overflow-y: scroll;
            border: 1px solid #ccc;
            padding: 10px;
        }

        .user {
            text-align: right;
            color: blue;
        }

        .bot {
            text-align: left;
            color: green;
        }
    </style>
    """
    html(chatbot_html)

# Streamlit application layout
header_container = st.container()
data_container = st.container()
chart_container = st.container()
input_container = st.container()
code_generator_container = st.container()
uploaded_file = None

# Sidebar
user_selection_option = st.sidebar.radio("Choose Option:", ["Explore My Data!", "Explore data from AWS S3!", "Code Generator"])

with header_container:
    if user_selection_option in ["Explore My Data!","Explore data from AWS S3!"]:
        st.markdown("<h2 style='text-align: center; padding: 10px 0;'>Interactive Data Visualization</h2>", unsafe_allow_html=True)
    elif user_selection_option == "Code Generator":
        st.markdown("<h2 style='text-align: center; padding: 10px 0;'>Code Generator</h2>", unsafe_allow_html=True)

with data_container:
    if user_selection_option != "Code Generator":
        if user_selection_option == "Explore data from AWS S3!":
            # List files from S3 bucket
            bucket_name = 'avengers-ba007'
            file_list = st.session_state.s3.list_objects_v2(Bucket=bucket_name)['Contents']
            file_names = [obj['Key'] for obj in file_list]    

            # Display file names in Streamlit dropdown
            selected_file = st.sidebar.selectbox('Select a file', [''] + file_names)

            # Load and process selected file
            if selected_file:
                file_obj = st.session_state.s3.get_object(Bucket=bucket_name, Key=selected_file)
                file_content = file_obj['Body'].read()

                # Process the file based on its format (e.g., CSV, Excel, JSON, Parquet)
                if selected_file.endswith('.csv'):
                    data = pd.read_csv(BytesIO(file_content), encoding='latin1')
                elif selected_file.endswith('.xlsx'):
                    data = pd.read_excel(BytesIO(file_content))
                elif selected_file.endswith('.json'):
                    data = pd.read_json(BytesIO(file_content), orient='records')
                elif selected_file.endswith('.parquet'):
                    data = pd.read_parquet(BytesIO(file_content))
                else:
                    # Handle other file formats or unsupported formats
                    data = None
                columns = data.columns.tolist() if not data.empty else []  # Initialize columns as an empty list if data is empty
            else:
                st.write("Select a data point from AWS S3..")

        if user_selection_option == "Explore My Data!":
            st.write("Upload your data file..")

            uploaded_file = st.sidebar.file_uploader("Choose a file", type=['csv', 'xlsx', 'json', 'parquet', 'txt'])

            if uploaded_file is not None:    
                data = load_data(uploaded_file)
                columns = data.columns.tolist() if not data.empty else []  # Initialize columns as an empty list if data is empty

        with st.expander("Data Insights"):
            data_preview_tab, data_explanation_tab, data_summary_tab, data_statistics_tab = st.tabs(["Data Preview", "Data Explanation", "Overall Data Summary", "Statistics"])

            with data_preview_tab:
                if data.empty:
                    st.warning("No data available for preview.")
                else:
                    st.dataframe(data)

            with data_explanation_tab:
                if not data.empty:
                    st.write(call_claude3_api(f"Provide a summary of the following data: {data.head().to_dict(orient='records')}", max_chars=12000))

            with data_summary_tab:
                if not data.empty:
                    display_summary(generate_summary(data))

            with data_statistics_tab:
                if not data.empty:
                    st.write(generate_descriptive_summary(data))

with input_container:
    if user_selection_option != "Code Generator":
        input_container_row = st.container()
        chatbot_component()

with chart_container:
    if st.session_state.get("user_query") or st.session_state.get("submit_button"):
        if not data.empty:
            st.markdown("<h3 style='padding: 10px 0;'>Chart Visualization</h3>", unsafe_allow_html=True)
            with st.spinner('Processing...'):
                chart_type, x_attr, y_attr = process_query(st.session_state.user_query, columns)
                
                if chart_type == "summary" or chart_type == "statistical" or chart_type == "statistics":
                    display_summary(generate_summary(data))
                elif x_attr and y_attr:
                    try:
                        if "top" in st.session_state.user_query:
                            number = int(''.join(filter(str.isdigit, st.session_state.user_query)))
                            result = data.groupby(y_attr)[x_attr].sum().reset_index().sort_values(by=x_attr, ascending=False).head(number)
                            title = f'Top {x_attr} by {y_attr}'
                        elif "bottom" in st.session_state.user_query:
                            number = int(''.join(filter(str.isdigit, st.session_state.user_query)))
                            result = data.groupby(y_attr)[x_attr].sum().reset_index().sort_values(by=x_attr, ascending=True).head(number)
                            title = f'Bottom {x_attr} by {y_attr}'
                        elif "quantity" in st.session_state.user_query or "sales" in st.session_state.user_query:
                            result = data.groupby(x_attr)[y_attr].sum().reset_index()
                            title = f'{y_attr} vs {x_attr}'
                        elif "date" in st.session_state.user_query or "time" in st.session_state.user_query:
                            result = data.groupby(x_attr)[y_attr].sum().reset_index()
                            title = f'{y_attr} Over Time'
                        elif "map" in st.session_state.user_query:
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
                    if "performance" in st.session_state.user_query or "overview" in st.session_state.user_query or "how is my" in st.session_state.user_query:
                        descriptive_summary = generate_descriptive_summary(data)
                        st.write("Data Summary:")
                        st.write(descriptive_summary)
                    else:
                        st.error("Query not understood. Please try again.")
        else:
            st.warning("No data available for process.. Please select/upload a file")

# Initialize session state for generated code
if 'generated_code' not in st.session_state:
    st.session_state.generated_code = None
if 'language' not in st.session_state:
    st.session_state.language = None

with code_generator_container:
    if user_selection_option == "Code Generator":
        task_description = st.text_area("Enter the task or problem statement", height=20)
        if st.button("Generate Code", key="gen_code_button"):
            if task_description != "":
                with st.spinner("Generating code..."):
                    generated_code, language = generate_code(task_description)
                    st.session_state.generated_code = generated_code
                    st.session_state.language = language
            else:
                st.warning("Enter the task or problem statement and try again...")

        if st.session_state.generated_code:
            file_ext = st.session_state.language
            if file_ext == "python":
                file_ext = "py"
            elif file_ext == "javascript":
                file_ext = "js"
            elif file_ext == "java":
                file_ext = "java"
            elif file_ext == "c++":
                file_ext = "cpp"
            filename = f"generated_code.{file_ext}"
            st.markdown(download_file(st.session_state.generated_code, filename, file_ext), unsafe_allow_html=True)
            st.markdown(f'{st.session_state.generated_code}', unsafe_allow_html=True)

# Declare the global data variable
data = pd.DataFrame()

def process_message(message):
    global data  # Declare data as global
    if message.startswith('load'):
        data_source = message.split()[1]
        if data_source == 'file':
            # Handle file upload
            uploaded_file = st.file_uploader("Upload a file", type=['csv', 'xlsx', 'json', 'parquet'])
            if uploaded_file is not None:
                data = load_data(uploaded_file)
                return "Data loaded successfully!"
            else:
                return "No file uploaded."
        elif data_source == 'aws':
            # Handle loading data from AWS S3
            bucket_name = st.text_input("Enter S3 bucket name")
            if bucket_name:
                data = load_data_from_s3(bucket_name)
                return "Data loaded successfully!"
            else:
                return "Please enter a valid S3 bucket name."
        else:
            return "Invalid command. Please try again."
    else:
        # Process the message as a query
        return process_query(message, data.columns)

def load_data_from_s3(bucket_name):
    global data  # Declare data as global
    file_obj = st.session_state.s3.get_object(Bucket=bucket_name, Key='data.csv')
    file_content = file_obj['Body'].read()
    data = pd.read_csv(BytesIO(file_content), encoding='latin1')
    return data

#st.set_option('server.port', 8501)  # Set the desired port number
#st.markdown(f"Server is running on port 8501")

@st.cache_data
def get_streamlit_session_state():
    return {}

session_state = get_streamlit_session_state()

if session_state:
    st.markdown(f"Session state: {session_state}")

st.stop()