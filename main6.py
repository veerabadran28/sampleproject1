import streamlit as st
import pandas as pd
import plotly.express as px
import spacy
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from difflib import get_close_matches
from Levenshtein import distance as levenshtein_distance
import torch

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load DistilBERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

#tokenizer.save_pretrained('distilbert-base-uncased')
#model.save_pretrained('distilbert-base-uncased')

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

# Function to autocorrect words using Levenshtein distance
def autocorrect(word, possibilities):
    word = word.lower()
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

# Function to process user query using DistilBERT
def process_query_distilbert(query):
    inputs = tokenizer(query, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    return predicted_class_id

# Refined function to determine chart type based on query and predictions
def determine_chart_type(query, predicted_class_id):
    query_lower = query.lower()
    
    # Explicit keyword priority
    if "bar" in query_lower:
        return "bar"
    elif "line" in query_lower:
        return "line"
    elif "scatter" in query_lower:
        return "scatter"
    elif "histogram" in query_lower:
        return "histogram"
    elif "pie" in query_lower:
        return "pie"
    elif "area" in query_lower:
        return "area"
    elif "box" in query_lower:
        return "box"
    elif "heatmap" in query_lower:
        return "heatmap"
    elif "violin" in query_lower:
        return "violin"
    elif "map" in query_lower:
        return "map"
    
    # Use DistilBERT prediction as fallback
    chart_types = ["bar", "line", "scatter", "histogram", "pie", "box", "heatmap", "violin", "map"]  # Extend this list based on your classification
    return chart_types[predicted_class_id] if predicted_class_id < len(chart_types) else "bar"

# Process user query with spaCy NLP and DistilBERT
def process_query(query, columns):
    doc = nlp(query.lower())
    
    # Determine chart type using DistilBERT
    predicted_class_id = process_query_distilbert(query)
    st.write(predicted_class_id)
    chart_type = determine_chart_type(query, predicted_class_id)
    st.write("Chart Type: ", chart_type)
    
    # Extract and autocorrect attributes
    tokens = [token.text for token in doc]
    st.write('tokens:', tokens)
    st.write('columns:', columns)
    combined_attributes = combine_and_match(tokens, columns)
    st.write('combined_attributes:', combined_attributes)
    
    # Ensure relevant and unique attributes
    relevant_attributes = []
    for attr in combined_attributes:
        if attr in columns and attr not in relevant_attributes:
            relevant_attributes.append(attr)
    
    # Handle individual tokens if not enough relevant attributes found
    if len(relevant_attributes) < 2:
        for token in tokens:
            corrected_attr = autocorrect(token, columns)
            if corrected_attr in columns and corrected_attr not in relevant_attributes:
                relevant_attributes.append(corrected_attr)
            if len(relevant_attributes) >= 2:
                break

    # Further autocorrect to find the closest matching columns if needed
    if len(relevant_attributes) < 2:
        for token in tokens:
            possible_corrections = get_close_matches(token, columns, n=1)
            if possible_corrections:
                relevant_attributes.append(possible_corrections[0])
    
    return chart_type, relevant_attributes

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
    # Ensure string column is on the x-axis and numeric column is on the y-axis
    if result[x].dtype != 'object' and result[y].dtype == 'object':
        x, y = y, x

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

# Streamlit application
st.title("Interactive Data Visualization")
st.write("Upload your data file and enter your query below:")

with st.expander("Upload Data File"):
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'json', 'parquet', 'txt'])

if uploaded_file is not None:
    data = load_data(uploaded_file)
    columns = data.columns.tolist()
    st.write("Data loaded successfully!")
    
    with st.expander("Data Preview"):
        st.write("Here's a preview of your data:")
        st.dataframe(data.head())  # Show a preview of the data
    
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
                chart_type, attributes = process_query(user_query, columns)

                if chart_type == "summary" or chart_type == "statistical" or chart_type == "statistics":
                    display_summary(summary)
                elif attributes:
                    # Log the extracted attributes for debugging
                    st.write(f"Extracted attributes: {attributes}")
                    if len(attributes) < 2:
                        st.error("Not enough attributes found for the query.")
                    else:
                        try:
                            # Ensure the attributes are correctly autocorrected to the exact column names
                            corrected_attributes = [autocorrect(attr, columns) for attr in attributes]
                            st.write(f"Corrected attributes: {corrected_attributes}")  # Log corrected attributes for debugging
                            if "top" in user_query:
                                number = int(''.join(filter(str.isdigit, user_query)))
                                result = data.groupby(corrected_attributes[1])[corrected_attributes[0]].sum().reset_index().sort_values(by=corrected_attributes[0], ascending=False).head(number)
                                title = f'Top {corrected_attributes[0]} by {corrected_attributes[1]}'
                            elif "bottom" in user_query:
                                number = int(''.join(filter(str.isdigit, user_query)))
                                result = data.groupby(corrected_attributes[1])[corrected_attributes[0]].sum().reset_index().sort_values(by=corrected_attributes[0], ascending=True).head(number)
                                title = f'Top {corrected_attributes[0]} by {corrected_attributes[1]}'
                            elif "quantity" in user_query or "sales" in user_query:
                                result = data.groupby(corrected_attributes[0])[corrected_attributes[1]].sum().reset_index()
                                title = f'{corrected_attributes[1]} vs {corrected_attributes[0]}'
                            elif "date" in user_query or "time" in user_query:
                                result = data.groupby(corrected_attributes[0])[corrected_attributes[1]].sum().reset_index()
                                title = f'{corrected_attributes[1]} Over Time'
                            elif "map" in user_query:
                                result = data
                                title = f'Map of {corrected_attributes[0]}'
                            else:
                                result = data.groupby(corrected_attributes[0])[corrected_attributes[1]].sum().reset_index()
                                title = f'{corrected_attributes[1]} by {corrected_attributes[0]}'
                            
                            if result is not None:
                                fig = generate_plot(result, chart_type, corrected_attributes[0], corrected_attributes[1], title)
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
