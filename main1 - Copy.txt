import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

# Set page layout to wide
st.set_page_config(layout="wide")

st.title("Interactive Chatbot for Data Analysis")

st.write("""
This chatbot allows you to interact with it using natural language. You can ask for data summaries, visualizations, and forecasts.
""")

# Load a pre-trained model for intent classification and context understanding
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define the possible intents
intents = [
    "show statistics",
    "generate histogram",
    "generate scatter plot",
    "generate correlation matrix",
    "generate line plot",
    "generate bar plot",
    "generate box plot",
    "generate pair plot",
    "generate pie chart",
    "generate area chart",
    "generate heatmap",
    "generate treemap",
    "generate sunburst",
    "project sales"
]

# Function to classify the user query
def classify_intent(query):
    result = classifier(query, intents)
    return result['labels'][0]

# Function to generate basic statistics
def generate_statistics(df):
    st.write("### Basic Statistics")
    st.write(df.describe())

# Function to generate a histogram
def generate_histogram(df, column):
    st.write(f"### Histogram of {column}")
    fig = px.histogram(df, x=column, marginal="box", nbins=50, title=f'Histogram of {column}')
    st.plotly_chart(fig)

# Function to generate a scatter plot
def generate_scatter_plot(df, col1, col2):
    st.write(f"### Scatter Plot between {col1} and {col2}")
    fig = px.scatter(df, x=col1, y=col2, title=f'Scatter Plot between {col1} and {col2}')
    st.plotly_chart(fig)

# Function to generate a correlation matrix heatmap
def generate_correlation_heatmap(df):
    st.write("### Correlation Matrix Heatmap")
    corr = df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix Heatmap")
    st.plotly_chart(fig)

# Function to generate a line plot
def generate_line_plot(df, col1, col2):
    st.write(f"### Line Plot between {col1} and {col2}")
    fig = px.line(df, x=col1, y=col2, title=f'Line Plot between {col1} and {col2}')
    st.plotly_chart(fig)

# Function to generate a bar plot
def generate_bar_plot(df, col1, col2):
    st.write(f"### Bar Plot between {col1} and {col2}")
    fig = px.bar(df, x=col1, y=col2, title=f'Bar Plot between {col1} and {col2}')
    st.plotly_chart(fig)

# Function to generate a box plot
def generate_box_plot(df, column):
    st.write(f"### Box Plot of {column}")
    fig = px.box(df, y=column, title=f'Box Plot of {column}')
    st.plotly_chart(fig)

# Function to generate a pair plot
def generate_pair_plot(df):
    st.write("### Pair Plot")
    fig = px.scatter_matrix(df)
    st.plotly_chart(fig)

# Function to generate a pie chart
def generate_pie_chart(df, names, values):
    st.write(f"### Pie Chart of {names}")
    fig = px.pie(df, names=names, values=values, title=f'Pie Chart of {names}')
    st.plotly_chart(fig)

# Function to generate an area chart
def generate_area_chart(df, x, y):
    st.write(f"### Area Chart of {y} over {x}")
    fig = px.area(df, x=x, y=y, title=f'Area Chart of {y} over {x}')
    st.plotly_chart(fig)

# Function to generate a heatmap
def generate_heatmap(df, x, y, z):
    st.write(f"### Heatmap of {z}")
    fig = px.density_heatmap(df, x=x, y=y, z=z, title=f'Heatmap of {z}')
    st.plotly_chart(fig)

# Function to generate a treemap
def generate_treemap(df, path, values):
    st.write(f"### Treemap of {path} by {values}")
    fig = px.treemap(df, path=path, values=values, title=f'Treemap of {path} by {values}')
    st.plotly_chart(fig)

# Function to generate a sunburst chart
def generate_sunburst(df, path, values):
    st.write(f"### Sunburst Chart of {path} by {values}")
    fig = px.sunburst(df, path=path, values=values, title=f'Sunburst Chart of {path} by {values}')
    st.plotly_chart(fig)

# Function to generate sales projections (simple example)
def generate_sales_projection(df, date_col, value_col, periods=3):
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df = df.resample('M').sum()
    last_date = df.index.max()
    projection_dates = [last_date + pd.DateOffset(months=i) for i in range(1, periods+1)]
    projected_values = [df[value_col].mean()] * periods

    projection_df = pd.DataFrame({date_col: projection_dates, value_col: projected_values})
    df = df.reset_index()
    projection_df = projection_df.reset_index(drop=True)
    
    combined_df = pd.concat([df, projection_df])
    st.write("### Sales Projection")
    fig = px.line(combined_df, x=date_col, y=value_col, title=f'Sales Projection for Next {periods} Months')
    st.plotly_chart(fig)

# Function to read CSV with specified encodings
def read_csv_with_encoding(file, encodings=['utf-8', 'latin1', 'ISO-8859-1']):
    for encoding in encodings:
        try:
            return pd.read_csv(file, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("Unable to read the file with the specified encodings.")

# Function to read Excel files
def read_excel(file):
    return pd.read_excel(file, sheet_name=None)  # Read all sheets

# Function to read JSON files
def read_json(file):
    return pd.read_json(file)

# Function to read Parquet files
def read_parquet(file):
    return pd.read_parquet(file)

# Function to read Text files
def read_text(file):
    return pd.read_csv(file, delimiter='\t')

# Main function to read and process user input
def main():
    # Upload file
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xls", "xlsx", "json", "parquet", "txt"])

    if uploaded_file:
        # Determine the type of file and read it accordingly
        file_type = uploaded_file.name.split('.')[-1]
        if file_type == 'csv':
            df = read_csv(uploaded_file)
        elif file_type == 'txt':
            df = read_text(uploaded_file)
        elif file_type in ['xls', 'xlsx']:
            df_dict = read_excel(uploaded_file)
            sheet_name = st.selectbox("Select sheet", df_dict.keys())
            df = df_dict[sheet_name]
        elif file_type == 'json':
            df = read_json(uploaded_file)
        elif file_type == 'parquet':
            df = read_parquet(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()

        # Display the DataFrame with column headers
        st.write("### Data Preview")
        st.dataframe(df, use_container_width=True)

        # Generate basic statistics
        generate_statistics(df)

        # User input for interacting with the chatbot
        st.write("## Chat with the Chatbot")
        user_input = st.text_input("You: ", "Please give me top 10 sales for the period starting from Jan till now and also give me the projection for next quarter.")

        if st.button("Send"):
            intent = classify_intent(user_input)
            st.write(f"Chatbot identified intent: {intent}")

            # Process user input based on identified intent
            if intent == "show statistics":
                generate_statistics(df)
            elif intent == "generate histogram":
                column_hist = user_input.split()[-1]  # Example parsing, adjust as needed
                generate_histogram(df, column_hist)
            elif intent == "generate scatter plot":
                col1_scatter = user_input.split()[3]  # Example parsing, adjust as needed
                col2_scatter = user_input.split()[6]  # Example parsing, adjust as needed
                generate_scatter_plot(df, col1_scatter, col2_scatter)
            elif intent == "generate correlation matrix":
                generate_correlation_heatmap(df)
            elif intent == "generate line plot":
                col1_line = user_input.split()[3]  # Example parsing, adjust as needed
                col2_line = user_input.split()[6]  # Example parsing, adjust as needed
                generate_line_plot(df, col1_line, col2_line)
            elif intent == "generate bar plot":
                col1_bar = user_input.split()[3]  # Example parsing, adjust as needed
                col2_bar = user_input.split()[6]  # Example parsing, adjust as needed
                generate_bar_plot(df, col1_bar, col2_bar)
            elif intent == "generate box plot":
                column_box = user_input.split()[-1]  # Example parsing, adjust as needed
                generate_box_plot(df, column_box)
            elif intent == "generate pair plot":
                generate_pair_plot(df)
            elif intent == "generate pie chart":
                col1_pie = user_input.split()[3]  # Example parsing, adjust as needed
                col2_pie = user_input.split()[6]  # Example parsing, adjust as needed
                generate_pie_chart(df, col1_pie, col2_pie)
            elif intent == "generate area chart":
                col1_area = user_input.split()[3]  # Example parsing, adjust as needed
                col2_area = user_input.split()[6]  # Example parsing, adjust as needed
                generate_area_chart(df, col1_area, col2_area)
            elif intent == "generate heatmap":
                col1_heat = user_input.split()[3]  # Example parsing, adjust as needed
                col2_heat = user_input.split()[6]  # Example parsing, adjust as needed
                col3_heat = user_input.split()[9]  # Example parsing, adjust as needed
                generate_heatmap(df, col1_heat, col2_heat, col3_heat)
            elif intent == "generate treemap":
                col1_tree = user_input.split()[3]  # Example parsing, adjust as needed
                col2_tree = user_input.split()[6]  # Example parsing, adjust as needed
                generate_treemap(df, [col1_tree], col2_tree)
            elif intent == "generate sunburst":
                col1_sun = user_input.split()[3]  # Example parsing, adjust as needed
                col2_sun = user_input.split()[6]  # Example parsing, adjust as needed
                generate_sunburst(df, [col1_sun], col2_sun)
            elif intent == "project sales":
                date_col = "date"  # Adjust column names based on your data
                value_col = "sales"
                generate_sales_projection(df, date_col, value_col)

# Call the main function
if __name__ == "__main__":
    main()
