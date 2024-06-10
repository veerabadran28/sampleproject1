import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

st.set_page_config(layout="wide")

# Function to predict the type of plot based on user query using GPT-2
def get_plot_type(query):
    inputs = tokenizer.encode("Analyze data: " + query, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.split()[-1].lower().strip()

# Function to handle file upload and return DataFrame
def load_data(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            return pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            return pd.read_json(uploaded_file)
        elif uploaded_file.name.endswith('.parquet'):
            return pd.read_parquet(uploaded_file)
    return None

# Function to generate visualizations dynamically
def generate_visualization(plot_type, df):
    try:
        if "histogram" in plot_type:
            col = st.selectbox("Select a column for histogram", df.columns)
            fig = px.histogram(df, x=col)
        elif "scatter" in plot_type:
            col_x = st.selectbox("Select the X-axis for scatter plot", df.columns)
            col_y = st.selectbox("Select the Y-axis for scatter plot", df.columns)
            fig = px.scatter(df, x=col_x, y=col_y)
        elif "line" in plot_type:
            col_x = st.selectbox("Select the X-axis for line plot", df.columns)
            col_y = st.selectbox("Select the Y-axis for line plot", df.columns)
            fig = px.line(df, x=col_x, y=col_y)
        elif "bar" in plot_type:
            col_x = st.selectbox("Select the X-axis for bar plot", df.columns)
            col_y = st.selectbox("Select the Y-axis for bar plot", df.columns)
            fig = px.bar(df, x=col_x, y=col_y)
        elif "box" in plot_type:
            col = st.selectbox("Select a column for box plot", df.columns)
            fig = px.box(df, y=col)
        elif "correlation" in plot_type:
            fig = px.imshow(df.corr(), text_auto=True)
        elif "pie" in plot_type:
            col_names = st.selectbox("Select the names column for pie chart", df.columns)
            col_values = st.selectbox("Select the values column for pie chart", df.columns)
            fig = px.pie(df, names=col_names, values=col_values)
        elif "area" in plot_type:
            col_x = st.selectbox("Select the X-axis for area chart", df.columns)
            col_y = st.selectbox("Select the Y-axis for area chart", df.columns)
            fig = px.area(df, x=col_x, y=col_y)
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Failed to generate plot: {e}")

# Streamlit main function
def main():
    st.title("Interactive Data Visualization Chatbot")
    uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel, JSON, Parquet)", type=["csv", "xls", "xlsx", "json", "parquet"])
    
    df = load_data(uploaded_file)
    if df is not None:
        st.dataframe(df.head())

        user_query = st.text_input("Ask a question about your data or request a specific graph:")
        if st.button("Generate"):
            plot_type = get_plot_type(user_query)
            st.write(f"Detected plot type: {plot_type}")
            generate_visualization(plot_type, df)

if __name__ == "__main__":
    main()
