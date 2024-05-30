import streamlit as st
import openai
import pandas as pd
import os

openai.api_key = '<yourkey>'

def get_openai_response(prompt):
    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo", 
            prompt=prompt,
            max_tokens=52  
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return str(e)

st.title("OpenAI with Streamlit")

st.write("")
user_input = st.text_area("Enter your text here:")

uploaded_file = st.file_uploader("Or upload an Excel file:", type=["xlsx", "xls"])

text_data = ""

if uploaded_file is not None:
    
    df = pd.read_excel(uploaded_file)
    
    st.write("### Uploaded Excel File Content:")
    st.write(df)
    
    text_data = df.to_string()

if st.button("Submit"):
    if user_input or text_data:
        with st.spinner("Processing..."):
            if user_input:
                result = get_openai_response(user_input)
            else:
                result = get_openai_response(text_data)
                
            st.write("### Output:")
            st.write(result)
    else:
        st.warning("Please enter some text or upload an Excel file.")
