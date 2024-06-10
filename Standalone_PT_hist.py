import streamlit as st
import pandas as pd
import json
import boto3

if 'bedrock_runtime' not in st.session_state:
    st.session_state.bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = [{"role": "user", "content": ""}]

# Function to load data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

# Function to call Claude-3 Sonnet API
def call_claude3_api(df, prompt, max_tokens=2000, max_chars=12000):
    try:
        # Convert the data to a string and include it in the prompt
        data = df.to_string(index=False)
        prompt = f"{prompt}\n\n{data}"

        # Split the prompt into smaller chunks
        prompt_chunks = [prompt[i:i+max_chars] for i in range(0, len(prompt), max_chars)]
        inference_results = []
        for chunk in prompt_chunks:
            st.session_state.conversation_history[-1]["content"] += chunk
            body = {
                "messages": st.session_state.conversation_history,
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens
            }
            response = st.session_state.bedrock_runtime.invoke_model(
                modelId='anthropic.claude-3-sonnet-20240229-v1:0',
                body=json.dumps(body)
            )
            inference_result = json.loads(response['body'].read()).get("content")[0].get("text")
            inference_results.append(inference_result)
            st.session_state.conversation_history.append({"role": "assistant", "content": inference_result})
            st.session_state.conversation_history.append({"role": "user", "content": ""})
        return ' '.join(inference_results)
    except Exception as e:
        st.error(f"Error calling Claude-3 Sonnet API: {e}")
        return None

# Main function
def main():
    st.title("Trend Prediction with AWS Claude Sonnet")

    # File uploader
    file = st.file_uploader("Upload a CSV file", type=["csv"])

    if file is not None:
        # Load data
        df = load_data(file)

        # Preview data
        if st.checkbox("Preview data"):
            st.write(df.head(10))

        # Display conversation history
        for message in st.session_state.conversation_history:
            st.write("")
            st.write(f"{message['role'].capitalize()}: {message['content']}")

        # Text input box
        prompt = st.text_input("Enter a prompt for Claude-3 Sonnet")

        # Predict button
        if st.button("Predict"):
            # Call Claude-3 Sonnet API
            output = call_claude3_api(df, prompt)

            if output:
                st.subheader("Text Generation Output")
                st.write(output)

    else:
        st.warning("Please upload a CSV file.")

if __name__ == "__main__":
    main()