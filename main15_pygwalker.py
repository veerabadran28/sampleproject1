import streamlit as st
from pygwalker import walk, PromptTemplate

def main():
    st.title("PyGWalker Example")
    
    # Define the prompt template
    template = PromptTemplate(
        prompt="""
        You are an AI assistant named Claude. You are helpful, kind, honest, and straightforward.
        
        Human: {input}
        Assistant: """,
        input_variables=["input"]
    )

    # Get user input
    user_input = st.text_input("Enter your message:")

    if user_input:
        # Generate a response using PyGWalker
        response = walk(
            template.format(input=user_input), 
            model="anthropic/claude-v1.3-100k", 
            max_tokens=100,
            temperature=0.7
        )
        
        # Display the response
        st.write("Assistant:", response)

if __name__ == "__main__":
    main()