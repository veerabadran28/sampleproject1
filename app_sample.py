import streamlit as st
import requests
import os
from urllib.parse import urljoin
from streamlit_javascript import st_javascript
import time
import prompt_lib as glib

# Get the Flask app's authentication endpoint URL from the environment variable
AUTH_ENDPOINT = os.environ.get("REACT_APP_USERS_SERVICE_URL", "") + "/auth/status"
print(f"AUTH_ENDPOINT: {AUTH_ENDPOINT}")

def get_token_from_local_storage(k):

    v = st_javascript(
        f"localStorage.getItem('{k}');"
    )
    token = v
    print(f"Token: {token}")
    return token

def is_authenticated():
    """
    Check if the user is authenticated by sending a request to the authentication endpoint.
    Returns True if the user is authenticated, False otherwise.
    """
    token = get_token_from_local_storage("authToken")
    # s = requests.get("/users/ping")
    # ping_url = os.environ.get("REACT_APP_USERS_SERVICE_URL", "") + "/users/ping"
    # ping_url = f"http://users:5000/users/ping"
    # s = requests.get(ping_url)
    # print(f"Ping: {s.json()}")

    try:        
        headers = {"Authorization": f"Bearer {token}"}
        print(f"headers: {headers}")
        print(f"AUTH_ENDPOINT: {AUTH_ENDPOINT}")
        #response = requests.get(AUTH_ENDPOINT, headers=headers)
        response = requests.get(f"http://users:5000/auth/status", headers=headers)
        # response = requests.get(f"http://localhost/auth/status", headers=headers)
        # time.sleep(3)
        print(f"Response: {response}")
        response.raise_for_status()
        data = response.json()
        if data["status"] == "success":
            return True
    except requests.exceptions.RequestException:
        pass
    return False

model_options_dict = {
    "anthropic.claude-3-sonnet-20240229-v1:0": "Claude",
    "cohere.command-text-v14": "Command",
    "ai21.j2-ultra-v1": "Jurassic",
    "meta.llama2-70b-chat-v1": "Llama",
    "amazon.titan-text-express-v1": "Titan",
    "mistral.mixtral-8x7b-instruct-v0:1": "Mixtral"
}

model_options = list(model_options_dict)

def get_model_label(model_id):
    print(model_id)
    return model_options_dict[model_id]

def main():
    # Set the page title
    st.set_page_config(page_title="ESG Data Assistant 2", layout="wide")
    st.header("ESG Data Assistant 2")
    col1, col2, col3 = st.columns(3)

    # Check if the user is authenticated
    if is_authenticated():
        # Add a header
        

        with col1:
            st.subheader("Context")
            context_list = glib.get_context_list()
            selected_context = st.radio(
                "Lab context:",
                context_list,
            )
            with st.expander("See context"):
                context_for_lab = glib.get_context(selected_context)
                context_text = st.text_area("Context text:", value=context_for_lab, height=350)
        with col2:
            st.subheader("Prompt & model")
            prompt_text = st.text_area("Prompt template text:", height=350)
            selected_model = st.radio("Model:", 
                model_options,
                format_func=get_model_label,
                horizontal=True
            )
            process_button = st.button("Run", type="primary")
        with col3:
            st.subheader("Result")
            if process_button:
                response_content = glib.get_text_response(model_id=selected_model, temperature=0.0, template=prompt_text, context=context_text)
                st.write(response_content)



        # ... (rest of your app code)
    else:
        # Redirect the user to the login page
        login_url = urljoin(os.environ.get("REACT_APP_USERS_SERVICE_URL", ""), "/login")
        print(login_url)
        st.markdown(f"You are not authenticated. Please [log in]({login_url}) to access the app.")

if __name__ == "__main__":
    main()

