
import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000"

st.title("Academic Research Assistant")

# Search Section
st.header("Search Research Papers")
topic = st.text_input("Enter research topic:")
if st.button("Search"):
    response = requests.post(f"{API_URL}/search", json={"topic": topic})
    if response.status_code == 200:
        papers = response.json()
        st.session_state['papers'] = papers
        
        
        df = pd.DataFrame(papers)
        st.dataframe(df[['title', 'published', 'authors']])
    else:
        st.error("Error: Unable to fetch papers.")


st.header("Ask Questions")
if 'papers' in st.session_state:
    selected_paper = st.selectbox(
        "Select a paper:",
        options=st.session_state['papers'],
        format_func=lambda x: x['title']
    )
    
    if selected_paper:
        st.write("Paper Summary:")
        st.write(selected_paper['summary'])
        
        question = st.text_input("Enter your question about this paper:")
        if st.button("Get Answer"):
            response = requests.post(
                f"{API_URL}/qa",
                json={
                    "question": question,
                    "context": selected_paper['summary']
                }
            )
            if response.status_code == 200:
                st.write("Answer:", response.json()['answer'])
            else:
                st.error("Error: Unable to get an answer.")


st.header("Generate Future Research Directions")
if st.button("Generate Future Directions"):
    response = requests.post(f"{API_URL}/future-directions", json={"topic": topic})
    if response.status_code == 200:
        st.write(response.json()['directions'])
    else:
        st.error("Error: Unable to generate future directions.")