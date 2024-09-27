import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Debate", page_icon="🎭", layout="wide")
st.title("AI Debate")

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")

@st.cache_data
def summarize_text(api_key, text):
    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, api_key=api_key)
    summary_prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in 50 words or less:\n\n{text}"
    )
    summary_chain = LLMChain(llm=model, prompt=summary_prompt)
    return summary_chain.run(text)

@st.cache_data
def generate_response(api_key, system_message, human_message):
    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, api_key=api_key)
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=human_message)
    ]
    return model(messages).content

@st.cache_data
def run_debate(api_key, debate_topic, affirmative_system, negative_system, num_iterations):
    debate_data = []

    affirmative_context = f"The debate topic is: {debate_topic}. Provide an opening argument in favor."
    negative_context = f"The debate topic is: {debate_topic}. Provide an opening argument against."

    for i in range(num_iterations):
        affirmative_response = generate_response(api_key, affirmative_system, affirmative_context)
        negative_response = generate_response(api_key, negative_system, negative_context)

        debate_data.append({
            "round": i+1,
            "topic": debate_topic.replace("\n", " "),
            "affirmative_system": affirmative_system.replace("\n", " "),
            "affirmative_context": affirmative_context.replace("\n", " "),
            "negative_system": negative_system.replace("\n", " "),
            "negative_context": negative_context.replace("\n", " "),
            "affirmative_response": affirmative_response.replace("\n", " "),
            "negative_response": negative_response.replace("\n", " ")
        })

        affirmative_summary = summarize_text(api_key, affirmative_response)
        negative_summary = summarize_text(api_key, negative_response)

        affirmative_context = f"Your previous argument: {affirmative_summary}\nOpponent's counterargument: {negative_summary}\nProvide a concise rebuttal."
        negative_context = f"Your previous argument: {negative_summary}\nOpponent's counterargument: {affirmative_summary}\nProvide a concise rebuttal."

    return debate_data

def main():
    st.sidebar.title("Settings")
    api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password", value=os.environ.get("OPENAI_API_KEY"))

    if not api_key:
        st.warning("Please enter your OpenAI API key to proceed.")
        return

    st.sidebar.subheader("Prompts")
    affirmative_system = st.sidebar.text_area("Affirmative System Prompt:", "You are debating in favor of the given topic. Provide concise arguments.")
    negative_system = st.sidebar.text_area("Negative System Prompt:", "You are debating against the given topic. Provide concise counterarguments.")

    num_iterations = st.sidebar.slider("Number of debate iterations:", 1, 20, 10)

    debate_topic = st.text_input("Enter the debate topic:")

    if debate_topic:
        st.subheader(f"Debate Topic: {debate_topic}")

        if st.button("Generate Debate"):
            with st.spinner("Generating debate..."):
                debate_data = run_debate(api_key, debate_topic, affirmative_system, negative_system, num_iterations)

            for entry in debate_data:  # Skip the info entries
                st.write(f"Round {entry['round']}")
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Affirmative")
                    st.write(entry['affirmative_response'])

                with col2:
                    st.subheader("Negative")
                    st.write(entry['negative_response'])

                st.write("---")

            st.success("Debate completed!")

            # Create DataFrame and download button
            df = pd.DataFrame(debate_data)
            csv = convert_df(df)
            filename = f"debate_export_{num_iterations}_turns.csv"
            
            st.download_button(
                label="Download debate as CSV",
                data=csv,
                file_name=filename,
                mime="text/csv",
            )

if __name__ == "__main__":
    main()