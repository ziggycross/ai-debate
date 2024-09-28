from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import io
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer


st.title("AI Debate Analyser")

with st.sidebar:
    st.title("Settings")
    
    kaggle_dataset = st.checkbox("Use Kaggle Dataset", value=True)
    custom_dataset = st.file_uploader("Custom Data", disabled=kaggle_dataset, type="csv")

@st.cache_data
def get_kaggle_dataset(user: str, proj: str, file: str) -> pd.DataFrame:
    api = kaggle.KaggleApi()
    res = api.datasets_download_file(user, proj, file)
    df = pd.read_csv(io.StringIO(res))
    return df

if kaggle_dataset:
    df = get_kaggle_dataset("ziggycross", "ai-debate-samples", "aidebates.csv")
else:
    df = pd.read_csv(custom_dataset)

with st.expander("Data source"):
    df

low_cut = st.sidebar.slider("Include responses after round ...", value=5, min_value=1, max_value=max(df["round"]))

vectorizer_model = CountVectorizer(stop_words="english")

@st.cache_data
def topic_modelling(docs: pd.Series):
    topic_model = BERTopic(vectorizer_model=vectorizer_model)
    topics, probs = topic_model.fit_transform(docs)
    return topic_model, topics, probs

with st.spinner("Performing topic modelling"):
    pos_docs = df["affirmative_response"].loc[df["round"]>=low_cut]
    neg_docs = df["negative_response"].loc[df["round"]>=low_cut]
    pos_model, pos_topics, pos_probs = topic_modelling(pos_docs)
    neg_model, neg_topics, neg_probs = topic_modelling(neg_docs)

tab_pos, tab_neg = st.tabs(["Positive", "Negative"])

with tab_pos:
    st.dataframe({"docs": pos_docs, "topic": pos_topics, "prob": pos_probs})
    st.plotly_chart(pos_model.visualize_topics())
    st.plotly_chart(pos_model.visualize_barchart())
    st.plotly_chart(pos_model.visualize_heatmap())

with tab_neg:
    st.dataframe({"docs": neg_docs, "topic": neg_topics, "prob": neg_probs})
    st.plotly_chart(neg_model.visualize_topics())
    st.plotly_chart(neg_model.visualize_barchart())
    st.plotly_chart(neg_model.visualize_heatmap())