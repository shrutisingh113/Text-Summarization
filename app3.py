import streamlit as st
import scipy as sp
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

def read_article(sample_text):
    article = sample_text.split(". ")
    sentences = []
    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop()

    return sentences


def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1

    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)


def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:  # ignore if both are same sentences
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(
                sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(sample_text, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences = read_article(sample_text)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarize texr
    return ''+'.'.join(summarize_text)

# Display list of experiments
experiments = [
    "Experiment 1: Case Study on Social Media Analytics",
    "Experiment 2: To Explore Social Media Analytics Tools",
    "Experiment 3: Web Scraping",
    "Experiment 4: Sentiment Analysis",
    "Experiment 5: Text Summarization",
    "Experiment 6: Virtual Lab",
    "Experiment 7: Network Visualization",
    "Experiment 8: To Implement TF-IDF",
    "Experiment 9: Text Classification using Naive Bayes Classifier"
]

st.set_page_config(
    page_title="Text Summarizer Vlab",
    page_icon=":books:",
    layout="wide",
)
st.markdown("<h1 style='text-align: center; font-size: 48px;'>List of Experiments</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: left; font-size: 36px;'>Select an Experiment</h1>", unsafe_allow_html=True)
exp_selectbox = st.selectbox("", experiments)

# If "Experiment 6" is selected, display the text summarizer code
if exp_selectbox == "Experiment 6: Virtual Lab":
    st.markdown("<h1 style='text-align: left; font-size: 36px;'>Text Summarizer</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: left; font-size: 20px;'>Enter the text to summarize:</h1>", unsafe_allow_html=True)
    text_sample = st.text_area('')
    st.markdown("<h1 style='text-align: left; font-size: 20px;'>Enter the no. of sentences in which you want summary:</h1>", unsafe_allow_html=True)
    numberOfSentences = st.number_input('',value=2)
    col1, col2, col3, col4, col5 = st.columns(5)
    if col3.button('Summarize'):
        with st.spinner('Wait for it...'):
            final_summary=generate_summary(text_sample, numberOfSentences)
        st.write(final_summary)
