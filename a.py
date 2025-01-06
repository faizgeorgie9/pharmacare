import streamlit as st
import base64
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from sklearn.preprocessing import Binarizer
from nltk.corpus import stopwords
import re



def load_corpus(json_file):
    with open(json_file, 'r') as file:
        corpus = json.load(file)
    processed_corpus = {condition: set(keywords) for condition, keywords in corpus.items()}
    return processed_corpus

df = pd.read_csv("dataset/drug_data_bersih.csv")

corpus_path = "dataset/conditions_corpus.json"
corpus = load_corpus(corpus_path)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


scaler = MinMaxScaler()
df[['Jumlah Ulasan', 'Keefektifan', 'Kemudahan', 'Kepuasan']] = scaler.fit_transform(
    df[['Jumlah Ulasan', 'Keefektifan', 'Kemudahan', 'Kepuasan']]
)

df['Kombinasi_Fitur'] = df.apply(
    lambda row: f"{row['Informasi']} {row['Interaksi Obat']}",
    axis=1
)

def get_sentence_vector(sentence, model):

    vectors = [model.wv[word] for word in sentence if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

ficur = df['Kombinasi_Fitur'].apply(lambda x: x.split()).to_list()

w2v_model = Word2Vec(sentences=ficur, vector_size=100, window=5, min_count=1, workers=4)

def jaccard_similarity(doc1, doc2):
    """
    Hitung Jaccard similarity antara dua dokumen teks.
    """
    tokens1 = set(doc1.split())
    tokens2 = set(doc2.split())
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union) if union else 0


def rekomen_obat_jaccard_tfidf(df, condition, top_n=5):
    df_condition = df[df['Kondisi'].str.contains(condition, case=False, na=False)]

    if df_condition.empty:
        print(f"Tidak ditemukan obat untuk kondisi '{condition}'.")
        return None

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_condition['Kombinasi_Fitur'])

    binarizer = Binarizer(threshold=0)
    binary_matrix = binarizer.fit_transform(tfidf_matrix.toarray())

    input_vector = binary_matrix[-1]
    jaccard_scores = []
    for vector in binary_matrix:
        intersection = np.sum(np.minimum(input_vector, vector))
        union = np.sum(np.maximum(input_vector, vector))
        jaccard_scores.append(intersection / union if union != 0 else 0)

    df_condition['Jaccard Score'] = jaccard_scores

    recommended_drugs = df_condition.iloc[:-1].sort_values(by='Jaccard Score', ascending=False).head(top_n)

    recommended_drugs = recommended_drugs.drop_duplicates(
        subset=['Obat 1', 'Tipe Obat', 'Keefektifan', 'Kemudahan', 'Kepuasan', 'Informasi']
    )

    return recommended_drugs[['Obat 1', 'Tipe Obat', 'Keefektifan', 'Kemudahan', 'Kepuasan', 'Informasi', 'Jaccard Score']]


def rekomen_obat_jaccard_w2v(df, condition, top_n=5):
    df_condition = df[df['Kondisi'].str.contains(condition, case=False, na=False)]

    if df_condition.empty:
        print(f"Tidak ditemukan obat untuk kondisi '{condition}'.")
        return None

    input_doc = ' '.join(df_condition.iloc[-1]['Kombinasi_Fitur'].split())
    jaccard_scores = df_condition['Kombinasi_Fitur'].apply(lambda x: jaccard_similarity(input_doc, x))

    w2v_features = np.array([get_sentence_vector(doc, w2v_model) for doc in df_condition['Kombinasi_Fitur']])

    df_condition['Jaccard Score'] = jaccard_scores

    numerical_features = df_condition[['Jumlah Ulasan', 'Keefektifan', 'Kemudahan', 'Kepuasan']].values
    final_features = np.hstack([w2v_features, numerical_features])

    recommended_drugs = df_condition.sort_values(by='Jaccard Score', ascending=False).head(top_n)

    recommended_drugs = recommended_drugs.drop_duplicates(
        subset=['Obat 1', 'Tipe Obat', 'Keefektifan', 'Kemudahan', 'Kepuasan', 'Informasi']
    )

    return recommended_drugs[['Obat 1', 'Tipe Obat', 'Keefektifan', 'Kemudahan', 'Kepuasan', 'Informasi', 'Jaccard Score']]



def rekomen_obat_cosine(df, condition, top_n=5):
    df_condition = df[df['Kondisi'].str.contains(condition, case=False, na=False)]

    if df_condition.empty:
        print(f"Tidak ditemukan obat untuk kondisi '{condition}'.")
        return None

    tfidf_agg = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_agg_matrix = tfidf_agg.fit_transform(df_condition['Kombinasi_Fitur'])

    numerical_features = df_condition[['Jumlah Ulasan', 'Keefektifan', 'Kemudahan', 'Kepuasan']].values
    final_features = np.hstack([tfidf_agg_matrix.toarray(), numerical_features])

    cosine_sim = cosine_similarity(final_features)

    similarity_scores = cosine_sim[-1][:-1]
    drug_indices = np.argsort(similarity_scores)[-top_n:][::-1]

    recommended_drugs = df_condition.iloc[drug_indices].copy()
    recommended_drugs['Similarity Score'] = similarity_scores[drug_indices]

    recommended_drugs = recommended_drugs.drop_duplicates(
        subset=['Obat 1', 'Tipe Obat', 'Keefektifan', 'Kemudahan', 'Kepuasan', 'Informasi']
    )

    return recommended_drugs[['Obat 1', 'Tipe Obat', 'Keefektifan', 'Kemudahan', 'Kepuasan', 'Informasi', 'Similarity Score']]

def rekomen_obat_cosine2(df, condition, top_n=5):
    df_condition = df[df['Kondisi'].str.contains(condition, case=False, na=False)]

    if df_condition.empty:
        print(f"Tidak ditemukan obat untuk kondisi '{condition}'.")
        return None

    corpus = df_condition['Kombinasi_Fitur'].apply(lambda x: x.split()).to_list()
    w2v_features = np.array([get_sentence_vector(sentence, w2v_model) for sentence in corpus])

    numerical_features = df_condition[['Jumlah Ulasan', 'Keefektifan', 'Kemudahan', 'Kepuasan']].values
    final_features = np.hstack([w2v_features, numerical_features])

    cosine_sim = cosine_similarity(final_features)

    similarity_scores = cosine_sim[-1][:-1]
    drug_indices = np.argsort(similarity_scores)[-top_n:][::-1]

    recommended_drugs = df_condition.iloc[drug_indices].copy()
    recommended_drugs['Similarity Score'] = similarity_scores[drug_indices]

    recommended_drugs = recommended_drugs.drop_duplicates(
        subset=['Obat 1', 'Tipe Obat', 'Keefektifan', 'Kemudahan', 'Kepuasan', 'Informasi']
    )
    return recommended_drugs[['Obat 1', 'Tipe Obat', 'Keefektifan', 'Kemudahan', 'Kepuasan', 'Informasi', 'Similarity Score']]

def preprocess_text(text):

    additional_stop_words = {'and', 'or', 'but', 'so', 'because', 'my', 'get', 'feel', 'today', 'i'}
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.lower().split()
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and word not in additional_stop_words
    ]
    return tokens

def filter_keywords(user_input, corpus):
    user_keywords = preprocess_text(user_input)
    valid_keywords = []

    for condition, keywords in corpus.items():
        for keyword in keywords:
            processed_keyword = preprocess_text(keyword)
            if all(word in user_keywords for word in processed_keyword):
                valid_keywords.append((condition, keyword))
    return valid_keywords




def add_background(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded_string}");
                background-size: cover;
                background-position: center;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

def main():
    st.set_page_config(page_title="Medicine Recommendation", page_icon="💊", layout="wide")

    if 'start' not in st.session_state:
        st.session_state['start'] = False

    if not st.session_state['start']:
        add_background("static/bg1.png")
        st.title("Sistem Rekomendasi Obat")
        st.write("Temukan Obat Yang Tepat Berdasarkan Keluhan Anda")
        st.subheader("Yuk Temukan Obat !")
        if st.button("Mulai"):
            st.session_state['start'] = True
    else:
        add_background("static/bg2.png")
        st.button("Back", on_click=lambda: st.session_state.update({'start': False}))
        st.markdown("<h2 style='text-align: center; color: white;'>Cek Obat Mu</h2>", unsafe_allow_html=True)
        user_input = st.text_input("Deskripsikan Kondisi Yang Kamu Alami:")

        if st.button("Kirim"):
            filtered_input = ' '.join(preprocess_text(user_input))
            filtered_keywords = filter_keywords(filtered_input, corpus)
            if filtered_keywords:
                all_recommended_drugs = {}

                for condition, keyword in filtered_keywords:
                    st.write(f"**Proses keyword:** {keyword} (Kondisi: {condition})")
                    recommended_drugs = rekomen_obat_cosine(df, condition, top_n=5)
                    if recommended_drugs is not None:
                        all_recommended_drugs[keyword] = recommended_drugs

                if len(all_recommended_drugs) > 1:
                    keys = list(all_recommended_drugs.keys())
                    drugs1 = all_recommended_drugs[keys[0]]
                    drugs2 = all_recommended_drugs[keys[1]]

                    interaksi_obat = df[['Obat 1', 'Obat 2', 'Interaksi Obat']]
                    interaksi = interaksi_obat[interaksi_obat['Obat 1'].isin(drugs1['Obat 1']) &
                                               interaksi_obat['Obat 2'].isin(drugs2['Obat 1'])]

                    if not interaksi.empty:
                        st.write("**Interaksi antara obat ditemukan:**")
                        interaksi_unique = interaksi.drop_duplicates(subset=['Obat 1', 'Obat 2', 'Interaksi Obat'])
                        st.table(interaksi_unique[['Obat 1', 'Obat 2', 'Interaksi Obat']])
                    else:
                        st.write("**Tidak ada interaksi antara obat dari dua kondisi. Berikut adalah rekomendasi terpisah:**")
                        st.write(f"**Rekomendasi untuk keyword '{keys[0]}':**")
                        st.table(drugs1)
                        st.write(f"**Rekomendasi untuk keyword '{keys[1]}':**")
                        st.table(drugs2)
                else:
                    for keyword, drugs in all_recommended_drugs.items():
                        st.write(f"**Rekomendasi untuk keyword '{keyword}':**")
                        st.table(drugs)
            else:
                st.write("**Tidak ada keyword yang cocok ditemukan dalam corpus.**")


if __name__ == "__main__":
    main()
