import pandas as pd
import numpy as np
import re
import os
import pickle
import lightgbm as lgb

from sklearn.feature_extraction.text import TfidfVectorizer
from underthesea import text_normalize, word_tokenize
from sklearn.preprocessing import LabelEncoder

TOOLS_FOLDER = "data/tools/"

# Đọc nội dung từ các file văn bản
def read_file_to_list(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return [line.strip() for line in lines]
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []

EMOJICON_LIST = read_file_to_list(TOOLS_FOLDER + "emojicon.txt")
ENGLISH_VNMESE_LIST = read_file_to_list(TOOLS_FOLDER + "english-vnmese.txt")
TEENCODE_LIST = read_file_to_list(TOOLS_FOLDER + "teencode.txt")
VIETNAMESE_STOPWORDS_LIST = read_file_to_list(TOOLS_FOLDER + "vietnamese-stopwords.txt")
WRONG_WORD_LIST = set(read_file_to_list(TOOLS_FOLDER + "wrong-word.txt"))

# Chuyển đổi danh sách bằng cách tách dòng theo dấu \t
EMOJICON_DICT = {emoji.split('\t')[0]:emoji.split('\t')[1] for emoji in EMOJICON_LIST}
ENGLISH_VNMESE_DICT = {vnmese.split('\t')[0]:vnmese.split('\t')[1] for vnmese in ENGLISH_VNMESE_LIST}
TEENCODE_DICT = {teen.split('\t')[0]:teen.split('\t')[1] for teen in TEENCODE_LIST}

# load models
with open("models/model_lgb_weighted.pkl", "rb") as model_file:
    model_lgb = pickle.load(model_file)

with open("models/vectorizer.pkl", "rb") as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

with open("models/label_encoder.pkl", "rb") as label_encoder_file:
    label_encoder = pickle.load(label_encoder_file)

# Sử dụng normalize_text chuẩn hoá dữ liệu, chuyển thành chữ thường và bỏ các kí tự đặc biệt
def normalize_text_manually(text):
    text = text.lower()
    text = text_normalize(text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    replace_emoji_words = [EMOJICON_DICT.get(word, word) for word in words]
    text = " ".join(words)

    words = word_tokenize(text)
    replace_vnmese_words = [ENGLISH_VNMESE_DICT.get(word, word) for word in replace_emoji_words]
    replace_teencode_words = [TEENCODE_DICT.get(word, word) for word in replace_vnmese_words]

    return ' '.join(replace_teencode_words)

def preprocess_text(text):
    text = normalize_text_manually(text)
    return text

def predict_sentiment(text):
    original_text = text[:]

    for i in range(len(text)):
        text[i] = preprocess_text(text[i])
    text = tfidf_vectorizer.transform(text)
    prediction = model_lgb.predict(text)
    prediction = label_encoder.inverse_transform(prediction)

    result_df = pd.DataFrame({
        'noi_dung_binh_luan': original_text,
        'sentiment': prediction
    })
    return result_df

if __name__ == "__main__":
    sample_text = "Sản phẩm này dùng không tốt"
    sentiment = predict_sentiment(sample_text)
    print(f"The predicted sentiment is: {sentiment}")