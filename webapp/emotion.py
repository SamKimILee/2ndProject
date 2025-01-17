from clean_text import clean_str
import json
from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Mecab

import numpy as np
import keras
import pandas as pd

SPELL_KEY = ('5ed196d426d8d449f3745001d1a13f5c1080f2ac', '1737006088018')

def remove_stopwords(text, stop_words):
    # 불용어 제거
    for stop_word in stop_words[0].values:
        text = text.replace(stop_word, '')
    return text

def predict_emotion(text):
    # Tokenizer 불러오기
    with open('모델/emotion_tokenizer.json', 'r') as f:
        tokenizer_json = json.load(f)  # JSON 형식으로 로드
        tokenizer = tokenizer_from_json(tokenizer_json)  # Tokenizer 객체로 변환

    m = Mecab(dicpath="C:/mecab/mecab-ko-dic")

    max_length = 8
    print(text)
    text = clean_str(text, SPELL_KEY)
    print('텍스트 전처리: ', text)

    # 불용어 어절 제거
    stop_words = pd.read_csv('../불용어/2_stopword_fixed.txt', header=None)

    text = remove_stopwords(text, stop_words)
    print('불용어 어절 제거: ', text)

    # 품사 추출
    text = m.pos(text)
    tags = ['NNG', 'VA', 'VV', 'XR']
    text = [i[0] for i in text if i[1] in tags]
    print('품사 추출: ', text)

    # 텍스트를 토큰화하고 패딩
    encoded_text = tokenizer.texts_to_sequences([text])
    padded_text = pad_sequences(encoded_text, maxlen=max_length, padding='post')

    emotion_model = keras.models.load_model('모델/emotion_model.keras')

    # 1. 단어 리스트를 다시 문장으로 결합
    sentence = ' '.join(text)

    # 2. 토크나이저로 시퀀스 변환
    sequences = tokenizer.texts_to_sequences([sentence])
    sequences_len = len(sequences)

    pre = emotion_model.predict(padded_text, sequences_len)

    # emotion_label = ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']
    emotion_label = ['공포', '행복', '분노', '슬픔', '중립', '행복', '혐오']
    print('모델 출력: ', pre)
    index = np.argmax(pre)
    percentage = f"{pre[0][index] * 100:.2f}"
    emotion_result = emotion_label[index]

    print('감정 분류 결과: ', emotion_result)
    print('정확도: ', percentage)
    return {'result': emotion_result, 'percentage': percentage}