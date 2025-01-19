from clean_text import clean_str
from keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Mecab
from xgboost import Booster, DMatrix

import json
import keras
import numpy as np
import pandas as pd
import pickle


'''
전처리==================================================================
'''
m = Mecab(dicpath="C:/mecab/mecab-ko-dic")

SPELL_KEY = ('b698e6ca14d0b9f707b24c770f6cefdbd8c5dfa4', '1737281827549')

def remove_stopwords(text, stop_words):
    # 불용어 제거
    for stop_word in stop_words[0].values:
        text = text.replace(stop_word, '')
    return text

def remove_stopword(list, stop_word):
    # 불용어 제거
    list = [i for i in list if i not in stop_word[0].values]
    return list

# JSON 파일에서 feature_names 추출 함수
def load_feature_names(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        feature_names = json_data.get('learner', {}).get('feature_names', [])
        return feature_names

'''
감정 분류================================================================
'''

def predict_emotion(text):

    text = clean_str(text, SPELL_KEY)
    print('텍스트 전처리: ', text)

    # 불용어 어절 제거
    stop_word = pd.read_csv('../불용어/1_stopword_fixed.txt', header=None)
    stop_words = pd.read_csv('../불용어/2_stopword_fixed.txt', header=None)

    text = remove_stopwords(text, stop_words)
    print('불용어 어절 제거: ', text)

    # 품사 추출
    text = m.pos(text)
    tags = ['NNG', 'NNP', 'VA', 'VV', 'XR']
    text = [i[0] for i in text if i[1] in tags or i[1].startswith('VA') or i[1].startswith('VV')]
    print('품사 추출: ', text)

    text = remove_stopword(text, stop_word)
    print('불용어 제거: ', text)

    # Feature Names 로드
    feature_names = load_feature_names('모델/model_0115_XGBoost.json')

    # XGBoost 모델 로드
    booster_model = Booster()
    booster_model.load_model('모델/model_0115_XGBoost.model')

    # Feature count 기반으로 입력 데이터 생성
    padded_text = np.array([text.count(f) for f in feature_names]).reshape(1, -1)
    dmatrix = DMatrix(padded_text, feature_names=feature_names)


    # 예측 수행
    pre = booster_model.predict(dmatrix)
    emotion_label = ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']


    # 예측된 결과를 감정 라벨로 변환
    print('모델 출력: ', pre)
    index = np.argmax(pre)
    percentage = f"{pre[0][index] * 100:.2f}"
    emotion_result = emotion_label[index]
    
    print('감정 분류 결과: ', emotion_result)
    print('정확도: ', percentage)
    return {'result': emotion_result, 'percentage': percentage}



'''
비속어 필터링==============================================================
'''
def predict_filter(text):
    # 비속어/혐오 필터링 모델 로드
    model = keras.models.load_model('모델/model_GRU_0115_cleaned_mecab_stop.keras')

    # Tokenizer 불러오기
    with open('모델/tokenizer_cleaned_mecab_stop.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    max_length = model.get_layer('embedding_1').input_shape[1]

    # 텍스트를 토큰화하고 패딩
    encoded_text = tokenizer.texts_to_sequences([text])
    padded_text = pad_sequences(encoded_text, maxlen=max_length, padding='post')

    # 모델을 사용해 예측
    prediction = model.predict(padded_text)
    percentage = f"{prediction[0][0] * 100:.2f}"

    # 예측 결과 출력 (0과 1로 반환되므로 0이 '비속어 없음', 1이 '비속어 있음')
    if prediction >= 0.6:
        fileter_result = "비속어 있음"
    else:
        fileter_result = "비속어 없음"


    print("filter percentage: ",percentage)
    print('filter result: ', fileter_result)

    return {'result': fileter_result, 'percentage': percentage}


if __name__ == "__main__":
    text = "바보같이 기뻐"
    predict_emotion(text)
    print('==============================================')
    predict_filter(text)