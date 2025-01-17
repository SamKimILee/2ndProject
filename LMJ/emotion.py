from clean_text import clean_str
import numpy as np
import json
from xgboost import Booster, DMatrix
from konlpy.tag import Mecab
import pandas as pd

SPELL_KEY = ('d2cd9d51cafed2595fe629dc5f80d2fcf0b89529', '1737075807896')

# JSON 파일에서 feature_names 추출 함수
def load_feature_names(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        feature_names = json_data.get('learner', {}).get('feature_names', [])
        return feature_names

def remove_stopwords(text, stop_words):
    # 불용어 제거
    for stop_word in stop_words[0].values:
        text = text.replace(stop_word, '')
    return text

def predict_emotion(text):
    # Feature Names 로드
    feature_names = load_feature_names('C:/labs_python/2ndProject/LMJ/model_0115_XGBoost.json')
    print("Feature Names:", feature_names)

    m = Mecab(dicpath="C:/mecab/mecab-ko-dic")

    print("원본 텍스트: ", text)

    # 텍스트 전처리
    text = text.replace('!', '').strip()
    print('정제된 텍스트: ', text)

    # 불용어 어절 제거
    stop_words = pd.read_csv('C:/labs_python/2ndProject/LMJ/2_stopword_fixed.txt', header=None)

    text = remove_stopwords(text, stop_words)
    print('불용어 어절 제거: ', text)

    # 품사 추출
    text = m.pos(text)
    tags = ['NNG', 'VA', 'VV', 'XR']
    text = [i[0] for i in text if i[1] in tags]
    print('품사 추출: ', text)

    # XGBoost 모델 로드
    booster_model = Booster()
    booster_model.load_model('C:/labs_python/2ndProject/LMJ/model_0115_XGBoost.model')

    # 예측 수행
    # Feature count 기반으로 입력 데이터 생성
    padded_text = np.array([text.count(f) for f in feature_names]).reshape(1, -1)
    print("생성된 입력 데이터:", padded_text)
    dmatrix = DMatrix(padded_text, feature_names=feature_names)
    pre = booster_model.predict(dmatrix)

     # 각 감정별 확률 출력
    emotion_label = ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']
    emotion_probabilities = {emotion_label[i]: pre[0][i] for i in range(len(emotion_label))}
    print("감정별 확률:", emotion_probabilities)

    # 예측된 결과를 감정 라벨로 변환
    predicted_emotion = emotion_label[int(np.argmax(pre))]
    print('예측된 감정:', predicted_emotion)

    return predicted_emotion, emotion_probabilities


# 테스트 코드
if __name__ == "__main__":
    test_text = "우울해"
    try:
        emotion = predict_emotion(test_text)
        print(f"입력 텍스트: {test_text}")
        print(f"예측된 감정: {emotion}")
    except Exception as e:
        print(f"에러 발생: {e}")
