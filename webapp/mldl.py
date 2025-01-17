from flask import render_template
from webapp import app
import pandas as pd
from flask import request
import keras
# import requests
from flask import jsonify
from keras.preprocessing.text import Tokenizer
from konlpy.tag import Mecab
from keras.preprocessing.sequence import pad_sequences
import pickle

# 비속어/혐오 필터링 모델 로드
model = keras.models.load_model('모델/model_GRU_0115_cleaned_mecab_stop.keras')
# tokenizer = Tokenizer()

# Tokenizer 불러오기
with open('모델/tokenizer_cleaned_mecab_stop.pkl', 'rb') as f:
    load_tokenizer = pickle.load(f)

max_length = model.get_layer('embedding_1').input_shape[1]


# 불용어 목록 로드
df_stopwords = pd.read_csv('../불용어/all_stopwords_0115.txt', header=None, encoding='utf-8-sig')
stop_words = df_stopwords[0].to_list()

# 기본 페이지
@app.route('/index')
def index():
    return render_template('filter_test.html')


# 비동기 방식으로 입력된 텍스트를 처리하고 결과를 반환
@app.route('/filtering', methods=['POST'])
def filtering():
    # 폼에서 입력된 텍스트를 받아옴
    user_input = request.form['user_input']
    app.logger.debug(user_input)
    print(user_input)

    # 예측
    pre = predict_text(model, load_tokenizer, stop_words, user_input, max_length)

    # 입력된 텍스트를 처리 (여기서는 텍스트 그대로 반환)
    result = f"{user_input} <br> 비속어여부: {pre}"
    
    
    # 결과를 JSON 형태로 반환
    return jsonify({'result': result})

# 불용어 처리
def remove_stopwords(words, stop_words):
    return [word for word in words if word not in stop_words]

# 임의의 텍스트를 판별하는 함수
def predict_text(model, tokenizer, stop_words, text, max_length):
    # 텍스트 전처리
    # text = clean_text(text)

    # 형태소 분석
    mecab = Mecab("C:/mecab/mecab-ko-dic")
    text = mecab.morphs(text)

    # 불용어 제거
    # text = remove_stopwords(text, stop_words)

    # 텍스트를 토큰화하고 패딩
    encoded_text = tokenizer.texts_to_sequences([text])
    padded_text = pad_sequences(encoded_text, maxlen=max_length, padding='post')

    # 모델을 사용해 예측
    prediction = model.predict(padded_text)
    print(prediction)

    # 예측 결과 출력 (0과 1로 반환되므로 0이 '비속어 없음', 1이 '비속어 있음')
    if prediction >= 0.6:
        return "비속어 있음"
    else:
        return "비속어 없음"


'''
# 크롤링 결과 페이지
@app.route('/crawl', methods=['POST'])
def crawl():
    url = request.form['url']  # 사용자로부터 URL 입력 받기
    id_text, class_text = crawl_website(url)  # 웹 크롤링 실행
    return render_template('result.html', id_text=id_text, class_text=class_text, url=url)

@app.route('/weather-data')
def weather_data():
    cities = [
    {"name": "Seoul", "lat": 37.5665, "lon": 126.9780},
    {"name": "Busan", "lat": 35.1796, "lon": 129.0756},
    {"name": "Incheon", "lat": 37.4563, "lon": 126.7052},
    {"name": "Daegu", "lat": 35.8714, "lon": 128.6014},
    {"name": "Daejeon", "lat": 36.3510, "lon": 127.3850},
    # 다른 도시들도 추가할 수 있습니다
    ]

    weather_data = []
    for city in cities:
        url = f"http://www.7timer.info/bin/api.pl?lon={city['lon']}&lat={city['lat']}&product=civil&output=json"
        response = requests.get(url)
        data = response.json()
        reconstruct_place = {
            "name": city["name"],
            "lat": city["lat"],
            "lon": city["lon"],
            "weather": data["dataseries"][0]
        }
        weather_data.append(reconstruct_place)
    return jsonify(weather_data)
'''

'''
@app.route('/reconstruct-data')
def reconstruct_data():
    # places = pd.read_csv('./webapp/reconstruction/d_none.csv')
    places = pd.read_csv('./webapp/reconstruction/df_latlong_cate_added.csv')'''
'''
    places = [
    {"name": "Seoul", "lat": 37.5665, "lon": 126.9780},
    {"name": "Busan", "lat": 35.1796, "lon": 129.0756},
    {"name": "Incheon", "lat": 37.4563, "lon": 126.7052},
    {"name": "Daegu", "lat": 35.8714, "lon": 128.6014},
    {"name": "Daejeon", "lat": 36.3510, "lon": 127.3850},
    # 다른 도시들도 추가할 수 있습니다
    ]
    '''

'''
    # df_filtered = places[['위도', '경도', 'cate_code']]

    # DataFrame을 리스트로 변환 
    reconstruct_data = places.to_dict(orient='records')
    # app.logger.debug(reconstruct_data)
    return jsonify(reconstruct_data)
'''

'''
# ============================여기부터는 함수들===========================


# TO-DO 크롤링 함수. 댓글을 통한 감정 분석 시도도 해보기
def crawl_website(url):
    try:
        with sync_playwright() as p:
            # 브라우저 시작 (Headless 모드)
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # 페이지 열기
            page.goto(url)

            # JavaScript가 완전히 로드될 때까지 기다리기
            page.wait_for_load_state('networkidle')  # 모든 네트워크 요청이 끝날 때까지 기다림

            # 페이지 소스 가져오기
            html = page.content()

            # BeautifulSoup으로 HTML 파싱
            soup = BeautifulSoup(html, 'html.parser')

            # 특정 id를 가진 div 요소 크롤링
            id_content = soup.find('div', id='container')
            if id_content:
                app.logger.debug(id_content)
            else:
                app.logger.debug('no id')
            id_text = id_content.get_text() if id_content else "No element with the given id"

            # 특정 class를 가진 div 요소 크롤링
            class_content = soup.find_all('div', class_='comment_text')
            class_text = [item.get_text() for item in class_content] if class_content else ["No elements with the given class"]

            crawl_prev_page(url)

            # 브라우저 닫기
            browser.close()

            return id_text, class_text

    except Exception as e:
        return f"Error: {e}", []
    
'''
