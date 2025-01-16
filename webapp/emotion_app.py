from flask import render_template, request
from webapp import app
from flask import jsonify
from emotion import predict_emotion

# 기본 페이지
@app.route('/')
def index_emotion():
    return render_template('emotion_test.html')


# 비동기 방식으로 입력된 텍스트를 처리하고 결과를 반환
@app.route('/classifying', methods=['POST'])
def classifying():
    # 폼에서 입력된 텍스트를 받아옴
    user_input = request.form['user_input']
    app.logger.debug(user_input)

    # 예측
    pre = predict_emotion(user_input)

    # 입력된 텍스트를 처리 (여기서는 텍스트 그대로 반환)
    result = f"{user_input} <br> 감정 분류 결과: {pre}"
    
    # 결과를 JSON 형태로 반환
    return jsonify({'result': result})