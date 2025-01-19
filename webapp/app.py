from flask import Flask, jsonify, render_template, request
from emotion_filtering import predict_emotion, predict_filter

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('test.html')

@app.route('/api', methods=['POST'])
def predict():
    # 폼에서 입력된 텍스트를 받아옴
    text = request.form['user_input']

    # 예측
    pre_filter = predict_filter(text)
    pre_emo = predict_emotion(text)    

    # 결과를 JSON 형태로 반환
    return jsonify({'filter_result': pre_filter['result'],
                    'filter_percentage': pre_filter['percentage'],
                    'emo_result': pre_emo['result'], 
                    'emo_percentage': pre_emo['percentage']})


if __name__ == '__main__':
    app.run(debug=True)