"""
실행파일
"""
from webapp import app
# __init__.py는 import가 실행되면 자동으로 먼저 실행됨. 
# import할때 app.py 실행. 그 다음 한번 더 실행

if __name__ == '__main__':
    #app.run() 
    app.run(debug=True)
#app.run(host='172.16.10.248')
#app.run(host='0.0.0.0')

