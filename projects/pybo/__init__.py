
from flask import Flask
# 플라스크 앱을 생성하는 코드, __name__이라는 변수에는 모듈명(여기서는 pybo)이 담긴다
app = Flask(__name__)


@app.route('/')
def hello_pybo():
    return 'Hello, Pybo!'
