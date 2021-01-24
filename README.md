한국어 악플 분류를 위한 머신러닝 프로젝트
========================================
본 프로젝트는 Kaggle Competition에서 사용된 데이터 셋을 이용하여 진행되었습니다. 
[Korean hate speech (Kaggle)](https://www.kaggle.com/c/korean-hate-speech-detection/overview)

I. 프로젝트 개요
----------------
1. 프로젝트 배경 및 목적
#### 댓글을 통해서 긍정적인 힘을 얻을 수도 있지만, 악성 댓글로 인해 사람을 망치는 경우가 더 많습니다. 연예인들을 향한 악성 댓글이 특히 더 큰 문제가 되어 왔으며, 이를 해결하기 위해 실명제를 적용해야 한다는 의견들에 결국 연예기사에는 댓글을 작성할 수 없게 되었습니다. 다음과 네이버는 결국 연예기사에 댓글을 작성할 수 없게 만들었고 이는 근본적인 해결 방법이 되지 못합니다.
#### 본 머신러닝 프로젝트에 사용된 데이터셋을 구성한 사람의 의도처럼, 다양한 모델을 사용하며 혐오 발언을 구분할 수 있는 최적의 방법을 찾아보기로 하였습니다.
  
2. 데이터셋 소개
 ### Train data와 Validation data에는 댓글 내용의 수위에 따라 구분되어 있습니다:
  - hate: 글의 대상 또는 관련 인물, 글의 작성자 또는 댓글 등에 대한 강한 혐오 또는 모욕이 표시 됨
  - offensive: 비록 논평이 hate만큼 혐오스럽거나 모욕스럽지 않지만 대상이나 독자를 불쾌하게 만듦
  - none: 어떠한 증오나 모욕도 내포되지 않음
  ##### Train data: 7896 rows / Validation data: 471 rows / Test data: 974 rows (no label)
  
3. EDA
- 2글자 이상 명사 통계: none 라벨과 비교했을 때, offensive와 hate 댓글에서 성별, 비속어, 혐오 표현 단어의 빈도가 높음

![image](https://user-images.githubusercontent.com/28764376/105451950-cac02b80-5cc0-11eb-8c29-5ce707c3d863.png)
![image](https://user-images.githubusercontent.com/28764376/105451997-dd3a6500-5cc0-11eb-96d0-503337a531ea.png)

- 형태소 기준 2-gram 통계: 형태소로 분석해 보았을 때도 마찬가지로 hate 댓글에서 성별, 비속어, 혐오 표현 단어의 빈도가 높음

![image](https://user-images.githubusercontent.com/28764376/105451226-82543e00-5cbf-11eb-8fcd-601ba1c148a7.png)
![image](https://user-images.githubusercontent.com/28764376/105451256-9009c380-5cbf-11eb-923d-9bebc9b79bfc.png)

- 형태소 기준 2-gram 통계: 부정적인 댓글일수록 토큰 수가 많고, 문장의 길이가 더 김

![image](https://user-images.githubusercontent.com/28764376/105451147-55a02680-5cbf-11eb-9f58-57d680e40851.png)

4. 설치 패키지 
- Python 3 (패키지 쓴거 추가해주세용 기억나는것만 썼어요 그리고 이 문장은 지워주세요)
```
pip install pykospacing
pip install kss
pip install konlpy
pip install soynlp
```

II. 전처리
-----------
1. 적용 모델에 따라 전처리를 다르게 진행하거나, 아예 전처리를 하지 않은 부분도 있으나 기본적인 전처리는 아래와 같은 프로세스로 진행 되었음:
- 특수문자 제거
- 문장 분리
- 띄어쓰기
- 중복 제거

2. 토크나이저
- Khaiii
- Okt
- Mecab

III. 모델별 실험 코드 파일 추가하실 분은 따와서 추가해주세여 그리고 이 문장은 지워주세용
----------------
1. 벡터화: Kfold, stratifiedKfold, validation data 등 세 가지 검증 방법에 각각 TF-IDF와 Count vectorizer 방법을 적용해 보았음
- CountVectorizer
- TF-IDFVectorizer
* (코드 파일 링크)

2. 머신러닝 모델: 직접 함수를 생성하기도 하고 다양한 모델링 기법을 사용하며 성능을 개선하기 위해 비교해 보았음
- Naive Bayes
  * 두 가지 전처리 방법으로 모델링을 시도하여, 특수문자와 공백을 모두 제거한 뒤 띄어쓰기한 경우에 성능이 더 좋았던 것으로 나왔음
  * 나이브 베이즈의 'Most informative features'를 통하여 각 라벨에 영향을 주는 단어들을 한눈에 확인할 수 있었음
  * validation data: F1-score 0.525420
  * [나이브 베이즈를 이용한 분류 예측](https://github.com/hayoon/nlp_hate_speech/blob/master/code/yeji/02_naivebayes_trial.ipynb)
- 코사인 유사도1
  * 훈련 데이터를 각 레이블별로 분류해 TFIDF Vectorize한 뒤, 각각의 평균 벡터값 산출
  * 'comment' 컬럼의 벡터값 <-> 각 레이블 평균 벡터값 간의 유사도 측정 후, 가장 유사도가 높은 레이블 알려주기
  * validation data: F1-score 0.498306
  * (코드 파일 링크)
- 코사인 유사도1
  * 댓글을 입력할 시, 코사인 유사도 top3 댓글을 출력하는 함수를 생성. khaiii를 이용한 형태소 분석 이전의 Naive Bayes를 써서 얻은 informative feature의 상위 100개에 포함되는 품사 단어들만 vocabulary list에 추가하여 분석하도록 함
  * 출력된 유사도가 높은 댓글 top3의 라벨이 모두 일치할 경우, 해당 라벨과 가장 유사하다고 판단하여 같은 라벨로 예측
  * 출력된 유사도가 높은 댓글 top3의 라벨이 두 가지가 나왔을 경우, 유사도 값의 차이가 있을 수 있기 때문에 같은 라벨을 가진 값들끼리 유사도를 더하며, bias가 존재한다면 0.1을, 이것이 gender bias라면 0.2를 또 더해줘서 최종적으로 높은 값을 가지는 라벨로 예측
  * 출력된 유사도가 높은 댓글 top3릐 라벨이 모두 다르다면, 부정적인 댓글이 최수 두 가지 (offensive, hate)이기 때문에 마찬가지로 bias에 대한 가중치를 적용하고, 이 두 라벨 중 유사도가 더 높은 라벨로 예측
  * validation data: F1-score (숫자 채워주세용)
  * (코드 파일 링크)
- 그 외 분류기들
  * Random Forest
    * validation data: F1-score (숫자 채워주세용)
  * Support Vector Machine
    * validation data: F1-score (숫자 채워주세용)
  * LightGBM
    * validation data: F1-score (숫자 채워주세용)
  * 전처리 과정 쓰세요
  * (코드 파일 링크)

IV. Logistic Regression에 집중한 분류
--------------------------------------
1.
2.
3.
4.

V. 딥러닝 (Bert)
----------------


아쉬운 점과 앞으로 나아갈 수 있는 방향
---------------------------------------
#### 각 머신러닝 모델들의 성능 결과에 대해 깊이 이해하지 못하고 넘어간 점에서 아쉬움

Built with: 대충 쓰긴 했는데 각자 한 일 추가할거 있으면 해주세용 그리고 이 문장은 지워주세용
----------
- 김예지
  * 전처리, 다양한 머신러닝 모델링, 코사인 유사도 이용한 분류, doc2vec 사용하여 모델 성능 개선 시도
  * Github: https://github.com/yeji0701
- 이기중
  * 전처리, 다양한 머신러닝 모델링, Logistic Regression에 집중한 성능 개선,
  * Github:
- 정하윤
  * 전처리, 다양한 머신러닝 모델링, 코사인 유사도 이용한 분류, BERT 이용하여 성능 개선
  * Github:
- 최재철
  * 전처리, 다양한 머신러닝 모델링, 코사인 유사도 이용한 분류, word2vec 사용하여 모델 성능 개선 시도
  * Github:
  
 Acknowledgements:
 -----------------
 생각나는 거 없는데 있다면 추가해주세용 그리고 이 문장은 지워주세용
