한국어 악플 분류를 위한 머신러닝 프로젝트
========================================
본 프로젝트는 Kaggle Competition에서 사용된 데이터 셋을 이용하여 진행되었습니다. 
[Korean hate speech (Kaggle)](https://www.kaggle.com/c/korean-hate-speech-detection/overview)

I. 프로젝트 개요
----------------
1. 프로젝트 배경 및 목적
### 댓글을 통해서 긍정적인 힘을 얻을 수도 있지만, 악성 댓글로 인해 사람을 망치는 경우가 더 많습니다. 연예인들을 향한 악성 댓글이 특히 더 큰 문제가 되어 왔으며, 이를 해결하기 위해 실명제를 적용해야 한다는 의견들에 결국 연예기사에는 댓글을 작성할 수 없게 되었습니다. 다음과 네이버는 결국 연예기사에 댓글을 작성할 수 없게 만들었고 이는 근본적인 해결 방법이 되지 못합니다.
### 본 머신러닝 프로젝트에 사용된 데이터셋을 구성한 사람의 의도처럼, 다양한 모델을 사용하며 혐오 발언을 구분할 수 있는 최적의 방법을 찾아보기로 하였습니다.
  
2. 데이터셋 소개
 ### Train data와 Validation data에는 댓글 내용의 수위에 따라 구분되어 있습니다:
  - hate: 글의 대상 또는 관련 인물, 글의 작성자 또는 댓글 등에 대한 강한 혐오 또는 모욕이 표시 됨
  - offensive: 비록 논평이 hate만큼 혐오스럽거나 모욕스럽지 않지만 대상이나 독자를 불쾌하게 만듦
  - none: 어떠한 증오나 모욕도 내포되지 않음
  ##### Train data: 7896 rows
  ##### Validation data: 471 rows
  ##### Test data: 974 rows (no label)
  
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
- Python 3
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

III. 모델별 실험
----------------
1. 벡터화
- CountVectorizer
- TF-IDFVectorizer

2. 머신러닝 모델
- Naive Bayes
- 코사인 유사도1
- 코사인 유사도1
- 분류기들

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
