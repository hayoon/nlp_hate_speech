#!/usr/bin/env python
# coding: utf-8



def predict_label(comments, df, cosine_sim=cosine_sim): # inputs: comments : str, df :pd.dataframe
    #입력한 코멘트로 부터 인덱스 가져오기
    try: 
        idx = indices[comments][0]
    except:
        idx = indices[comments]

    # 모든 코멘트에 대해서 해당 코멘트와의 유사도를 구하기
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 유사도에 따라 코멘트들을 정렬
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse = True)

    # 가장 유사한 10개의 코멘트를 받아옴
    sim_scores = sim_scores[1:4]

    # 가장 유사한 10개 코멘트의 인덱스 받아옴
    comment_indices = [i[0] for i in sim_scores]
    
    #기존에 읽어들인 데이터에서 해당 인덱스의 값들을 가져온다. 그리고 스코어 열을 추가하여 코사인 유사도도 확인할 수 있게 한다.
    result_df = df.iloc[comment_indices].copy()
    result_df['score'] = [i[1] for i in sim_scores]

    # hate 컬럼의 라벨 값들을 리스트로 변환후 label_list 변수에 저장
    label_list = result_df['hate'].tolist()
    # bias 컬럼의 라벨 값들을 리스트로 변환후 bias_list 변수에 저장
    bias_list = result_df['bias'].tolist()

    if len(set(label_list)) == 1:
        pred = label_list[0]
    
    elif len(set(label_list)) == 2:
        pred = [x for x in label_list if label_list.count(x) > 1][0]
    
    else:
        label_index = result_df[result_df['hate'] != 'none'].index[0]
        pred = result_df['hate'][label_index]
          
    return pred

