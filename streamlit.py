import main
import visualization
import pandas as pd
import streamlit as st
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

st.title('🔍 NEO k-means 클러스터링')

def save_uploaded_file(directory, file):
    '''파일 저장 함수'''
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(os.path.join(directory, file.name), 'wb') as f:
        f.write(file.read())

####### 아직 지정되지 않은 변수
k = None

csv_file = st.sidebar.file_uploader('**💻 Load a CSV file**', type=['csv'])

print(csv_file)
if csv_file is not None:
    current_time = datetime.now()
    filename = current_time.isoformat().replace(':', '_') + '.csv'
    csv_file.name = filename
    save_uploaded_file('csv', csv_file)
    
    # csv를 보여주기 위해 pandas 데이터 프레임으로 만들어야한다.
    df = pd.read_csv('csv/'+filename)

    # 데이터 샘플 보여주기
    st.write('데이터 샘플')
    st.dataframe(df.head(2))

if csv_file is not None: # csv파일을 선택하면
    # 사용자에게 보여줄 칼럼을 선택
    st.subheader('고객 세그멘테이션')
    selected_columns = st.sidebar.multiselect('**📝 Select Columns**', df.columns)
    selected_df = df[selected_columns]
    k = st.sidebar.slider("**💡Select Number of Clusters**", 2, 10) # 클러스터 개수 : k

    ##### 선택된 칼럼에 해당하는 데이터프레임 표시 #####
    if len(selected_columns) >= 2: # 선택된 변수가 두 개 이상일 때만
        df_pca, arr_pca = visualization.label_preprocessing(selected_df)
        NEOU = main.cluster(arr_pca, k)
        grouped_final_df, U_columns = visualization.create_U_dataframe(k, NEOU, df_pca)
        visualization.display_cluster_2d(grouped_final_df, k)
        st.pyplot(plt.gcf())  # 현재 그림의 핸들을 전달

    else:
        st.write('칼럼을 선택하세요.')

try:
    # U_df 칼럼명 : cluster_1, cluster_2 .. 이런식으로 바꾸기
    U_df = pd.DataFrame(NEOU)
    U_df.columns = ["cluster_{}".format(i) for i in range(1, k+1)]
    U_df.reset_index() # 인덱스 초기화

    ##### 클러스터 번호 선택하면 그 클러스터의 특징 보여줌
    st.subheader('클러스터별 특징')
    column = st.selectbox(
        '확인하고 싶은 변수 선택',
        selected_columns)

    visualization.feature_display(grouped_final_df,column,k)
    st.pyplot(plt.gcf())
except:
    pass