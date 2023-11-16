import main
import numpy as np
import pandas as pd
import streamlit as st
import os
from datetime import datetime


# ## Header/Subheader
# st.header('NEO K-means')
# st.subheader('고객 세그멘테이션')

st.sidebar.title('NEO k-means \n클러스터링🌸')

def save_uploaded_file(directory, file):
    '''파일 저장 함수'''
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(os.path.join(directory, file.name), 'wb') as f:
        f.write(file.read())
        
    st.success('파일 업로드 성공')


csv_file = st.file_uploader('원하는 CSV 파일을 업로드 해주세요', type=['csv'])
print(csv_file)
if csv_file is not None:
    current_time = datetime.now()
    filename = current_time.isoformat().replace(':', '_') + '.csv'
    csv_file.name = filename
    save_uploaded_file('csv', csv_file)
    
    # csv를 보여주기 위해 pandas 데이터 프레임으로 만들어야한다.
    df = pd.read_csv('csv/'+filename)
    # st.dataframe(df)

if csv_file is not None: # csv파일을 선택하면
    # 사용자에게 보여줄 칼럼을 선택
    st.subheader('고객 세그멘테이션')
    selected_columns = st.sidebar.multiselect('원하는 칼럼을 선택하세요', df.columns)
    k = st.sidebar.slider("군집개수를 선택하세요.", 2, 10)

    # 선택된 칼럼에 해당하는 데이터프레임 표시
    if selected_columns:
        st.write(df[selected_columns])
        neo_df = df[selected_columns].dropna(axis=0) # 결측행 제거
        neo_df.reset_index() # 인덱스 초기화
        num_columns = neo_df.select_dtypes(include=np.number).columns.tolist() # 수치형 변수만 선택
        num_df = neo_df[num_columns]
        num_df = np.array(num_df) # array로 만들어야 돌아감

        U = main.cluster(num_df, k)
        U_df = pd.DataFrame(U)
        st.write(U_df)

    else:
        st.write('칼럼을 선택하세요.')
