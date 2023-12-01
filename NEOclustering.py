import main
import numpy as np
import pandas as pd
import NEOclustering as st
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# ## Header/Subheader
# st.header('NEO K-means')
# st.subheader('고객 세그멘테이션')

st.title('🔍 NEO k-means 클러스터링')

def save_uploaded_file(directory, file):
    '''파일 저장 함수'''
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(os.path.join(directory, file.name), 'wb') as f:
        f.write(file.read())


def preprocessing(input_X) :
    '''레이블 링코딩, 스케일링 전처리 함수'''
    X = input_X[:]
    label_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
    label_encoder = LabelEncoder()
    X[label_cols] = X[label_cols].apply(lambda col: label_encoder.fit_transform(col))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


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
    st.dataframe(df.head())

if csv_file is not None: # csv파일을 선택하면
    # 사용자에게 보여줄 칼럼을 선택
    st.subheader('고객 세그멘테이션')
    selected_columns = st.sidebar.multiselect('**📝 Select Columns**', df.columns)
    k = st.sidebar.slider("**💡Select Number of Clusters**", 2, 10) # 클러스터 개수 : k

    ##### 선택된 칼럼에 해당하는 데이터프레임 표시 #####
    if selected_columns:
        neo_df = df[selected_columns].dropna(axis=0) # 결측행 제거
        neo_df.reset_index() # 인덱스 초기화

        preprocessed_df = preprocessing(neo_df)
        num_df = np.array(preprocessed_df) # array로 만들어야 돌아감

        U = main.cluster(num_df, k)
        U_df = pd.DataFrame(U)
        # U_df 칼럼명 : cluster_1, cluster_2 .. 이런식으로 바꾸기
        U_df.columns = ["cluster_{}".format(i) for i in range(1, k+1)]
        U_df.reset_index() # 인덱스 초기화

        ##### 클러스터 번호 선택하면 그 클러스터의 특징 보여줌
        st.subheader('클러스터별 특징')
        cluster_df = pd.concat([neo_df, U_df], axis=1, join='inner') # 원데이터(결측행 제거) + 클러스터 결과

    else:
        st.write('칼럼을 선택하세요.')

    col1, col2 = st.columns(2)

    with col1:
        cluster = st.radio(
                "Select Cluster 👉",
                ["cluster_{}".format(i) for i in range(1, k+1)],
                index=None
        )
        st.write("you select", cluster)

    with col2:
        column = st.selectbox(
            "Select Columns",
            selected_columns,
        )

    if cluster != None:
        clust = cluster_df.loc[cluster_df[cluster] == 1 , [cluster, column]]
        st.write(cluster_df[column].dtypes)
    else:
        st.write("No selection")

    clus_char, origin_char = st.columns(2)

    with clus_char:
        if clust[column].dtypes == object:
            clu_cnt = clust.groupby(by=column).count()
            origin_cnt = pd.DataFrame(neo_df.groupby(by=column).count().iloc[:,0])
            origin_cnt.columns.values[0] = 'Total'

            cnt = pd.concat([clu_cnt, origin_cnt], axis=1)
            st.write(cnt)
            st.bar_chart(cnt)


        else:
            st.write(clust)
            origin = neo_df[column]
            st.write(origin)
