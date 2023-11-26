import main
import numpy as np
import pandas as pd
import streamlit as st
import os
from datetime import datetime
from streamlit_elements import elements, mui
from sklearn.preprocessing import MinMaxScaler # 이거 하니까 느려짐!!


# ## Header/Subheader
# st.header('NEO K-means')
# st.subheader('고객 세그멘테이션')

st.title('NEO k-means \n클러스터링🌸')

def save_uploaded_file(directory, file):
    '''파일 저장 함수'''
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(os.path.join(directory, file.name), 'wb') as f:
        f.write(file.read())
        

csv_file = st.sidebar.file_uploader('**💻 Load a CSV file**', type=['csv'])

print(csv_file)
if csv_file is not None:
    current_time = datetime.now()
    filename = current_time.isoformat().replace(':', '_') + '.csv'
    csv_file.name = filename
    save_uploaded_file('csv', csv_file)
    
    # csv를 보여주기 위해 pandas 데이터 프레임으로 만들어야한다.
    df = pd.read_csv('csv/'+filename)

    # 예시 데이터 5개만 보여주기
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
        st.write(neo_df)
        
        num_columns = neo_df.select_dtypes(include=np.number).columns.tolist() # 수치형 변수만 선택
        num_df = neo_df[num_columns]

        # Min-Max Scailing
        scaler = MinMaxScaler()
        scaler.fit(num_df)

        scaled_df = pd.DataFrame(scaler.transform(num_df))

        num_df = np.array(scaled_df) # array로 만들어야 돌아감

        U = main.cluster(num_df, k)
        U_df = pd.DataFrame(U)
        # U_df 칼럼명 : cluster_1, cluster_2 .. 이런식으로 바꾸기
        U_df.columns = ["cluster_{}".format(i) for i in range(1, k+1)]
        U_df.reset_index() # 인덱스 초기화
        # 클러스터 결과 dataframe
        st.write(U_df)


        ##### 클러스터 번호 선택하면 그 클러스터의 특징 보여줌
        st.subheader('클러스터별 특징')
        cluster_df = pd.concat([neo_df, U_df], axis=1, join='inner')
        # 원데이터(결측행 제거) + 클러스터 결과
        st.write(cluster_df)

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
            neo_df.columns,
        )
    if cluster != None:
        st.write(cluster_df.loc[cluster_df[cluster] == 1, [cluster, column]])
        st.write(cluster_df[column].dtypes)
    else:
        st.write("No selection")

    # # 선택한 칼럼이 범주형(object)인 경우,
    # if cluster_df[column].dtypes == object:
        


with elements("dashboard"):

    # You can create a draggable and resizable dashboard using
    # any element available in Streamlit Elements.

    from streamlit_elements import dashboard

    # First, build a default layout for every element you want to include in your dashboard

    layout = [
        # Parameters: element_identifier, x_pos, y_pos, width, height, [item properties...]
        dashboard.Item("first_item", 0, 0, 2, 2),
        dashboard.Item("second_item", 2, 0, 2, 2, isDraggable=False, moved=False),
        dashboard.Item("third_item", 0, 2, 1, 1, isResizable=False),
    ]

    # Next, create a dashboard layout using the 'with' syntax. It takes the layout
    # as first parameter, plus additional properties you can find in the GitHub links below.

    with dashboard.Grid(layout):
        mui.Paper("First item", key="first_item")
        mui.Paper("Second item (cannot drag)", key="second_item")
        mui.Paper("Third item (cannot resize)", key="third_item")

    # If you want to retrieve updated layout values as the user move or resize dashboard items,
    # you can pass a callback to the onLayoutChange event parameter.

    def handle_layout_change(updated_layout):
        # You can save the layout in a file, or do anything you want with it.
        # You can pass it back to dashboard.Grid() if you want to restore a saved layout.
        print(updated_layout)

    with dashboard.Grid(layout, onLayoutChange=handle_layout_change):
        mui.Paper("First item", key="first_item")
        mui.Paper("Second item (cannot drag)", key="second_item")
        mui.Paper("Third item (cannot resize)", key="third_item")