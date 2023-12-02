from NEOkmeans import *
from kmeans import *
from main import *
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# df = pd.read_csv("/content/drive/MyDrive//Train.csv")

def label_preprocessing(df) :
    '''
    Arguments
    - df : dataframe without a target(ground-truth) column
    output
    - df_pca : pca 열을 원래 데이터프레임 df_org에 concat (나중에 클러스터링 결과, pca값, 원본 데이터 모두 합칠 예정)
    - arr_pca : 클러스터링에 사용할 pca 배열
    '''
    id_col = [col for col in df.columns if 'id' in col.lower()]   # ID 열이 존재하면 삭제
    df = df.drop(id_col, axis=1)

    df = df.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype == 'O' else x.fillna(x.median())) # 최빈값으로 대체

    df_org = copy.deepcopy(df)
    cat_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()
    label_encoder = LabelEncoder()
    df[cat_cols] = df[cat_cols].apply(lambda col: label_encoder.fit_transform(col))

    scaler = RobustScaler()   # 표준화
    df_scaled = scaler.fit_transform(df)    # array

    X_df = pd.DataFrame(df_scaled)      
    pca = PCA(n_components=2) # PCA
    df_pca = pca.fit_transform(X_df)
    df_pca = pd.DataFrame(df_pca, columns=['pca_x', 'pca_y'])
    arr_pca = np.array(df_pca)

    df_pca = pd.concat([df_org,df_pca],axis=1)

    return df_pca, arr_pca

# df_pca, arr_pca = label_preprocessing(df.iloc[:,:-1])
# k = 3

# NEOU = cluster(arr_pca, k)

# make NEOU dataFrame and concat
def create_U_dataframe(k, NEOU,df_pca): #np.array NEOU

    columns = [f'cluster{i}' for i in range(1, k + 1)]

    df_U = pd.DataFrame(NEOU, columns=columns)
    df_pca = pd.concat([df_pca, df_U], axis=1)

    # U_columns 기준으로 정렬
    final_df = df_pca.sort_values(by=columns,ascending=False)
    final_df = final_df.reset_index(drop=True)

    #cluster columns을 기준으로 그룹화
    grouped_final_df = final_df.groupby(columns).apply(lambda group: group.reset_index(drop=True))

    return grouped_final_df, columns

# grouped_final_df, U_columns = create_U_dataframe(k, NEOU, df_pca)

# 군집 시각화
def display_cluster_2d(grouped_final_df, k) :
    plt.figure(figsize=(10,8))
    markers = ['v','<','o', 's', '*', '^','D','x','p']
    colors = ['red', 'blue','green','yellow','purple','orange','gray','pink']
    target_key = (0.0,) * k

    for i, (group_key, group_data) in enumerate(grouped_final_df.groupby(level=list(range(k)))) :
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]

        if group_key == target_key :
            color = 'black'
            marker = '+'
            continue

        print(f'cluster group : {group_key}, color : {color}, marker : {marker}')
        plt.plot(group_data['pca_x'], group_data['pca_y'], color=color, marker=marker, linestyle='', label=f'Cluster {group_key}')

    plt.xlabel('pca_x')
    plt.ylabel('pca_y')
    plt.title(f'2D Visualization of Neo K-means Clustering')
    plt.legend()
    plt.show()

# display_cluster_2d(grouped_final_df, k)

# 특징 시각화
def feature_display(df, column, k) :
    
    if df[column].dtype == 'object' : # pie chart
        fig,axes = plt.subplots(1,k,figsize=(15, 5))
        fig.suptitle(f'Distribution of {column} in Clusters', y=1.02)

        all_categories = df[column].unique()
        colors_all = sns.color_palette('Set2', n_colors=len(all_categories))

        for cluster_num, ax in zip(range(1, k + 1), axes):
            cluster_data = df[df[f'cluster{cluster_num}'] == 1]
            value_counts = cluster_data[column].value_counts()

            colors = [colors_all[all_categories.tolist().index(category)] for category in value_counts.index] # 각 범주에 대해 동일한 색상 팔레트 선택

            index_list = value_counts.index.tolist()
            for i in range(5,len(value_counts)) :
                index_list[i] = ' '
            labels = pd.Index(index_list)   # 상위 5개의 레이블만 표시

            autopct_func = lambda p: '{:.1f}%'.format(p) if p > 5 else ''  # 5% 이상인 데이터만 표시 (겹치는거 방지)
            ax.pie(value_counts, labels = labels, autopct=autopct_func, startangle=90,colors=colors)
            ax.set_title(f'cluster{cluster_num}')

        plt.legend(title=column, labels=value_counts.index, bbox_to_anchor=(1.5, 1), loc='upper left')

    elif df[column].dtype in ['int', 'float'] : #swarm plot
        fig, ax = plt.subplots(figsize=(10, 6))
        cols = []
        for i in range(1,k+1) :
            cols.append(f"cluster{i}")

        df['cluster'] = np.argmax(df[cols].values, axis=1) + 1

        sns.violinplot(x="cluster", y=column, data=df, palette="Set2")
        ax.set_title(f'Violin Plot of {column} across Clusters')
        ax.set_xlabel('Cluster')
        ax.set_ylabel(column)

    plt.show()

# feature_display(grouped_final_df,'Var_1',k)