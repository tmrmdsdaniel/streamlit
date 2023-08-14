### 生物學案例 - 鳶尾花預測 ###
#%% (III) API 與 App 設計
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans 
from sklearn.preprocessing import scale, LabelEncoder # for scaling the data
import sklearn.metrics as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import warnings
warnings.filterwarnings("ignore")
np.random.seed(0)

# (I) 情境介紹與前處理 - 資料標準化
def set_vars(df):
    # 定義X變數 與目標變數y
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1] # define the target 

    X_scaled = scale(X) # scale the iris data

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y, X_scaled, y_encoded

# (II) 模型訓練與視覺化結果 - 模型建制與訓練
# shift + enter
def modeling(X_scaled):
    clustering = KMeans(n_clusters=3, random_state=42)
    clustering.fit(X_scaled) #fit the dataset
    return clustering
    
# (II) 模型訓練與視覺化結果 - 預測結果
# F9
def get_prediction(X, clustering):
    # 預測結果
    X['prediction'] = clustering.labels_
    return X
    
# (II) 模型訓練與視覺化結果 - 視覺化呈現: 預測
# shift + enter
def plot_reseult(X, y, y_encoded, predict=True):
    
    if predict:
        color_index = X.prediction
        title_ = "預測"
    else:
        color_index = y_encoded
        title_ = "實際"
    
    colors = np.array(["Red","Green","Blue"])
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True)

    fig.add_trace(
        go.Scatter(
            x=X["petal_length"], y=X["petal_width"],
            name = "petal",
            mode='markers',
            marker_color=colors[color_index]
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=X["sepal_length"], y=X["sepal_width"],
            mode='markers',
            name = "sepal",
            marker_color=colors[color_index]
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title_text=f"<b>{title_}</b>",
        title_x=0.1,
        width=900,
        height=600,
        font=dict(
            family="Courier New, monospace",
            size=15,
        ),
    )

    return fig

def backend(df):
    
    X, y, X_scaled, y_encoded = set_vars(df)
    clustering = modeling(X_scaled)
    X = get_prediction(X, clustering)
    
    return X, y, y_encoded

# 測試
if __name__ == "__main__":
    import plotly.io as pio
    pio.renderers.default='browser'
    
    df = pd.read_csv("iris.csv",encoding="utf=8-sig")
    X, y, y_encoded = backend(df)
    predict_fig = plot_reseult(X, y, y_encoded, predict=True)
    actual_fig = plot_reseult(X, y, y_encoded, predict=False)
    
#%%
if __name__ == "__main__":
    predict_fig.show()
    
#%%
if __name__ == "__main__":
    actual_fig.show()
