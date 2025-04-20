

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Import model dependencies to ensure pickle unpacks correctly
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

pipeline = pickle.load(open('champion_pipeline.pkl', 'rb'))(open('champion_pipeline.pkl', 'rb'))
vec = pickle.load(open('vectorizer.pkl', 'rb'))
lda = pickle.load(open('lda_model.pkl', 'rb'))

CLUSTER_OPTS = ['0', '1', '2']
DOW_OPTS = ['Thursday', 'Monday', 'Tuesday',
            'Saturday', 'Wednesday', 'Friday', 'Sunday']
TOPIC_PERF = {0: 87767.30685920578, 1: 29872.04375,
              2: 46434.066666666666, 3: 71323.46124031008, 4: 83114.3584131327}
SLOT_PERF = {('Sunday', 0): 86140.45497630331, ('Tuesday', 0): 82976.74545454545, ('Monday', 0)             : 82853.79841897234, ('Wednesday', 0): 73940.75357142858, ('Saturday', 0): 72698.1551724138}

st.title("YouTube View Predictor & Recommendations")

# Inputs
publishHour = 0

duration = st.sidebar.number_input("Duration (s)", min_value=0)
likes = st.sidebar.number_input("Like Count", min_value=0)
comments = st.sidebar.number_input("Comment Count", min_value=0)
subs = st.sidebar.number_input("Channel Subscribers", min_value=0)
days = st.sidebar.number_input("Days Since First Video", min_value=0)
durxsub = duration * subs
cluster = st.sidebar.selectbox("Cluster", CLUSTER_OPTS)
dow = st.sidebar.selectbox("Day Of Week", DOW_OPTS)

topic_probs = {f'topic_{i}': st.sidebar.slider(
    f'Topic {i} Prob', 0.0, 1.0, 0.0) for i in range(5)}


input_df = pd.DataFrame([{**{
    'durationSeconds': duration,
    'likeCount': likes,
    'commentCount': comments,
    'channelSubscriberCount': subs,
    'days_since_first': days,
    'dur_x_subs': durxsub,
    'cluster': cluster,
    'publishDayOfWeek': dow,
    'publishHour': publishHour,
    **topic_probs
}}])

# st.subheader("Input Features")
# st.write(input_df)

st.subheader("Input Features")
# Hide publishHour column
st.write(input_df.drop(columns=['publishHour'], errors='ignore'))

pred_log = pipeline.predict(input_df)[0]
pred = int(np.expm1(pred_log))
st.subheader("Predicted View Count")
st.write(f"{pred:,}")

st.subheader("Top Content Themes")
for t, v in TOPIC_PERF.items():
    st.write(f"Topic {t}: avg views {int(v):,}")

st.subheader("Recommended Publish Days")
for key, views in SLOT_PERF.items():
    # key may be a tuple (day, hour); extract only the day
    day = key[0] if isinstance(key, (tuple, list)) else key
    st.write(f"{day} → avg views {int(views):,}")


st.subheader("Topic Definitions")
feature_names = vec.get_feature_names_out()
n = 10
for idx, comp in enumerate(lda.components_):
    top_idxs = comp.argsort()[:-n-1:-1]
    terms = [feature_names[i] for i in top_idxs]
    st.write(f"Topic {idx}: {', '.join(terms)}")
