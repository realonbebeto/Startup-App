import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title='Startup Recommender', page_icon='random')
st.markdown(""" ## Startup Recommender """)
st.text('Get to know potential startups to invest in using our recommender')

# Read data
main_data = pd.read_csv('./recom_data/main_data.csv')
data = pd.read_csv('./recom_data/data.csv')

status_dict = {'acquired':0 ,'closed':1, 'ipo':2, 'operating':3}
status = st.sidebar.multiselect('Status', ['acquired', 'closed', 'ipo', 'operating'], default=['acquired'])

if len(status) == 0:
    st.error('No Status selected! Try and choose at least one option')

vote = st.sidebar.slider('Chance of Success', 0, 100, (75, 90), 1)
n_startups = st.sidebar.number_input('Number of Startups to Recommend', value=3)

k_nbrs = NearestNeighbors(algorithm='brute', n_jobs=-1, n_neighbors=10, metric='cosine')

k_nbrs.fit(data)

def get_idx(status:list, vote:tuple, n_startups:int):
    recom_idx = np.ndarray(shape=(1,))
    if len(status) == 1:
        startup_list = data.loc[data['status'] == status_dict[status[0]]]
    elif len(status) == 2:
        startup_list = data.loc[(data['status'] == status_dict[status[0]]) | (data['status'] == status_dict[status[1]])]
    else:
        startup_list = data.loc[(data['status'] == status_dict[status[0]]) | (data['status'] == status_dict[status[1]]) | (data['status'] == status_dict[status[2]])]

    rt = list(startup_list.loc[(startup_list['avg_vote'] >= vote[0]) & (startup_list['avg_vote'] <= vote[1])].index)
    if rt:
        if (n_startups > 0):
            picked = random.choice(rt)
            recom_idx = k_nbrs.kneighbors(X= data.iloc[picked].values.reshape(1,-1), n_neighbors=n_startups, return_distance=False)
            return recom_idx
        elif (n_startups < 0):
            st.warning('Invalid: Negative Number! Change the number of startups to recommend')
    

def get_recomd_startups(ridx:np.ndarray):
    result_recom = pd.DataFrame()

    if ridx.any():
        if len(ridx) > 0:
            result_recom = main_data.loc[ridx[0].tolist(), ['normalized_name', 'category_code', 'country_code', 'relationships', 'avg_vote']].copy()
            result_recom.columns = ['Name', 'Category', 'Country', 'Employees', 'Chance']
            result_recom.fillna('NA', inplace=True, axis=1)
            return result_recom

    elif ridx == None:
        st.warning('Try changing the status, chance of success and number of startups to recommend')

if status:
    if st.sidebar.button('Recommend'):
        r = pd.DataFrame()
        r = get_recomd_startups(get_idx(status = status, vote=vote, n_startups=n_startups))
        st.write(r)