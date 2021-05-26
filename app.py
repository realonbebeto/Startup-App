"""
@author: nyamwamu
"""

import streamlit as st
import streamlit_analytics
import pandas as pd
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors


def main():
    # Setting the header of the page
    st.title("Startup Recommender")
    

    menu = ['Home', 'Recommend', 'About']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        st.subheader('Home')
        st.write('A simple startup recommender implementing the KNN algorithm')
        st.write('Get to know potential startups to invest in using our recommender')
    
    elif choice == 'Recommend':
        st.subheader('Recommend')
        status, vote, n_startups = take_inputs()
        main_data, data = load_data()
        # Model Instance and training
        k_nbrs = NearestNeighbors(algorithm='brute', n_jobs=-1, n_neighbors=10, metric='cosine')
        k_nbrs.fit(data)

        if status:
            if st.button('Recommend'):
                r = pd.DataFrame()
                r = get_recomd_startups(main_data, get_idx(data, k_nbrs, status, vote, n_startups))
                st.write(r)



    elif choice == 'About':
        st.subheader('About')
        st.write('This is simple application focuses on giving recommendation of startups by implementing KNN algorithm on data that has previously been processed by classification using 4 models')
        st.write('Application is developed by Bebeto Nyamwamu to recommend startups to investors')
        st.write(" For the report and more [read](https://drive.google.com/file/d/1OtYSzBA_Sjrw4H_EGeDsRD9yO7aw3IpP/view?usp=sharing)")
        st.warning('Warning! `Nuff said: To talk to your Data Scientist/Machine Learning Engineer at :email: [email](mailto:nberbetto@gmail.com) or view his works on [Github](https://github.com/realonbebeto)')

def load_data():
    # Read data
    main_data = pd.read_csv('./recom_data/main_data.csv')
    data = pd.read_csv('./recom_data/data.csv')

    return main_data, data

def take_inputs():
    status = st.multiselect('Status', ['acquired', 'closed', 'ipo', 'operating'], default=['acquired'])

    if len(status) == 0:
        st.error('No Status selected! Try and choose at least one option')

    vote = st.slider('Chance of Success', 0, 100, (75, 90), 1)
    n_startups = st.number_input('Number of Startups to Recommend', value=3)

    return status, vote, n_startups

def get_idx(data, model, status:list, vote:tuple, n_startups:int):
    status_dict = {'acquired':0 ,'closed':1, 'ipo':2, 'operating':3}
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
            recom_idx = model.kneighbors(X= data.iloc[picked].values.reshape(1,-1), n_neighbors=n_startups, return_distance=False)
            return recom_idx
        elif (n_startups < 0):
            st.warning('Invalid: Negative Number! Change the number of startups to recommend')
    

def get_recomd_startups(data, ridx:np.ndarray):
    result_recom = pd.DataFrame()

    if ridx.any():
        if len(ridx) > 0:
            result_recom = data.loc[ridx[0].tolist(), ['normalized_name', 'category_code', 'country_code', 'relationships', 'avg_vote']].copy()
            result_recom.columns = ['Name', 'Category', 'Country', 'Employees', 'Chance']
            result_recom.fillna('NA', inplace=True, axis=1)
            return result_recom

    elif ridx == None:
        st.warning('Try changing the status, chance of success and number of startups to recommend')

if __name__ == '__main__':
    with streamlit_analytics.track(unsafe_password='1'):
        # Setting the title bar page name
        st.set_page_config(page_title='Startup Recommender', page_icon='random', layout='centered', initial_sidebar_state='auto')
        try: 
            main()
        except: 
            st.error('Oops! Something went wrong...Please check your input.\nIf you think there is a bug, please open up an [issue](https://github.com/realonbebeto/Startup-App/issues) and help us improve. ')
            raise