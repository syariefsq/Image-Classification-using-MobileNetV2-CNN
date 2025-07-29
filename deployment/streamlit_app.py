import streamlit as st

# set_page_config must be first Streamlit command
st.set_page_config(
    page_title='Road Clean/Littered Image Classifier', 
    initial_sidebar_state='expanded'
)

import eda
import prediction

page = st.sidebar.selectbox('Choose Page', ('EDA', 'Prediction'))

if page == 'EDA':
    eda.run()
else:
    prediction.run()