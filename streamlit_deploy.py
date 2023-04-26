#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st


# In[ ]:


import pickle
import pandas as pd
import numpy as np
import os
import sklearn



# In[ ]:


model_file_name="dev-model.pkl"
with open(model_file_name,'rb') as f:
    mod_LR=pickle.load(f)


# In[ ]:


upload_markdown = """Upload CSV Input Data File."""
run_markdown="""Run the Latest Model using the inputs"""


# In[ ]:


with st.container():
    st.title("Clustering Model Serving Web App")
    st.caption("Get Predictions from the latest model.")


# In[ ]:


st.markdown("---")

with st.container():
    
    upload_title=st.subheader('Upload Input Data File')
    upload_describe=st.caption(upload_markdown)
    
    
    input_file=st.file_uploader("Upload Excel Workbook")
    if input_file is not None:
        with open(input_file.name, mode='wb') as f:
            f.write(input_file.getbuffer())


# In[ ]:


st.markdown("---")

run_title=st.subheader("Run The Model")
run_describe=st.caption(run_markdown)



if input_file:
    run_requested=st.button("Make Predictions")
    
    if run_requested:
        st.info("Loading the new data for predictions.")
        
        
        data1=pd.read_csv(input_file.name)
        
        
        st.write(data1)
        
        st.info("Data Loaded. Staring Predictions.")

        
        
        data=data1.iloc[:,:-1].copy()

        y_kmeans = mod_LR.fit_predict(data)
        cluster = list(y_kmeans)
        data['cluster'] = cluster
        
        
        st.info("Predictions Complete")
        
        
        st.subheader("Predicted Result Of The Dataset")
        st.write("Cluster ",y_kmeans)


# In[ ]:




