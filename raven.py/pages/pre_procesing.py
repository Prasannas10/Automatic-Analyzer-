from AutoClean import AutoClean
import pandas as pd
import numpy as np
import streamlit as st


st.subheader("Dataset")
data= st.file_uploader("Upload CSV",type=["csv"])

def pre_proceses(data):
    pipeline = AutoClean(data)
    procesed_data=pipeline.output
    return procesed_data


def convert_df(df):
   return df.to_csv().encode('utf-8')

if data is not None:
    df = pd.read_csv(data)
    procesed_data=pre_proceses(df)
    
    csv = convert_df(procesed_data)


    file_name = st.text_input('Download Procesed Data Namea as', 'proceses_data')
    st.write('The current file name is', file_name)




    st.download_button(
    label="Download  preprosed data ",
    data= csv,
     file_name=file_name,
    
 )


    



