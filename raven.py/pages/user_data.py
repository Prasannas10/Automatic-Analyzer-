
import streamlit as st
#from panda_profile_streport import data,df

import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport


from AutoClean import AutoClean
import numpy as np


st.subheader("Dataset")
data= st.file_uploader("Upload CSV",type=["csv"])

if data is not None:
    df = pd.read_csv(data)
   # x=data.data
    #y=data.target
    #feture_data=data.feature_names
    #df = pd.DataFrame(data=x, columns=feture_data)
    df.info()
    
    profile = ProfileReport(df,

                            title="Breast Canser Dataset",

            dataset={

            "description": "This profiling report was generated by Prsanna sakarkar",

            "copyright_holder": "Prasanna Sakarkar",

            "copyright_year": "2022",

            

        },)
    
    st_profile_report(profile)











   # if data is not None:
   #     df = pd.read_csv(data)
    # x=data.data
        #y=data.target
        #feture_data=data.feature_names
        #df = pd.DataFrame(data=x, columns=feture_data)
   #     df.info()
        
   #     profile = ProfileReport(df,

    #                            title="Breast Canser Dataset",

      #          dataset={

    #            "description": "This profiling report was generated by Prsanna sakarkar",
#
    #            "copyright_holder": "Prasanna Sakarkar",

    #            "copyright_year": "2022",

                

    #        },)
        
   #     st_profile_report(profile)






#file_details = {"filename":data.name, "filetype":data.type,
                          #  "filesize":data.size}
			