import streamlit as st
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA
#streamlit run main.py

st.title("Automatic Analyzer🦊")

st.write("""
May the Force be with you.....
""")

#creating side bar for selecting

dataset_name=st.sidebar.selectbox("select data set",("Iris Dataset","Breast Canser Dataset","Wine Dataset"))

classifier_name=st.sidebar.selectbox("select classifier",("KNN","SVM","Random Forest"))

#loading selected clasifier by user

def get_dataset(dataset_name):
    if(dataset_name=="Iris Dataset"):
        data=datasets.load_iris()
    elif(dataset_name=="Breast Canser Dataset"):
        data=datasets.load_breast_cancer()
    else:
        data=datasets.load_wine()
    x=data.data
    y=data.target
    return x,y

#discribing dataset
x,y=get_dataset(dataset_name)
#st.write("shape of data set",x.shape)
#st.write("number of classes",len(np.unique(y)))

st.write("Hello amigos,you can use these tool to improve your learning in ML😻")

st.write("1.It's a playground to learn different algorithms🤩")
st.write("2.Upload your on dataset for preprocessing😉 ")
st.write("3.You can obtain a data report😛.")






#adding differet parameter according to clasifier
def add_parameter_ui(classifier_name):
    params={}
    if classifier_name=="KNN":
        k=st.sidebar.slider("k",1,15)
        params["k"]=k
    elif classifier_name=="SVM":
        c=st.sidebar.slider("c",0.01,10.0)
        params["c"]=c
    else:
        max_depth=st.sidebar.slider("max_depth",2,15)
        n_estimatours=st.sidebar.slider("n_estimatours",1,100)
        params["max_depth"]=max_depth
        params["n_estimatoure"]=n_estimatours
    

    return params

params=add_parameter_ui(classifier_name)

#fitting classification

def get_classifier(classifier_name,params):
     if classifier_name=="KNN":
        clf=KNeighborsClassifier(n_neighbors=params["k"])
     elif classifier_name=="SVM":
        clf=SVC(C=params["c"])
     else:
         clf=RandomForestClassifier(n_estimators=params["n_estimatoure"],max_depth=params["max_depth"],random_state=1234)
    
     return clf

clf=get_classifier(classifier_name,params)

#classification

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1234)

clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)

accuracy=accuracy_score(y_test,y_pred)
#st.write(f"classifire ={classifier_name}")
#st.write(f"accuracy ={accuracy}")

#plot
pca=PCA(2)
x_projected=pca.fit_transform(x)

x1=x_projected[:,0]
x2=x_projected[:,1]

fig=plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap="viridis")
plt.xlabel("principal componunt =1")
plt.ylabel("principal componunt =2")
plt.colorbar()

#pltshow
st.set_option('deprecation.showPyplotGlobalUse', False)  #for warning as st.pyplot has no aurgument
#st.pyplot()

