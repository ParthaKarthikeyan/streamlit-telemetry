# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from pycaret.clustering import *
import seaborn as sns

st.set_page_config(layout="wide")

def h(s,color):
   return f"background-color: {color};" if s == "Cluster 0" else f"background-color:#46A7D1" if s == "Cluster 1" else f"background-color: #0AA076"

def c(s):
   return f"background-color: yellow;"
#kmeans = load_model('C:\\Users\\hansi\\.spyder-py3\\Saved_Model')
r = pd.read_csv("After_Modelling_DF.csv")
r.set_index(['agentInfos_globalId'],inplace = True)
count_agn = r.shape[0]
s = r.style.applymap(h,subset= 'Cluster',color = '#F29E55')
# .set_table_styles([{'selector': 'th','props': [('font-weight','bold'),('font-style', ' italic'),('color','black'),('font-size','100')]}])\

st.title("Clustered output")
st.dataframe(s)
# img = mpimg.imread('C:\\Users\\hansi\\.spyder-py3\\kmeans.png')
feature_selected  = st.selectbox("Select feature", options=[c for c in r.columns[3:-1]])

col1,mid,col2 = st.beta_columns([5,15,5])

with mid:
    st.write(px.scatter(r, x="duration", y=feature_selected, color_discrete_map={'Cluster 0' : '#F29E55','Cluster 1' :'#46A7D1','Cluster 2': '#0AA076'},color="Cluster", size_max=60))
cm = sns.light_palette("orange", as_cmap=True)
r['variation'] = 0


# Calculating distance between the 2 points
j = -1
n = len(r.columns[2:-1])
st.sidebar.title("Median of each feature")
for c in r.columns[2:n]:
    r['dist'] = np.sqrt(r[c]**2 + r['duration']**2)
    s = r.loc[r['dist'].idxmin()]
    r.drop(columns = ['dist'],inplace = True)
    Cluster = s.Cluster
    medi = r.groupby(["Cluster"])[[c]].median().reset_index()
    i = medi[medi.Cluster == Cluster].index.tolist()
    median_val = medi[c][i[0]]
    st.sidebar.write(c,round(median_val,3))
    r[c] = r[c] - median_val
    r['variation'] = r['variation']+r[c]
st.sidebar.write(" ")
st.sidebar.text("The Total number of agents is: " + str(count_agn))
s = r.style.background_gradient(cmap=cm)
# .set_table_styles([{'selector': 'th','props': [('font-weight','bold'),('font-style', ' italic'),('color','black'),('font-size','100')]}])
st.dataframe(s)
st.write("")
st.write("")
st.title("5 least performing agents:")
st.write("")

least_per = r.nlargest(5, 'variation')
least_per.drop(columns = ["duration","variation","Cluster"],inplace = True)
df =least_per.copy()
least_per['Worst Performance'] = least_per.idxmax(axis=1)
least_per['Time Consumed (Seconds)'] = df.max(axis=1)
least_per = least_per.iloc[:,-2:]
s = least_per.style.set_table_styles([{'selector': 'th','props': [('font-weight','bold'),('font-style', ' italic'),('color','black'),('font-size','100')]}])
# st.table(s)

col1,mid1,col2 = st.beta_columns([5,10,5])

with mid1:
    st.table(s)
