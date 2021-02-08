# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 15:24:06 2021

@author: Hammad
"""
# Webapp design
# 1. Selection of data
# 2. Viewing of data
# 3. Data Transformations
# 3. Descriptiv Statistics
# 4. Regressions Analysis




#%% Preamble
import pandas as pd
import requests
import json
import streamlit as st
import numpy as np
#import plotly
#import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import itertools
font = {'family': 'calibri',
        'size' : 16}
matplotlib.rc('font', **font)
#%% fetch data
#st.title("Data Query from empirica regio")
zugang = ('empirica.hammad.mufti', 'MYynlmbVZPSJ')
baseurl = 'https://api.empirica-regio.de/api/'

@st.cache(persist=True, suppress_st_warning=True)
def meta_data(baseurl, zugang):
    
    ## Extracting all variables
    # as Dataframe
    results = requests.get(baseurl + 'meta_variables/-1', auth = zugang)
    variables = pd.DataFrame(results.json())
    
    # as Dictionary
    variables_dict = {}
    for id in variables.id:
        variables_dict[id] = tuple(variables[variables.id == id].name_de)[0]
          
    results = requests.get(baseurl + '/meta_regions/-1', auth = zugang)
    all_regions = pd.DataFrame(results.json())
    #all_regions.to_excel('meta regions.xlsx')
    
    all_regions_dict = {}
    for r in all_regions.reg:
        all_regions_dict[r] = [all_regions[all_regions.reg == r].reg_name_de.values[0], all_regions[all_regions.reg == r].reg_typ.values[0]]
        
    return variables_dict, all_regions_dict

variables_dict, all_regions_dict = meta_data(baseurl, zugang)

#variable = st.sidebar.selectbox(
#        "Select a Variable",
#        sorted([value for key, value in variables_dict.items()])
#        )
#
region = st.sidebar.multiselect(
        "Select Regions",
        sorted([value for key, value in all_regions_dict.items()]),
        [['Deutschland', 0]])

#varid = [key for key, value in variables_dict.items() if value == variable][0]
regids = [key for key, value in all_regions_dict.items() if value in region]
reg_labels = {0: '', 10: '', 1000: ', (LK)', 1000000 : ', (Gem.)'}
varid = 2001
base = 2011
@st.cache(persist=True)
def load_data(regids, varid):
    dataframe = pd.DataFrame(list(range(2004,2021)), columns = ['year'])
    for regid in regids:
        results = requests.get(baseurl+'data/' + str(regid) + '/' + str(varid), auth = zugang)
        df = pd.DataFrame(results.json())[['year', 'value']]
        df.columns = ['year', str(regid)+ ',' + str(all_regions_dict[regid][1])]
        dataframe = dataframe.merge(df, how = 'outer', left_on = 'year', right_on = 'year')
    dataframe = dataframe.dropna().set_index('year')
    return dataframe
#
@st.cache(persist=True)
def update_base(data, base):
    base_values = [data.loc[base, i] for i in list(data.columns)]
    column_ids = [tuple(i.split(',')) for i in data.columns]
    column_names = [all_regions_dict[int(i)][0] + reg_labels[int(j)] for i,j in column_ids]
    Indexframe = pd.DataFrame()
    for i in range(len(base_values)):
        Indexframe[column_names[i]] = (data.iloc[:,i]/base_values[i])*100
    return Indexframe


def Abbildung_1(data):
    fig = plt.figure(figsize=(10.17,6.57))
    ax = fig.gca()
    marker = itertools.cycle((',', 's', 'o', '^', '.', 'x','+','p','*'))
    rgbs = [(0, 0, 0), (255, 102, 0), (253,174,107), (242, 203, 178), (191, 191, 191), (127, 127, 127), (140,140,140), (80,80,80)]
    colors = itertools.cycle([(x/255, y/255, z/255) for x,y,z in rgbs])
    #ax.plot(data)
    for i in data.columns:
        ax.plot(data[i], linewidth = 3, marker = next(marker), markersize = 10, color = next(colors))
    plt.xticks(data.index)
    if len(data) >= 10:
        plt.xticks(rotation = 'vertical')
    plt.yticks()
    ax.legend(bbox_to_anchor = (1.22, 0.5), labels = data.columns, loc = 10, frameon = False, borderaxespad=0.)
    plt.ylabel('Index: '+str(base) + ' = 100', fontweight='bold')
    plt.title('Abbildung 1: Entwicklung der Einwohner')
    ax.grid(axis = 'y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig
#
#
#
##years = st.sidebar.slider("Period", int(data.index[0]), int(data.index[len(data.index)-1]), (int(data.index[0]), int(data.index[len(data.index)-1])), 1)
#data = load_data(regids, varid)
##base = st.sidebar.selectbox(
##        "Select the base year",
##        list(data.index))
#
##data = data[data.index >= int(years[0])][data.index <= int(years[1])]
data = load_data(regids, varid)
base = st.sidebar.selectbox(
        "Select the base year",
        list(data.index))
data = update_base(data, base)
fig = Abbildung_1(data)
st.pyplot(fig)
st.write(data)
#    
