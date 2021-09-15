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
import base64
from io import BytesIO
import math
#import plotly
#import plotly.express as px
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
#import locale
import matplotlib.ticker as mtick

#locale.setlocale(locale.LC_ALL, 'de_de')
#locale.setlocale(locale.LC_NUMERIC, "german")
#locale.setlocale(locale.LC_ALL, 'de_DE.utf8')
font = {'family': 'calibri',
        'size' : 16}
mpl.rc('font', **font)
plt.rc('axes', axisbelow=True)
#%% fetch data
#st.title("Data Query from empirica regio")
zugang = ('regio.hammad.mufti', 'U2cqRHfPpVgF')
baseurl = 'https://api.empirica-regio.de/api/'

@st.cache
def meta_data():
    variables = pd.read_excel('meta variables.xlsx')
    variables['name'] = variables.agg('{0[id]},{0[name_de]}'.format, axis = 1)
    variables = variables.set_index('id')

    all_regions = pd.read_excel('meta regions.xlsx')
    all_regions['name'] = all_regions.agg('{0[reg]},{0[reg_name_de]},{0[reg_typ]}'.format, axis=1)
    all_regions_dict = all_regions.set_index('reg')['name'].transpose().to_dict()
    all_regions = all_regions.set_index('reg')
    all_regions_list = sorted([value for key, value in all_regions_dict.items()], 
                               key=lambda x:x.split(',')[2])            
    return all_regions_dict, variables, all_regions, all_regions_list

all_regions_dict, variables, all_regions, all_regions_list = meta_data()

#%% Choosing Inputs
cat = st.sidebar.selectbox(
        "Select Graph Type",
        ["Abbildung 1 & 2","Abbildung 4", "Abbildung 7", "Abbildung 8", 
         "Abbildung 9", "Abbildung 10","Abbildung 12","Abbildung 18",
         "Abbildung 19&20", "Abbildung 28&29", "Abbildung 33", "Abbildung 34"])

if cat == "Abbildung 1 & 2":
    data_vars_dict = variables[variables.endpoint == 'data']['name'].transpose().to_dict()
    data_vars_list = [value for key, value in data_vars_dict.items()]    
    variable = st.sidebar.selectbox(
            "Select a Variable",
            sorted(data_vars_list)
            ) 
    variable = [variable]
    region = st.sidebar.multiselect(
            "Select Regions",
            all_regions_list,
            ["0,Deutschland,0"])
    
if cat == "Abbildung 4":
    kreisgemeind_dict = all_regions[(all_regions.reg_typ == 1000) | (all_regions.reg_typ == 1000000)]['name'].transpose().to_dict() 
    variable = ["1101,Einpendler", "1102,Auspendler"]
    
    region = st.sidebar.selectbox(
            "Select Region",
            sorted([value for key, value in kreisgemeind_dict.items()])
            )
    region = [region]
    
if cat == "Abbildung 7":
    ids = list(range(202131, 202136))
    variable = list(variables.loc[ids, 'name'])
    region = st.sidebar.multiselect(
            "Select Regions",
            all_regions_list,
            ["0,Deutschland,0"])

if cat == "Abbildung 8":
    ids = [202831, 202833]
    variable = list(variables.loc[ids, 'name'])
    region = st.sidebar.multiselect(
            "Select Regions",
            all_regions_list,
            ["0,Deutschland,0"])

if cat == "Abbildung 9":
    ids = [202002]
    variable = list(variables.loc[ids, 'name'])
    region = st.sidebar.multiselect(
            "Select Regions",
            all_regions_list,
            ["0,Deutschland,0"])

if cat == "Abbildung 10":
    ids = [1401, 201401]
    variable = list(variables.loc[ids, 'name'])
    region1 = st.sidebar.selectbox(
            "Focus area",
            all_regions_list
            )
    region2 = st.sidebar.multiselect(
            "Comparison areas",
            all_regions_list,
            ['5,Nordrhein-Westfalen,10'])
    region = [region1] + region2
    
if cat == "Abbildung 12": 
    ids = [2011, 2012, 10221, 10211]
    variable = list(variables.loc[ids, 'name'])
    
    region = st.sidebar.selectbox(
            "Select Region",
            all_regions_list
            )
    region = [region]
    
if cat == "Abbildung 18":
    ids = [3521, 3522]
    variable = list(variables.loc[ids, 'name'])
    region = st.sidebar.selectbox(
            "Select Region",
            all_regions_list
            )
    region = [region]
    
if cat == "Abbildung 19&20":
    ids = ['103521,EZFH', '103522,MFH']
    variable = st.sidebar.selectbox(
            "Fertiggestellte Neubauwohnungen in ",
            ids)
    variable = [variable]
    region = st.sidebar.multiselect(
            "Select Regions",
            all_regions_list,
            ["0,Deutschland,0"])

if cat == "Abbildung 28&29":
    ids = [12521, 12021]
    data_vars_list = [variables.loc[i,'name'] for i in ids]    
    variable = st.sidebar.selectbox(
            "Select a Variable",
            sorted(data_vars_list)
            )
    variable = [variable]
    region = st.sidebar.multiselect(
            "Select Regions",
            all_regions_list,
            ["0,Deutschland,0"])
    
if cat == "Abbildung 33":
    ids = [5100, 5110, 5009]
    variable = list(variables.loc[ids, 'name'])
    region = st.sidebar.selectbox(
            "Select Region",
            all_regions_list
            )
    region = [region]
    
if cat == "Abbildung 34":
    ids = [5100, 5110, 5009]
    variable = list(variables.loc[ids, 'name'])
    region = st.sidebar.multiselect(
            "Select Regions",
            all_regions_list,
            ["0,Deutschland,0"])
    
varids = [int(variable[i].split(',')[0]) for i in list(range(len(variable)))]
#regids = [key for key, value in all_regions_dict.items() if value in region]
regids = [int(region[i].split(',')[0]) for i in list(range(len(region)))]
reg_labels = {0: '', 10: '', 1000: ' (LK)', 1000000 : ' (Gem.)'}
#varid = 2001
#base = 2011

   
            
#%% Defining Functions

def roundup(x):
    return int(math.ceil(x / 1000.0)) * 1000

def numb_out2(n):
    parts = str(abs(n)).replace(".",",").partition(",")
    digits = list(parts[0])
    pos = [p for p in range(len(digits)%3,len(digits),3) if p>0][::-1]
    for p in pos: digits[p:p] = "."
    return (n<0)*"-"+"".join(digits)+"".join(parts[1:])

@st.cache
def load_data(regids, varids):
    dataframe = pd.DataFrame(list(range(2004,2021)), columns = ['year'])
    for regid in regids:
        for varid in varids:
            results = requests.get(baseurl+'data/' + str(regid) + '/' + str(varid), auth = zugang)
            df = pd.DataFrame(results.json())[['year', 'value']]
            df.columns = ['year', all_regions.loc[regid, 'name'] + ' ,' + str(varid)]
            dataframe = dataframe.merge(df, how = 'inner', left_on = 'year', right_on = 'year')
    dataframe.columns = ['year'] + [all_regions.loc[i, 'name'] +',' + str(j) for i in regids for j in varids] 
    dataframe = dataframe.set_index('year')
    return dataframe

@st.cache
def update_base(data, base):
    base_values = data.loc[base, :]
    column_ids = map(lambda x:tuple(x.split(',')[1:3]), data.columns)
    column_names = [j + reg_labels[int(k)] for j,k in column_ids]
    Indexframe = pd.DataFrame()
    for i in range(len(base_values)):
        Indexframe[column_names[i]] = (data.iloc[:,i]/base_values[i])*100
    return Indexframe


def Abbildung_1(data):
    fig = plt.figure(figsize=(11,6.57))
    ax = fig.gca()
    marker = itertools.cycle((',', 's', 'o', '^', '.', 'x','+','p','*'))
    rgbs = [(0, 0, 0), (255, 102, 0), (253,174,107), (191, 191, 191), (127, 127, 127), (140,140,140), (80,80,80)]
    colors = itertools.cycle([(x/255, y/255, z/255) for x,y,z in rgbs])
    #ax.plot(data)
    for i in data.columns:
        ax.plot(data[i], linewidth = 3, marker = next(marker), markersize = 8, color = next(colors))
    plt.xticks(data.index)
    plt.tick_params(axis='y', length = 0)
    if len(data) >= 12:
        plt.xticks(rotation = 'vertical')
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:n}'))
    ax.legend(bbox_to_anchor = (1, 0.5), labels = data.columns, loc = 6, frameon = False, borderaxespad=0.)
    plt.ylabel('Index: '+str(base) + ' = 100', fontweight='bold')
    plt.title('Entwicklung der '+ variable[0].split(',')[1], pad = 10)
    ax.grid(axis = 'y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    #plt.tight_layout()
    return fig

def Abbildung_4(data):
    data.iloc[:,1] = data.iloc[:,1]*(-1)
    data.columns = ["Einpendler", "Auspendler"]
    data['Saldo'] = data.iloc[:,0] + data.iloc[:,1]
    fig = plt.figure(figsize=(10.17,6.57))
    ax = plt.subplot(111)
    ax.bar(data.index, list(data["Einpendler"]), color = (255/255, 102/255, 0/255), width = 0.5, label = "Einpendler")
    ax.bar(data.index, list(data["Auspendler"]), color = (253/255, 174/255, 107/255), width = 0.5, label = "Auspendler")
    ax.plot(data.index, list(data["Saldo"]), color = (0,0,0), label = "Saldo")
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:n}'))
    yticks = [int(i) for i in list(ax.get_yticks())]
    yticks = map(numb_out2, yticks)
    ax.set_yticklabels(list(yticks))
    plt.tick_params(axis='y', length = 0)
    plt.xticks(data.index)
    if len(data) >= 12:
        plt.xticks(rotation = 'vertical')
    plt.ylabel("Anzahl", fontweight='bold')
    h, l = ax.get_legend_handles_labels()
    ax.legend([h[0],h[1],h[2]], [l[0], l[1], l[2]], bbox_to_anchor = (1, 0.5), loc = 6, frameon = False, borderaxespad=0.)
    plt.title('Entwicklung der Ein- und Auspendler in ' + region[0].split(',')[1], pad = 20)
    ax.grid(axis = 'y',zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    return fig


def Abbildung_7(data, year):
    data = data.loc[year,:].to_frame()
    #data.index = [list(variables.loc[ids,'name_de'])]
    index_tuples = map(lambda x : (x.split(',')[1], reg_labels[int(x.split(',')[2])], x.split(',')[3]), data.index)
    index_tuples = map(lambda x : (''.join(x[0:2]), int(x[2])), list(index_tuples))
    Index = pd.MultiIndex.from_tuples(list(index_tuples), names = ('region', 'variable'))
    data.index = Index
    data = data.swaplevel(0, 1, axis = 0).unstack()
    data.columns = map(lambda x:x[1], data.columns)
    #fig = plt.figure(figsize=(10.17,6.57))
    #ax = plt.subplot(111)
    rgbs = [(0, 0, 0), (255, 102, 0), (253,174,107), (191, 191, 191), (127, 127, 127), (140,140,140), (80,80,80)]
    colors = [(x/255, y/255, z/255) for x,y,z in rgbs]
    fig = data.plot(kind = 'bar', figsize = (10.17,6.57), color = colors)
    #fig.plot(data.iloc[:,0].to_frame(), linewidth = 0.2, color = 'b')
    #rgbs = [(0, 0, 0), (255, 102, 0), (253,174,107), (191, 191, 191), (127, 127, 127), (140,140,140), (80,80,80)]
    #colors = itertools.cycle([(x/255, y/255, z/255) for x,y,z in rgbs])
#    if len(data.columns) > 1:
#        for col in data.columns[1::]:
#            fig.plot(data[col], linewidth = 3, color = next(colors), label = col)
    #fig.tick_params(axis='x', which='minor', bottom=False)        
    fig.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:n}'))
    fig.tick_params(axis='y', length = 0)
    fig.yaxis.set_major_formatter(mtick.PercentFormatter())
    fig.set_xticklabels(list(map(lambda x:x.split(',')[1][:-34], variable)), fontsize = 16,rotation = 25, ha="right")
    
    fig.set_ylabel("Anteil", fontweight='bold')
    fig.set_xlabel('')
    fig.legend(bbox_to_anchor = (0.5, -0.3), loc = 9, frameon = False, borderaxespad=0., ncol=len(data.columns))
    fig.set_title('Einwohner nach Alter ('+str(year)+')'  , fontweight = 'bold', pad = 20)
    fig.grid(axis = 'y',zorder=0)
    fig.spines['top'].set_visible(False)
    fig.spines['right'].set_visible(False)
    fig.spines['left'].set_visible(False)
    #plt.tight_layout()
    return fig.figure, data

def Abbildung_8(data, year):
    data = data.loc[year,:].to_frame()
    index_tuples = map(lambda x : (x.split(',')[1], reg_labels[int(x.split(',')[2])], x.split(',')[3]), data.index)
    index_tuples = map(lambda x : (''.join(x[0:2]), int(x[2])), list(index_tuples))
    Index = pd.MultiIndex.from_tuples(list(index_tuples), names = ('region', 'variable'))
    data.index = Index
    data = data.swaplevel(0, 1, axis = 0).unstack()
    data.columns = map(lambda x:x[1], data.columns)
    data = data.transpose()
    data.columns = map(lambda x:x.split(',')[1][:-22], variable)
    fig = data.plot.bar(stacked = True, width = 0.5, figsize = (10.17,6.57), color = [(255/255, 102/255, 0/255),(191/255, 191/255, 191/255)])
    fig.yaxis.set_major_formatter(mtick.PercentFormatter())
    fig.set_ylabel("Anteil", fontweight='bold')
    fig.legend(bbox_to_anchor = (1, 0.5), loc = 6, frameon = False, borderaxespad=0.)
    fig.set_title('Anteil Househalte nach Haushaltsgröße ('+str(year)+')'  ,pad = 20)
    fig.set_xticklabels(data.index,rotation = 0)
    if len(data.index) >= 4:
        fig.set_xticklabels(data.index, rotation = 45, ha="right", rotation_mode="anchor")
    for rect in fig.patches:
    # Find where everything is located
        height = rect.get_height()
        width = rect.get_width()
        x = rect.get_x()
        y = rect.get_y()
        # The height of the bar is the data value and can be used as the label
        label_text = f'{height:.0f}%'  # f'{height:.2f}' to format decimal values
        # ax.text(x, y, text)
        label_x = x + width / 2
        label_y = y + height / 2
        # plot only when height is greater than specified value
        fig.text(label_x, label_y, label_text, ha='center', va='center', fontsize=16)
    fig.grid(axis = 'y',zorder=0)
    fig.spines['top'].set_visible(False)
    fig.spines['right'].set_visible(False)
    fig.spines['left'].set_visible(False)
    #plt.tight_layout
    return fig.figure, data

def Abbildung_9(data, years):
    data = data.loc[years,:].transpose().sort_values(by = years[1])
    data.index = map(lambda x : x.split(',')[1] + reg_labels[int(x.split(',')[2])], data.index)
    fig = data.plot.barh(figsize = (10.17,6.57), color = [(255/255, 102/255, 0/255),(191/255, 191/255, 191/255)], width = 0.6)
    fig.xaxis.set_major_formatter(mtick.PercentFormatter())
    xticks = [int(i) for i in list(fig.get_xticks())]
    xticks = map(numb_out2, xticks)
    xticks = [i+'%' for i in xticks]
    fig.set_xticklabels(xticks)
    fig.set_xlabel("Anteil", fontweight='bold')
    h, l = fig.get_legend_handles_labels()
    fig.legend([h[1],h[0]],[l[1],l[0]],bbox_to_anchor = (1, 0.5), loc = 6, frameon = False, borderaxespad=0.)
    fig.set_title('Ausländeranteil in '+region[0].split(',')[1]+' und in der Region ('+str(years[0])+' und '+str(years[1])+')', pad = 20)
    fig.grid(axis = 'x',zorder=0)
    fig.spines['top'].set_visible(False)
    fig.spines['right'].set_visible(False)
    fig.spines['left'].set_visible(False)
    
    for rect in fig.patches:
    # Find where everything is located
        height = rect.get_height()
        width = rect.get_width()
        x = rect.get_x()
        y = rect.get_y()
        # The height of the bar is the data value and can be used as the label
        label_text = f'{width:.0f}%'  # f'{height:.2f}' to format decimal values
        # ax.text(x, y, text)
        label_x = x + width + 0.8
        label_y = y + height / 2
        # plot only when height is greater than specified value
        fig.text(label_x, label_y, label_text, ha='center', va='center', fontsize=16)
    return fig.figure, data

def Abbildung_10(data):
    data1 = data.reset_index()
    data1.year = data1.year.apply(str)
    rgbs = [(255, 102, 0), (253,174,107), (127, 127, 127), (140,140,140), (80,80,80)]
    colors = itertools.cycle([(x/255, y/255, z/255) for x,y,z in rgbs])
    fig, ax1 = plt.subplots(figsize = (10.17,6.57))
    ax2 = ax1.twinx()
    data1[data1.columns[1]].plot(kind='bar', color=(191/255, 191/255, 191/255), ax=ax1, label = 'Arbeitslose ' + data1.columns[1].split(',')[1] +' absolut')
    data1[data1.columns[2]].plot(kind='line', ax=ax2, linewidth = 3, color=(0,0,0), label = data1.columns[1].split(',')[1] + ' (Index)')
    if len(data1.columns) > 3:
        for col in data1.columns[3:]:
            data1[col].plot(kind='line', color = next(colors), ax=ax2, linewidth = 3, label = col.split(',')[1] + ' (Index)')
    l = len(str(int(data1[data1.columns[1]].max())))
    ax1.set_ylim(0, roundup(data1[data1.columns[1]].max()))
    dist = ax1.get_yticks()[1] - ax1.get_yticks()[0]
    ax1.set_ylim(0, ax1.get_yticks()[-2] + dist)
    #ax1.set_ybound(0, round(data1[data1.columns[1]].max(), -(l-2)))
    #ax1.locator_params(axis='y', nbins=6)
    if len(ax1.get_yticks()) == 6:
        ax2.set_ylim(0,10)
    if len(ax1.get_yticks()) == 7:
        ax2.set_ylim(0,12)
    if len(ax1.get_yticks()) == 8:
        ax2.set_ylim(0,14)
    if len(ax1.get_yticks()) == 9:
        ax2.set_ylim(0,16)
    if len(ax1.get_yticks()) == 10:
        ax2.set_ylim(0,18)   
        
    ax2.set_xlim(-1, len(data1.year))
    
    ax1.set_xticklabels(list(data1.year))
    ax2.tick_params(axis='y', which='both',length=0)
    ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.7n}'))
    yticks = [int(i) for i in list(ax1.get_yticks())]
    yticks = map(numb_out2, yticks)
    ax1.set_yticklabels(list(yticks))
    
    ax1.legend(bbox_to_anchor = (0.5, -0.2), loc = 10, frameon = False, borderaxespad=0., ncol=len(data1.columns)-1)
    ax2.legend(bbox_to_anchor = (0.5, -0.3), loc = 10, frameon = False, borderaxespad=0., ncol=len(data1.columns)-1)
    ax1.set_title('Entwicklung der Arbeitslosigkeit in ' + data1.columns[1].split(',')[1]+ ' und in der Region ('+str(years[0])+'-'+str(years[1])+')' , fontweight = 'bold', pad = 20)
    ax1.grid(axis ='y')
    ax1.set_ylabel('Anzahl', fontweight = 'bold')
    ylabel = variables.loc[201401,'name_de']
    ax2.set_ylabel(ylabel[:32]+'\n'+ylabel[32:], fontweight = 'bold')
    #ax2.grid(None)
    #ax2.set_yticks(np.linspace(ax2.get_yticks()[1], ax2.get_yticks()[-1], len(ax1.get_yticks())))
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    yticks = [int(i) for i in list(ax2.get_yticks())]
    yticks = map(numb_out2, yticks)
    yticks = [i+'%' for i in list(yticks)]
    ax2.set_yticklabels(list(yticks))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    #plt.tight_layout
    return fig, data1
    
def Abbildung_12(data):
    data1 = pd.DataFrame()
    data1["natürlicher Saldo"] = data.iloc[:,0] - data.iloc[:,1]
    data1["Wanderungssaldo"] = data.iloc[:,2] - data.iloc[:,3]
    data1["Gesamtsaldo"] = data1["natürlicher Saldo"] + data1["Wanderungssaldo"]
    data1 = data1.reset_index()
    data1.year = data1.year.apply(str)
    
    fig, ax = plt.subplots(figsize = (10.17,6.57)) 
    ax = data1[["year","Gesamtsaldo"]].plot(x = 'year',y = 'Gesamtsaldo', ax=ax, marker='^', markersize=8, color=(0,0,0))
    data1[['year', 'natürlicher Saldo', 'Wanderungssaldo']].plot(x='year', y = ['natürlicher Saldo', 'Wanderungssaldo'], 
         kind='bar', ax=ax, stacked = True, color = [(255/255, 102/255, 0/255),(191/255, 191/255, 191/255)])
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:n}'))
    yticks = [int(i) for i in list(ax.get_yticks())]
    yticks = map(numb_out2, yticks)
    ax.set_yticklabels(list(yticks))
    ax.legend(bbox_to_anchor = (0.5, -0.2), loc = 10, frameon = False, borderaxespad=0., ncol=len(data.columns))
    ax.set_title('Komponenten der Einwohnerentwicklung in ' + region[0].split(',')[1] , pad = 20)
    ax.set_xlabel('')
    ax.set_ylabel('Anzahl', fontweight = 'bold')
    ax.grid(axis = 'y',zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    return fig, data1
    
def Abbildung_18(data):
    data.columns = ['EZFH', 'MFH']
    fig, ax = plt.subplots(figsize = (10.17,6.57))
    data.plot(kind='bar', ax=ax, stacked = True, color = [(255/255, 102/255, 0/255),(191/255, 191/255, 191/255)])
    for rect in ax.patches:
    # Find where everything is located
        height = rect.get_height()
        width = rect.get_width()
        x = rect.get_x()
        y = rect.get_y()
        # The height of the bar is the data value and can be used as the label
        label_text = f'{height:.0f}'  # f'{height:.2f}' to format decimal values
        # ax.text(x, y, text)
        label_x = x + width / 2
        label_y = y + height / 2
        # plot only when height is greater than specified value
        ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=12)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:n}'))
    yticks = [int(i) for i in list(ax.get_yticks())]
    yticks = map(numb_out2, yticks)
    ax.set_yticklabels(list(yticks))
    ax.legend(bbox_to_anchor = (0.5, -0.2), loc = 10, frameon = False, borderaxespad=0., ncol=len(data.columns))
    ax.set_title('Fertiggestellte Neubauwohnungen in ' + region[0].split(',')[1], pad = 20)
    ax.set_xlabel('')
    ax.set_ylabel('Anzahl', fontweight = 'bold')
    ax.grid(axis = 'y',zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    return fig, data

def Abbildung_1920(data):
    #data.iloc[[-1, -5, -6, -10]]
    #data.columns = [i.split(',')[1] for i in data.columns]
    data = data.transpose()
    data1 = pd.DataFrame()
    data1['{}-{}'.format(data.columns[-10], data.columns[-6])] = data.iloc[:,-10:-5].mean(axis = 1)
    data1['{}-{}'.format(data.columns[-5], data.columns[-1])] = data.iloc[:,-5:].mean(axis = 1)
    data1 = data1.transpose()
    data1.columns = [i.split(',')[1] for i in data1.columns]
    rgbs = [(0, 0, 0), (255, 102, 0), (253,174,107), (191, 191, 191), (127, 127, 127), (140,140,140), (80,80,80)]
    colors = [(x/255, y/255, z/255) for x,y,z in rgbs]
    fig = data1.plot.bar(figsize=(10.17,6.57), rot = 0, color = colors)
    fig.legend(bbox_to_anchor = (1, 0.5), loc = 6, frameon = False, borderaxespad=0.)
    fig.set_xlabel('')
    fig.set_ylabel('Fertiggestellte Neubauwohnungen\nje 1000 Einwohner')
    fig.set_title('Fertiggestellte Neubauwohnungen ({}) in {} und der Region \nje 1.000 Einwohner({}-{})'.format(
            variable[0].split(',')[1], all_regions.loc[regids[-1],'reg_name_de'], data.columns[-10],data.columns[-1]), pad = 20)
    fig.grid(axis = 'y',zorder=0)
    fig.spines['top'].set_visible(False)
    fig.spines['right'].set_visible(False)
    fig.spines['left'].set_visible(False)
    return fig.figure, data1

def Abbildung_2829(data):
    data1 = data.copy()
    #data1.columns = [i.split(',')[1] for i in data.columns]
    data1 = data1.transpose()
    rgbs = [(0, 0, 0), (255, 102, 0), (253,174,107), (191, 191, 191), (127, 127, 127), (140,140,140), (80,80,80)]
    colors = [(x/255, y/255, z/255) for x,y,z in rgbs]
    data1.index = [i.split(',')[1] for i in data1.index]
    fig = data1.plot.bar(figsize=(10.17,6.57), rot = 0, color = colors)
    data1 = data1.transpose()
    fig.legend(bbox_to_anchor = (1, 0.5), loc = 6, frameon = False, borderaxespad=0.)
    fig.set_xlabel('')
    if varids[0] == 12021:
        fig.set_ylabel('Mietpreis in Euro pro m^2 \nim Bestand', fontweight = 'bold')
        fig.set_title('Standardmieten (Median, nettokalt) für Mietwohnungen (gebraucht) \nin {} und der Region'.format(all_regions.loc[regids[0],'reg_name_de']))
    if varids[0] == 12521:
        fig.set_ylabel('Preis für ein und Ein- und Zweifamilienhäusern \nim Bestand', fontweight = 'bold')
        fig.set_title('Standard Preise für Ein/Zweifamilienhäuser (gebraucht) \nin {} und der Region'.format(all_regions.loc[regids[0],'reg_name_de']))
    fig.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.7n}'))
    #fig.ticklabel_format(useOffset=False)
    fig.grid(axis = 'y',zorder=0)
    fig.spines['top'].set_visible(False)
    fig.spines['right'].set_visible(False)
    fig.spines['left'].set_visible(False)
    return fig.figure, data1

def Abbildung_33(data):
    data1 = data.copy()
    data1.columns = [variables.loc[int(i.split(',')[3]),'name_de'] for i in data.columns]
    fig = data1.plot.bar(rot = 0, stacked = True, width = 0.5, figsize = (10.17,6.57), color = [(255/255, 102/255, 0/255),(191/255, 191/255, 191/255), (0, 0, 0)])
    fig.legend(bbox_to_anchor = (0.5, -0.2), loc = 10, frameon = False, borderaxespad=0.)
    fig.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.7n}'))
    yticks = [int(i) for i in list(fig.get_yticks())]
    yticks = map(numb_out2, yticks)
    fig.set_yticklabels(list(yticks))
    fig.set_xlabel('')
    fig.set_ylabel('Anzahl', fontweight = 'bold')
    
    fig.set_title('Eintwicklung der Empfänger von Transferleistungen in {} ({}-{})'.format(all_regions.loc[regids[0],'reg_name_de'], data1.index[0], data1.index[-1]), pad = 20)
    fig.grid(axis = 'y',zorder=0)
    fig.spines['top'].set_visible(False)
    fig.spines['right'].set_visible(False)
    fig.spines['left'].set_visible(False)
    return fig, data1
    
def Abbildung_34(data):
    fig = plt.figure(figsize=(10.17,6.57))
    ax = fig.gca()
    marker = itertools.cycle((',', 's', 'o', '^', '.', 'x','+','p','*'))
    rgbs = [(0, 0, 0), (255, 102, 0), (253,174,107), (191, 191, 191), (127, 127, 127), (140,140,140), (80,80,80)]
    colors = itertools.cycle([(x/255, y/255, z/255) for x,y,z in rgbs])
    #ax.plot(data)
    for i in data.columns:
        ax.plot(data[i], linewidth = 3, marker = next(marker), markersize = 8, color = next(colors))
    plt.xticks(list(data.index))
    plt.tick_params(axis='y', length = 0)
    if len(data) >= 12:
        plt.xticks(rotation = 'vertical')
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:n}'))
    ax.legend(bbox_to_anchor = (0.5, -0.2), labels = data.columns, loc = 10, frameon = False, borderaxespad=0., ncol=3)
    plt.ylabel('Index: '+str(base) + ' = 100', fontweight='bold')
    plt.title('Entwicklung der der Empfänger von sozialen Sicherungsleistungen \nin {} und der Region ({}-{})'.format(
            all_regions.loc[regids[0],'reg_name_de'], data1.index[0], data1.index[-1]), pad = 20)
    ax.grid(axis = 'y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    #plt.tight_layout()
    return fig

def update_base1(data, base):
    base_values = data.loc[base, :]
    Indexframe = pd.DataFrame()
    for col in data.columns:
        Indexframe[col] = (data.loc[:,col]/base_values[col])*100
    return Indexframe

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:file/csv;base64,{b64.decode()}" download="extract.xlsx">Download excel file</a> (right-click and save link as &lt;some_name&gt;.xlsx)' # decode b'abc' => abc
#%% Managing Output
if cat != "Abbildung_10":     
    data = load_data(regids, varids)



if cat == "Abbildung 1 & 2":
    years = st.sidebar.slider("Years to select", int(data.index.min()), int(data.index.max()), (int(data.index.min()), int(data.index.max())), 1)
    base = st.sidebar.selectbox(
            "Select the base year",
            list(range(years[0],years[1]+1)))
    data_subset = data[(data.index >= years[0])]
    data_subset = data_subset[(data_subset.index <= years[1])]
    data_subset = update_base(data_subset, base)
    fig = Abbildung_1(data_subset)
    st.pyplot(fig)
    if st.checkbox("Show Data", False):
        st.markdown(get_table_download_link(data_subset), unsafe_allow_html=True)
        st.write(data_subset)
        
if cat == "Abbildung 4":
    years = st.sidebar.slider("Years to select", int(data.index.min()), int(data.index.max()), (int(data.index.min()), int(data.index.max())), 1)
    data_subset = data[(data.index >= years[0])]
    data_subset = data_subset[(data_subset.index <= years[1])]
    fig = Abbildung_4(data_subset)
    st.pyplot(fig)
    if st.checkbox("Show Data", False):
        st.markdown(get_table_download_link(data_subset), unsafe_allow_html=True)
        st.write(data_subset)
        
if cat == "Abbildung 7":
    year = st.sidebar.selectbox(
            "Select the year",
            sorted(list(range(data.index.min() ,data.index.max())), reverse = True))
    fig, data = Abbildung_7(data, year)
    st.pyplot(fig)
    if st.checkbox("Show Data", False):
        st.markdown(get_table_download_link(data), unsafe_allow_html=True)
        st.write(data)
        
if cat == "Abbildung 8":
    year = st.sidebar.selectbox(
            "Select the year",
            sorted(list(range(data.index.min() ,data.index.max())), reverse = True))
    fig, data = Abbildung_8(data, year)
    st.pyplot(fig)
    if st.checkbox("Show Data", False):
        st.markdown(get_table_download_link(data), unsafe_allow_html=True)
        st.write(data)
        
if cat == "Abbildung 9":
    years = st.sidebar.slider("Years to select", int(data.index.min()), int(data.index.max()), (int(data.index.min()), int(data.index.max())), 1)
    fig, data = Abbildung_9(data, years)
    st.pyplot(fig)
    if st.checkbox("Show Data", False):
        st.markdown(get_table_download_link(data), unsafe_allow_html=True)
        st.write(data)
        
if cat == "Abbildung 10":
    data1 = load_data([regids[0]], varids)
    data2 = load_data(regids[1:], [varids[1]])
    data = pd.concat([data1, data2], axis = 1)
    data = data.dropna()
    years = st.sidebar.slider("Years to select", int(data.index.min()), int(data.index.max()), (int(data.index.min()), int(data.index.max())), 1)
    data_subset = data[(data.index >= years[0])]
    data_subset = data_subset[(data_subset.index <= years[1])]
    fig, data = Abbildung_10(data_subset)
    st.pyplot(fig)
    if st.checkbox("Show Data", False):
        st.markdown(get_table_download_link(data), unsafe_allow_html=True)
        st.write(data)
        
if cat == "Abbildung 12":
    years = st.sidebar.slider("Years to select", int(data.index.min()), int(data.index.max()), (int(data.index.min()), int(data.index.max())), 1)
    data_subset = data[(data.index >= years[0])]
    data_subset = data_subset[(data_subset.index <= years[1])]
    fig, data = Abbildung_12(data_subset)
    st.pyplot(fig)
    if st.checkbox("Show Data", False):
        st.markdown(get_table_download_link(data), unsafe_allow_html=True)
        st.write(data)
        
if cat == "Abbildung 18":
    years = st.sidebar.slider("Years to select", int(data.index.min()), int(data.index.max()), (int(data.index.min()), int(data.index.max())), 1)
    data_subset = data[(data.index >= years[0])]
    data_subset = data_subset[(data_subset.index <= years[1])]
    fig, data = Abbildung_18(data_subset)
    st.pyplot(fig)
    if st.checkbox("Show Data", False):
        st.markdown(get_table_download_link(data), unsafe_allow_html=True)
        st.write(data)
        
if cat == "Abbildung 19&20":
    fig, data = Abbildung_1920(data)
    st.pyplot(fig)
    if st.checkbox("Show Data", False):
        st.markdown(get_table_download_link(data), unsafe_allow_html=True)
        st.write(data.transpose())     
        
if cat == "Abbildung 28&29":
    years = st.sidebar.slider("Years to select", int(data.index.min()), int(data.index.max()), (int(data.index.min()), int(data.index.max())), 1)
    data_subset = data[(data.index >= years[0])]
    data_subset = data_subset[(data_subset.index <= years[1])]
    fig, data1 = Abbildung_2829(data_subset)
    st.pyplot(fig)
    if st.checkbox("Show Data", False):
        st.markdown(get_table_download_link(data), unsafe_allow_html=True)
        st.write(data1)
        
        
        
if cat == "Abbildung 33":
    years = st.sidebar.slider("Years to select", int(data.index.min()), int(data.index.max()), (int(data.index.min()), int(data.index.max())), 1)
    data_subset = data[(data.index >= years[0])]
    data_subset = data_subset[(data_subset.index <= years[1])]
    fig, data1 = Abbildung_33(data_subset)
    st.pyplot(fig.figure)
    if st.checkbox("Show Data", False):
        st.markdown(get_table_download_link(data1), unsafe_allow_html=True)
        st.write(data1)
        
if cat == "Abbildung 34":
    data1 = data.copy()
    data1.columns = [i.split(',')[1] for i in data1.columns]
    data1 = data1.transpose().reset_index()
    data1 = data1.groupby(by=['index']).sum().transpose()
    years = st.sidebar.slider("Years to select", int(data1.index.min()), int(data1.index.max()), (int(data1.index.min()), int(data1.index.max())), 1)
    base = st.sidebar.selectbox(
            "Select the base year",
            list(range(years[0],years[1]+1)))
    data_subset = data1[(data1.index >= years[0])]
    data_subset = data_subset[(data_subset.index <= years[1])]
    data_subset = update_base1(data_subset, base)
    fig = Abbildung_34(data_subset)
    st.pyplot(fig)
    if st.checkbox("Show Data", False):
        st.markdown(get_table_download_link(data_subset), unsafe_allow_html=True)
        st.write(data_subset)
