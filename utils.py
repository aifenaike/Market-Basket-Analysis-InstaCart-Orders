#Import relevant libraries
import numpy as np
import pandas as pd
pd.set_option('max_columns', 150)

import gc
import os

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#configure plots from matplotlib
matplotlib.rcParams['figure.dpi'] = 180 #resolution
matplotlib.rcParams['figure.figsize'] = (8,6) #figure size
color = sns.color_palette()

#Compilation of helper functions

def bar_visual(df, xlabel,ylabel, title):
    fig, ax = plt.subplots(figsize=(15,8))
    ax = sns.barplot(x = df.index, y = df.values, color = color[1])
    ax.set_xlabel(f'{xlabel}')
    ax.set_ylabel(f'{ylabel}')
    ax.xaxis.set_tick_params(rotation=90, labelsize=10)
    ax.set_title(f'{title}')
    fig.savefig(f'{title}.png')


def count_viz(df,xlabel,ylabel,title):
    fig, ax = plt.subplots(figsize = (10,5))
    ax = sns.countplot(df, color = color[1])
    ax.set_xlabel(f'{xlabel}', size = 10 )
    ax.set_ylabel(f'{ylabel}', size = 10)
    ax.tick_params(axis = 'both', labelsize = 8)
    ax.set_title(f'{title}')
    fig.savefig(f'{title}.png')
    plt.show()

def read(root,path):
    return pd.read_csv(root + path + "/" + path + ".csv")


def reduce_memory(df):
    
    """
    This function reduce the dataframe memory usage by converting it's type for easier handling.
    
    Parameters: Dataframe
    Return: Dataframe
    """
    
    start_mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    
    for col in df.columns:
        if df[col].dtypes in ["int64", "int32", "int16"]:
            
            cmin = df[col].min()
            cmax = df[col].max()
            
            if cmin > np.iinfo(np.int8).min and cmax < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            
            elif cmin > np.iinfo(np.int16).min and cmax < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            
            elif cmin > np.iinfo(np.int32).min and cmax < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        
        if df[col].dtypes in ["float64", "float32"]:
            
            cmin = df[col].min()
            cmax = df[col].max()
            
            if cmin > np.finfo(np.float16).min and cmax < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            
            elif cmin > np.finfo(np.float32).min and cmax < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
    
    print("")
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    
    return df
    

def overlapped_bar(y_df,x1_df,x2_df, xlabel,ylabel,title):
    fig, ax = plt.subplots(figsize = (15,8))
    ax = sns.barplot(y = y_df, x = x1_df, color=color[1], label = "total")
    ax = sns.barplot(y = y_df, x = x2_df, color=color[3], label = "reordered")
    ax.set_ylabel("Aisle")
    ax.set_xlabel("Orders Count")
    ax.set_title("Total Orders and Reorders From Most Popular Aisles")
    ax.legend(loc = 4, prop={'size': 12})
    plt.show()
    
# Annotate text on graph
def annotate_text(ax, append_to_text='%'):
    for p in ax.patches:
        txt = str(p.get_height().round(2)) + append_to_text
        txt_x = p.get_x() + p.get_width()/2.
        txt_y = 0.92*p.get_height()
        ax.text(txt_x,txt_y,txt, fontsize=20, color='#004235', ha='center', va='bottom')