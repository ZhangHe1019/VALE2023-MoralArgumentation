# Contents of ~/my_app/pages/page_3.py
import math
import copy
import streamlit as st
from streamlit_elements import elements, mui, html
st.set_option('deprecation.showPyplotGlobalUse', False)
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import numpy as np
from networkx.drawing.nx_pydot import graphviz_layout
import plotly.express as px


st.markdown("# Comparative corpora analysis 🎉")
st.sidebar.markdown("# Comparative corpora analysis 🎉")


# Contents of ~/my_app/pages/page_2.py
import streamlit as st
from streamlit_tree_select import tree_select
import time

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')
    
def select_corpora(i):
    nodes = [{"label": "US2016reddit (Real-time Reactions about US Presidential TV Debate on Reddit)", "value": "US2016reddit",
                 "children": [
                {"label": "US2016r1D (Democrats)", "value": "US2016r1D"},
                {"label": "US2016r1G (General)", "value": "US2016r1G"},
                {"label": "US2016r1R (Republicans)", "value": "US2016r1R"},
                ]
                 },

                {"label": "MoralMaze (Live Discussions on Moral Issues)", "value": "MoralMaze","children": [
                {"label": "British Empire", "value": "British Empire"},
                {"label": "DDay", "value": "DDay"},
                {"label": "Hypocrisy", "value": "Hypocrisy"},
                {"label": "Money", "value": "Money"},
                {"label": "Welfare", "value": "Welfare"},
                ]},
             
                {"label": "US2016tv (US Presidential TV Debate)", "value": "US2016tv",
                 "children": [
                {"label": "US2016tvD (Democrats)", "value": "US2016tvD"},
                {"label": "US2016tvG (General)", "value": "US2016tvG"},
                {"label": "US2016tvR (Republicans)", "value": "US2016tvR"},
                ]
                 },
            ]
    return_select = tree_select(nodes,show_expand_all=True,expand_on_click=True,check_model="leaf",key=i)
    # st.write(return_select)
    return return_select
def at_least_number_not_empty(d,num):
    non_empty_lists = [name for name, lst in d.items() if lst]
    return [len(non_empty_lists) >= num, non_empty_lists]
def add_datasetname(data_dict):
    dfs = []
    for key, df in data_dict.items():
        df['dataset_name'] = key
        dfs.append(df)

    # Concatenate all DataFrames together
    result = pd.concat(dfs, ignore_index=True)
    return result


def speak_speration(text):
    try:
        # Extract speaker name
        speaker_end_index = text.index(":")
        speaker_name = text[:speaker_end_index].strip()
        print(f"Speaker name: {speaker_name}")
    except ValueError:
        speaker_name = "None"
    return speaker_name

def Make_Word_Cloud(lexicon):
    wordcloud = WordCloud(background_color="#493E38", colormap='YlOrRd', width=1500, height=800,
                          normalize_plurals=False).generate(" ".join(lexicon))
    fig, ax = plt.subplots(figsize=(10, 10), facecolor=None)
    ax.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    return fig

def Word_Cloud(df, category):
    unique_words = list(set(",".join(df[category.lower()].dropna().values).split(",")))
    if len(unique_words) == 1 and len(unique_words[0]) == 0:
        f = st.info("No word cloud because of no {} in this corpus".format(category.lower()))
    else:
        f = st.pyplot(Make_Word_Cloud(unique_words))
    return f

def Compare_Shared_Words_Frequency(data, word_type):
    column_name = word_type.lower()

    df1 = data.copy(deep=True)
    df = df1.fillna("nan")
    df = df[df[column_name] != "nan"][["Text", "dataset_name", column_name]]
    df["dataset_name"] = df["dataset_name"].astype(str)
    df = df[[column_name, "Text", "dataset_name"]].apply(lambda x: x.str.split(","), axis=0).explode(column_name)
    df["dataset_name"] = df["dataset_name"].apply(lambda x: "".join(x))
    shared_words = df.groupby(['dataset_name', column_name])["Text"].count().unstack()
    shared_words1 = pd.DataFrame(shared_words.T, columns=shared_words.index,
                                 index=shared_words.columns).dropna().reset_index()
    if ((len(shared_words) == len(at_least_number_not_empty(data_selection,2)[1])) and (len(shared_words1) != 0)):
        shared_words_df = shared_words1
        shared_words_df.index = np.arange(1, len(shared_words_df) + 1, 1)
        shared_words_df = shared_words_df.set_index(column_name)
        shared_words_df["Total"] = shared_words_df.sum(axis=1)
        shared_words_df = shared_words_df.sort_values(["Total"], ascending=False)

    else:
        shared_words_df = pd.DataFrame()
    return shared_words_df

def Compare_Unique_Words_Frequency(data, word_type):
    column_name = word_type.lower()
    df1 = data.copy(deep=True)
    df = df1.fillna("nan")
    df = df[df[column_name] != "nan"][["Text", "dataset_name", column_name]]
    df = df[[column_name, "Text", "dataset_name"]].apply(lambda x: x.str.split(","), axis=0).explode(column_name)
    df["dataset_name"] = df["dataset_name"].apply(lambda x: "".join(x))
    shared_words = df.groupby(['dataset_name', column_name])["Text"].count().unstack()
    shared_words1 = pd.DataFrame(shared_words.T, columns=shared_words.index,
                                 index=shared_words.columns).reset_index()

    shared_words1 = shared_words1.set_index(column_name).fillna(0)
    column_list = shared_words1.columns
    unique_words_list = dict()
    for i in column_list.values.tolist():
        shared_words = shared_words1.copy(deep=True).sort_values([i], ascending=False)
        for j in column_list.values.tolist():
            if j != i:
                shared_words = shared_words[shared_words[i] != 0][shared_words[j] == 0]
        unique_words_list[i] = shared_words[i]

    return unique_words_list, column_list

def Compare_Moral_Foundation_Word_In_Tweet_Group(df, format1, moral_scale,num_scale, customisation):
    palette_map = customisation[0]
    width = customisation[1]
    height = customisation[2]
    add_baseline = customisation[3]
    
    annotation_fontsize = customisation[4]
    legend_fontsize = customisation[5]
    tick_fontsize = customisation[6]
    axis_fontsize = customisation[7]
    title_option = customisation[8]
    
    if moral_scale == 'Moral vs No moral':
        df["morals"] = (df["contains_loyalty_vice"]) | \
                                          (df["contains_loyalty_virtue"]) | \
                                          (df["contains_authority_vice"]) | \
                                          (df["contains_authority_virtue"]) | \
                                          (df["contains_sanctity_virtue"]) | \
                                          (df["contains_sanctity_vice"]) | \
                                          (df["contains_care_virtue"]) | \
                                          (df["contains_care_vice"]) | \
                                          (df["contains_fairness_virtue"]) | \
                                          (df["contains_fairness_vice"])
        df["no moral foundation words"] = (df["contains_loyalty_vice"] != 1) & \
                                          (df["contains_loyalty_virtue"] != 1) & \
                                          (df["contains_authority_vice"] != 1) & \
                                          (df["contains_authority_virtue"]!= 1) & \
                                          (df["contains_sanctity_virtue"]!= 1) & \
                                          (df["contains_sanctity_vice"]!= 1) & \
                                          (df["contains_care_virtue"]!= 1) & \
                                          (df["contains_care_vice"]!= 1) & \
                                          (df["contains_fairness_virtue"]!= 1) & \
                                          (df["contains_fairness_vice"]!= 1)

        column_list = ["morals","no moral foundation words","dataset_name"]
        column_renamemap = {"morals":"Morals",
                            "no moral foundation words":"No morals"}
        pattern = {"Morals": "-",
                   "No morals": "/"}
    elif moral_scale  == '2 Moral Valences':
        df["positive moral valence"] = (df["contains_loyalty_virtue"]) | \
                                          (df["contains_authority_virtue"]) | \
                                          (df["contains_sanctity_virtue"]) | \
                                          (df["contains_care_virtue"]) | \
                                          (df["contains_fairness_virtue"])
        df["negative moral valence"] = (df["contains_loyalty_vice"]) | \
                                          (df["contains_authority_vice"]) | \
                                          (df["contains_sanctity_vice"]) | \
                                          (df["contains_care_vice"]) | \
                                          (df["contains_fairness_vice"])
        column_list = ["positive moral valence", "negative moral valence","dataset_name"]
        column_renamemap = {"positive moral valence": "Positive",
                            "negative moral valence": "Negative"}
    
        pattern = {"Positive": "-",
                   "Negative": "/"}
        df["no moral foundation words"] = (df["contains_loyalty_vice"] != 1) & \
                                          (df["contains_loyalty_virtue"] != 1) & \
                                          (df["contains_authority_vice"] != 1) & \
                                          (df["contains_authority_virtue"]!= 1) & \
                                          (df["contains_sanctity_virtue"]!= 1) & \
                                          (df["contains_sanctity_vice"]!= 1) & \
                                          (df["contains_care_virtue"]!= 1) & \
                                          (df["contains_care_vice"]!= 1) & \
                                          (df["contains_fairness_virtue"]!= 1) & \
                                          (df["contains_fairness_vice"]!= 1)
        df = df[df["no moral foundation words"] != 1]
    elif moral_scale == "5 Moral Foundations":
        df["contains_care"] = (df["contains_care_virtue"]) | \
                                          (df["contains_care_vice"])
        df["contains_fairness"] = (df["contains_fairness_virtue"]) | \
                              (df["contains_fairness_vice"])
        df["contains_authority"] = (df["contains_authority_virtue"]) | \
                              (df["contains_authority_vice"])
        df["contains_loyalty"] = (df["contains_loyalty_virtue"]) | \
                              (df["contains_loyalty_vice"])
        df["contains_sanctity"] = (df["contains_sanctity_virtue"]) | \
                              (df["contains_sanctity_vice"])
        column_list = ["contains_care",
                       "contains_fairness",
                       "contains_loyalty",
                       "contains_authority",
                       "contains_sanctity",
                       "dataset_name"]
        column_renamemap = {"contains_care": "care",
                            "contains_fairness": "fairness",
                            "contains_sanctity": "sanctity",
                            "contains_authority": "authority",
                            "contains_loyalty": "loyalty"}
        pattern = {"care": "-",
                       "fairness": "/",
                        "loyalty": "+",
                        "authority": "x",
                        "sanctity": "."}
        df["no moral foundation words"] = (df["contains_loyalty_vice"] != 1) & \
                                          (df["contains_loyalty_virtue"] != 1) & \
                                          (df["contains_authority_vice"] != 1) & \
                                          (df["contains_authority_virtue"]!= 1) & \
                                          (df["contains_sanctity_virtue"]!= 1) & \
                                          (df["contains_sanctity_vice"]!= 1) & \
                                          (df["contains_care_virtue"]!= 1) & \
                                          (df["contains_care_vice"]!= 1) & \
                                          (df["contains_fairness_virtue"]!= 1) & \
                                          (df["contains_fairness_vice"]!= 1)
        df = df[df["no moral foundation words"] != 1]
    else:
        df["no moral foundation words"] = (df["contains_loyalty_vice"] != 1) & \
                                          (df["contains_loyalty_virtue"] != 1) & \
                                          (df["contains_authority_vice"] != 1) & \
                                          (df["contains_authority_virtue"]!= 1) & \
                                          (df["contains_sanctity_virtue"]!= 1) & \
                                          (df["contains_sanctity_vice"]!= 1) & \
                                          (df["contains_care_virtue"]!= 1) & \
                                          (df["contains_care_vice"]!= 1) & \
                                          (df["contains_fairness_virtue"]!= 1) & \
                                          (df["contains_fairness_vice"]!= 1)
        column_list = ["contains_care_virtue",
                       "contains_care_vice",
                       "contains_fairness_virtue",
                       "contains_fairness_vice",
                       "contains_sanctity_virtue",
                       "contains_sanctity_vice",
                       "contains_authority_virtue",
                       "contains_authority_vice",
                       "contains_loyalty_virtue",
                       "contains_loyalty_vice","dataset_name"]
        column_renamemap = {"contains_care_virtue": "care+",
                            "contains_care_vice": "care-",
                            "contains_fairness_virtue": "fairness+",
                            "contains_fairness_vice": "fairness-",
                            "contains_sanctity_virtue": "sanctity+",
                            "contains_sanctity_vice": "sanctity-",
                            "contains_authority_virtue": "authority+",
                            "contains_authority_vice": "authority-",
                            "contains_loyalty_virtue": "loyalty+",
                            "contains_loyalty_vice": "loyalty-"}
        pattern = {"care+": "-",
                       "care-": "-",
                       "fairness+": "/",
                       "fairness-": "/",
                       "loyalty+": "+",
                       "loyalty-": "+",
                       "authority+": "x",
                       "authority-": "x",
                       "sanctity+": ".",
                       "sanctity-": "."}
        df = df[df["no moral foundation words"] != 1]

    statistic_num = df[column_list].groupby(["dataset_name"]).sum().unstack().reset_index().rename(columns={"level_0": "Word_type", 0: "Number"})
    statistic_num["Word_type"] = statistic_num["Word_type"].map(column_renamemap)
    statistic_num = statistic_num[["dataset_name","Word_type","Number"]]
    tweet = df.groupby(["dataset_name"])["Text"].count().reset_index().rename(columns={"Text": "Number"})

    #st.write(len(df[df["dataset_name"]=="Dataset1"]))
    statistic_per = round(
        100 * statistic_num.set_index(["dataset_name", "Word_type"]) / tweet.set_index(["dataset_name"]),
        2).reset_index()
    statistic_per = statistic_per.rename(columns={"Number":"Percentage"})
    statistic_per = statistic_per.sort_values(by="Percentage", ascending=False).reset_index(drop=True)[["dataset_name","Word_type","Percentage"]]
    baselines = statistic_per.groupby(["Word_type"])["Percentage"].mean().to_dict()
    
    palette = palette_map
    if format1 == "number":
        table = statistic_num.sort_values(by="dataset_name").reset_index(drop=True)
        fig1 = go.Figure()
        for i, j in palette.items():
            fig1.add_trace(go.Bar(x=statistic_num[statistic_num["Word_type"] == i]["dataset_name"],
                                  y=statistic_num[statistic_num["Word_type"] == i]["Number"],
                                  name=i,
                                  marker_color=j,
                                  hovertemplate="Dataset: %{x} <br> Number: %{y}",
                                  texttemplate='%{y}',
                                  textposition='outside',
                                  textfont=dict(
                                      family="Arial",  # Font family
                                      size=annotation_fontsize,  # Font size
                                      color="black"  # Font color
                                  ),
                                  marker_pattern_shape=pattern[i]))

        fig1.update_layout(title={
                               'text': "{}".format(title_option),
                               'y': 0.9,
                               'x': 0.5,
                               'xanchor': 'center',
                               'yanchor': 'top'},
                           legend=dict(font=dict(
                               family='Arial',  # Specify your desired font family
                               size=legend_fontsize,  # Specify your desired font size
                               color='black'  # Specify your desired font color
                           )),
                           barmode='group',
                           width=width,
                           height=height,)
        fig1.update_yaxes(type=num_scale, ticklabelstep=2,titlefont=dict(
                                  family="Arial",
                                  color="black",  # Change font color
                                  size=axis_fontsize  # Change font size
                              ),
                              tickfont=dict(family="Arial",  # Change font family
                                            size=tick_fontsize,  # Change font size
                                            color="black"  # Change font color
                                            ))
        fig1.update_xaxes(categoryorder='category ascending',titlefont=dict(
                                  family="Arial",
                                  color="black",  # Change font color
                                  size=axis_fontsize  # Change font size
                              ),
                              tickfont=dict(family="Arial",  # Change font family
                                            size=tick_fontsize,  # Change font size
                                            color="black"  # Change font color
                                            ))
        fig1.update_yaxes(title_text='Number')
    else:
        table = statistic_per.sort_values(by="dataset_name").reset_index(drop=True)
        fig1 = go.Figure()
        for i, j in palette.items():
            fig1.add_trace(go.Bar(x=statistic_per[statistic_per["Word_type"] == i]["dataset_name"],
                                  y=statistic_per[statistic_per["Word_type"] == i]["Percentage"],
                                  name=i,
                                  marker_color=j,
                                  hovertemplate="Dataset: %{x} <br> Percentage: %{y}%",
                                  texttemplate='%{y}%',
                                  textposition='outside',
                                  textfont=dict(
                                      family="Arial",  # Font family
                                      size=annotation_fontsize,  # Font size
                                      color="black"  # Font color
                                  ),
                                  marker_pattern_shape=pattern[i]))
        if add_baseline == True:
            for type, baseline in baselines.items():
                fig1.add_shape(
                    type='line',
                    x0=-0.5,  # Adjust the x-axis position of the line
                    x1=len(statistic_per["dataset_name"].unique()) - 0.5,  # Adjust the x-axis position of the line
                    y0=baseline,
                    y1=baseline,
                    line=dict(
                        color=palette[type],  # Color of the dotted line
                        width=2,  # Width of the dotted line
                        dash='dot'  # Style of the line (dotted)
                    )
                )
        fig1.update_layout(title={'text': "{}".format(title_option),
                               'y': 0.9,
                               'x': 0.5,
                               'xanchor': 'center',
                               'yanchor': 'top'},
                           legend=dict(font=dict(
                               family='Arial',  # Specify your desired font family
                               size=legend_fontsize,  # Specify your desired font size
                               color='black'  # Specify your desired font color
                           )),
                                   barmode='group',
                                   width=width,
                                   height=height,)
        fig1.update_yaxes(type=num_scale, ticklabelstep=2)
        fig1.update_xaxes(categoryorder='category ascending',
                              titlefont=dict(
                                  family="Arial",
                                  color="black",  # Change font color
                                  size=axis_fontsize  # Change font size
                              ),
                              tickfont=dict(family="Arial",  # Change font family
                                            size=tick_fontsize,  # Change font size
                                            color="black"  # Change font color
                                            ))
        fig1.update_yaxes(title_text='Percentage',
                             titlefont=dict(
                                  family="Arial",
                                  color="black",  # Change font color
                                  size=axis_fontsize  # Change font size
                              ),
                              tickfont=dict(family="Arial",  # Change font family
                                            size=tick_fontsize,  # Change font size
                                            color="black"  # Change font color
                                            ))
    return fig1,table






def Compare_Moral_Foundation_Word_In_Tweet_Group_Deviation(df, format1, moral_scale,num_scale, customisation):
    palette_map = customisation[0]
    if moral_scale == 'Moral vs No moral':
        df["morals"] = (df["contains_loyalty_vice"]) | \
                                          (df["contains_loyalty_virtue"]) | \
                                          (df["contains_authority_vice"]) | \
                                          (df["contains_authority_virtue"]) | \
                                          (df["contains_sanctity_virtue"]) | \
                                          (df["contains_sanctity_vice"]) | \
                                          (df["contains_care_virtue"]) | \
                                          (df["contains_care_vice"]) | \
                                          (df["contains_fairness_virtue"]) | \
                                          (df["contains_fairness_vice"])
        df["no moral foundation words"] = (df["contains_loyalty_vice"] != 1) & \
                                          (df["contains_loyalty_virtue"] != 1) & \
                                          (df["contains_authority_vice"] != 1) & \
                                          (df["contains_authority_virtue"]!= 1) & \
                                          (df["contains_sanctity_virtue"]!= 1) & \
                                          (df["contains_sanctity_vice"]!= 1) & \
                                          (df["contains_care_virtue"]!= 1) & \
                                          (df["contains_care_vice"]!= 1) & \
                                          (df["contains_fairness_virtue"]!= 1) & \
                                          (df["contains_fairness_vice"]!= 1)

        column_list = ["morals","no moral foundation words","dataset_name"]
        column_renamemap = {"morals":"Morals",
                            "no moral foundation words":"No morals"}
        pattern = {"Morals": "-",
                   "No morals": "/"}
    elif moral_scale  == '2 Moral Valences':
        df["positive moral valence"] = (df["contains_loyalty_virtue"]) | \
                                          (df["contains_authority_virtue"]) | \
                                          (df["contains_sanctity_virtue"]) | \
                                          (df["contains_care_virtue"]) | \
                                          (df["contains_fairness_virtue"])
        df["negative moral valence"] = (df["contains_loyalty_vice"]) | \
                                          (df["contains_authority_vice"]) | \
                                          (df["contains_sanctity_vice"]) | \
                                          (df["contains_care_vice"]) | \
                                          (df["contains_fairness_vice"])
        df["no moral foundation words"] = (df["contains_loyalty_vice"] != 1) & \
                                          (df["contains_loyalty_virtue"] != 1) & \
                                          (df["contains_authority_vice"] != 1) & \
                                          (df["contains_authority_virtue"]!= 1) & \
                                          (df["contains_sanctity_virtue"]!= 1) & \
                                          (df["contains_sanctity_vice"]!= 1) & \
                                          (df["contains_care_virtue"]!= 1) & \
                                          (df["contains_care_vice"]!= 1) & \
                                          (df["contains_fairness_virtue"]!= 1) & \
                                          (df["contains_fairness_vice"]!= 1)
        column_list = ["positive moral valence", "negative moral valence","dataset_name"]
        column_renamemap = {"positive moral valence": "Positive",
                            "negative moral valence": "Negative"}
    
        pattern = {"Positive": "-",
                   "Negative": "/"}
        df = df[df["no moral foundation words"] != 1]
    elif moral_scale == "5 Moral Foundations":
        df["contains_care"] = (df["contains_care_virtue"]) | \
                                          (df["contains_care_vice"])
        df["contains_fairness"] = (df["contains_fairness_virtue"]) | \
                              (df["contains_fairness_vice"])
        df["contains_authority"] = (df["contains_authority_virtue"]) | \
                              (df["contains_authority_vice"])
        df["contains_loyalty"] = (df["contains_loyalty_virtue"]) | \
                              (df["contains_loyalty_vice"])
        df["contains_sanctity"] = (df["contains_sanctity_virtue"]) | \
                              (df["contains_sanctity_vice"])
        column_list = ["contains_care",
                       "contains_fairness",
                       "contains_loyalty",
                       "contains_authority",
                       "contains_sanctity",
                       "dataset_name"]
        column_renamemap = {"contains_care": "care",
                            "contains_fairness": "fairness",
                            "contains_sanctity": "sanctity",
                            "contains_authority": "authority",
                            "contains_loyalty": "loyalty"}
        pattern = {"care": "-",
                       "fairness": "/",
                        "loyalty": "+",
                        "authority": "x",
                        "sanctity": "."}
        df["no moral foundation words"] = (df["contains_loyalty_vice"] != 1) & \
                                          (df["contains_loyalty_virtue"] != 1) & \
                                          (df["contains_authority_vice"] != 1) & \
                                          (df["contains_authority_virtue"]!= 1) & \
                                          (df["contains_sanctity_virtue"]!= 1) & \
                                          (df["contains_sanctity_vice"]!= 1) & \
                                          (df["contains_care_virtue"]!= 1) & \
                                          (df["contains_care_vice"]!= 1) & \
                                          (df["contains_fairness_virtue"]!= 1) & \
                                          (df["contains_fairness_vice"]!= 1)
        df = df[df["no moral foundation words"] != 1]
    else:
        column_list = ["contains_care_virtue",
                       "contains_care_vice",
                       "contains_fairness_virtue",
                       "contains_fairness_vice",
                       "contains_sanctity_virtue",
                       "contains_sanctity_vice",
                       "contains_authority_virtue",
                       "contains_authority_vice",
                       "contains_loyalty_virtue",
                       "contains_loyalty_vice","dataset_name"]
        column_renamemap = {"contains_care_virtue": "care+",
                            "contains_care_vice": "care-",
                            "contains_fairness_virtue": "fairness+",
                            "contains_fairness_vice": "fairness-",
                            "contains_sanctity_virtue": "sanctity+",
                            "contains_sanctity_vice": "sanctity-",
                            "contains_authority_virtue": "authority+",
                            "contains_authority_vice": "authority-",
                            "contains_loyalty_virtue": "loyalty+",
                            "contains_loyalty_vice": "loyalty-"}
        pattern = {"care+": "-",
                       "care-": "-",
                       "fairness+": "/",
                       "fairness-": "/",
                       "loyalty+": "+",
                       "loyalty-": "+",
                       "authority+": "x",
                       "authority-": "x",
                       "sanctity+": ".",
                       "sanctity-": "."}
        df["no moral foundation words"] = (df["contains_loyalty_vice"] != 1) & \
                                          (df["contains_loyalty_virtue"] != 1) & \
                                          (df["contains_authority_vice"] != 1) & \
                                          (df["contains_authority_virtue"]!= 1) & \
                                          (df["contains_sanctity_virtue"]!= 1) & \
                                          (df["contains_sanctity_vice"]!= 1) & \
                                          (df["contains_care_virtue"]!= 1) & \
                                          (df["contains_care_vice"]!= 1) & \
                                          (df["contains_fairness_virtue"]!= 1) & \
                                          (df["contains_fairness_vice"]!= 1)
        df = df[df["no moral foundation words"] != 1]

    statistic_num = df[column_list].groupby(["dataset_name"]).sum().unstack().reset_index().rename(
        columns={"level_0": "Word_type", 0: "Number"})
    statistic_num["Word_type"] = statistic_num["Word_type"].map(column_renamemap)
    statistic_num = statistic_num[["dataset_name","Word_type","Number"]]
    tweet = df.groupby(["dataset_name"])["Text"].count().reset_index().rename(columns={"Text": "Number"})

    #st.write(len(df[df["dataset_name"]=="Dataset1"]))
    statistic_per = round(
        100 * statistic_num.set_index(["dataset_name", "Word_type"]) / tweet.set_index(["dataset_name"]),
        2).reset_index()
    statistic_per = statistic_per.rename(columns={"Number":"Percentage"})
    statistic_per = statistic_per.sort_values(by="Percentage", ascending=False).reset_index(drop=True)[["dataset_name","Word_type","Percentage"]]
    baselines = statistic_per.groupby(["Word_type"])["Percentage"].mean().reset_index()
    deviation = (statistic_per.set_index(["dataset_name","Word_type"])-baselines.set_index(["Word_type"])).reset_index()
    #st.write(deviation)
    deviation.columns = ["dataset_name","Word_type","deviation"]
    deviation = pd.merge(deviation,statistic_per, on=["Word_type","dataset_name"])
    #st.write(baselines, deviation)
    return baselines, deviation



def Compare_Moral_Foundation_Word_In_Tweet_Layer(df, format1, moral_scale,num_scale, customisation):
    palette_map = customisation[0]
    width = customisation[1]
    height = customisation[2]
    annotation_fontsize = customisation[3]
    legend_fontsize = customisation[4]
    tick_fontsize = customisation[5]
    axis_fontsize = customisation[6]
    title_option = customisation[7]
    pattern_list = ["-","/", "+","x", "."]
    pattern_dict = {}
    for index, (key, value) in enumerate(palette_map.items()):
        pattern_dict[key] = pattern_list[index]
    
    if moral_scale == 'Moral vs No moral':
        df["morals"] = (df["contains_loyalty_vice"]) | \
                                          (df["contains_loyalty_virtue"]) | \
                                          (df["contains_authority_vice"]) | \
                                          (df["contains_authority_virtue"]) | \
                                          (df["contains_sanctity_virtue"]) | \
                                          (df["contains_sanctity_vice"]) | \
                                          (df["contains_care_virtue"]) | \
                                          (df["contains_care_vice"]) | \
                                          (df["contains_fairness_virtue"]) | \
                                          (df["contains_fairness_vice"])

        df["no moral foundation words"] = (df["contains_loyalty_vice"] != 1) & \
                                          (df["contains_loyalty_virtue"] != 1) & \
                                          (df["contains_authority_vice"] != 1) & \
                                          (df["contains_authority_virtue"] != 1) & \
                                          (df["contains_sanctity_virtue"] != 1) & \
                                          (df["contains_sanctity_vice"] != 1) & \
                                          (df["contains_care_virtue"] != 1) & \
                                          (df["contains_care_vice"] != 1) & \
                                          (df["contains_fairness_virtue"] != 1) & \
                                          (df["contains_fairness_vice"] != 1)
        column_list = ["morals","no moral foundation words","dataset_name"]
        column_renamemap = {"morals":"Morals",
                            "no moral foundation words":"No morals"}
        order_list = ["Morals","No morals"]
    elif moral_scale  == '2 Moral Valences':
        df["positive moral valence"] = (df["contains_loyalty_virtue"]) | \
                                          (df["contains_authority_virtue"]) | \
                                          (df["contains_sanctity_virtue"]) | \
                                          (df["contains_care_virtue"]) | \
                                          (df["contains_fairness_virtue"])
        df["negative moral valence"] = (df["contains_loyalty_vice"]) | \
                                          (df["contains_authority_vice"]) | \
                                          (df["contains_sanctity_vice"]) | \
                                          (df["contains_care_vice"]) | \
                                          (df["contains_fairness_vice"])
        column_list = ["positive moral valence", "negative moral valence","dataset_name"]
        column_renamemap = {"positive moral valence": "Positive",
                            "negative moral valence": "Negative"}
        order_list = ["Positive","Negative"]
        df["no moral foundation words"] = (df["contains_loyalty_vice"] != 1) & \
                                          (df["contains_loyalty_virtue"] != 1) & \
                                          (df["contains_authority_vice"] != 1) & \
                                          (df["contains_authority_virtue"]!= 1) & \
                                          (df["contains_sanctity_virtue"]!= 1) & \
                                          (df["contains_sanctity_vice"]!= 1) & \
                                          (df["contains_care_virtue"]!= 1) & \
                                          (df["contains_care_vice"]!= 1) & \
                                          (df["contains_fairness_virtue"]!= 1) & \
                                          (df["contains_fairness_vice"]!= 1)
        df = df[df["no moral foundation words"] != 1]
    elif moral_scale == "5 Moral Foundations":
        df["contains_care"] = (df["contains_care_virtue"]) | \
                                          (df["contains_care_vice"])
        df["contains_fairness"] = (df["contains_fairness_virtue"]) | \
                              (df["contains_fairness_vice"])
        df["contains_authority"] = (df["contains_authority_virtue"]) | \
                              (df["contains_authority_vice"])
        df["contains_loyalty"] = (df["contains_loyalty_virtue"]) | \
                              (df["contains_loyalty_vice"])
        df["contains_sanctity"] = (df["contains_sanctity_virtue"]) | \
                              (df["contains_sanctity_vice"])
        column_list = ["contains_care",
                       "contains_fairness",
                       "contains_loyalty",
                       "contains_authority",
                       "contains_sanctity",
                       "dataset_name"]
        column_renamemap = {"contains_care": "care",
                            "contains_fairness": "fairness",
                            "contains_sanctity": "sanctity",
                            "contains_authority": "authority",
                            "contains_loyalty": "loyalty"}
        order_list = ["care",
                       "fairness",
                       "loyalty",
                       "authority",
                       "sanctity"]
        df["no moral foundation words"] = (df["contains_loyalty_vice"] != 1) & \
                                          (df["contains_loyalty_virtue"] != 1) & \
                                          (df["contains_authority_vice"] != 1) & \
                                          (df["contains_authority_virtue"]!= 1) & \
                                          (df["contains_sanctity_virtue"]!= 1) & \
                                          (df["contains_sanctity_vice"]!= 1) & \
                                          (df["contains_care_virtue"]!= 1) & \
                                          (df["contains_care_vice"]!= 1) & \
                                          (df["contains_fairness_virtue"]!= 1) & \
                                          (df["contains_fairness_vice"]!= 1)
        df = df[df["no moral foundation words"] != 1]
    else:
        column_list = ["contains_care_virtue",
                       "contains_care_vice",
                       "contains_fairness_virtue",
                       "contains_fairness_vice",
                       "contains_sanctity_virtue",
                       "contains_sanctity_vice",
                       "contains_authority_virtue",
                       "contains_authority_vice",
                       "contains_loyalty_virtue",
                       "contains_loyalty_vice","dataset_name"]
        column_renamemap = {"contains_care_virtue": "care+",
                            "contains_care_vice": "care-",
                            "contains_fairness_virtue": "fairness+",
                            "contains_fairness_vice": "fairness-",
                            "contains_sanctity_virtue": "sanctity+",
                            "contains_sanctity_vice": "sanctity-",
                            "contains_authority_virtue": "authority+",
                            "contains_authority_vice": "authority-",
                            "contains_loyalty_virtue": "loyalty+",
                            "contains_loyalty_vice": "loyalty-"}
        order_list = ["care+",
                       "care-",
                       "fairness+",
                       "fairness-",
                       "loyalty+",
                       "loyalty-",
                       "authority+",
                       "authority-",
                       "sanctity+",
                       "sanctity-"]
        df["no moral foundation words"] = (df["contains_loyalty_vice"] != 1) & \
                                          (df["contains_loyalty_virtue"] != 1) & \
                                          (df["contains_authority_vice"] != 1) & \
                                          (df["contains_authority_virtue"]!= 1) & \
                                          (df["contains_sanctity_virtue"]!= 1) & \
                                          (df["contains_sanctity_vice"]!= 1) & \
                                          (df["contains_care_virtue"]!= 1) & \
                                          (df["contains_care_vice"]!= 1) & \
                                          (df["contains_fairness_virtue"]!= 1) & \
                                          (df["contains_fairness_vice"]!= 1)
        df = df[df["no moral foundation words"] != 1]

    statistic_num = df[column_list].groupby(["dataset_name"]).sum().unstack().reset_index().rename(
        columns={"level_0": "Word_type", 0: "Number"})
    statistic_num["Word_type"] = statistic_num["Word_type"].map(column_renamemap)
    tweet = df.groupby(["dataset_name"])["Text"].count().reset_index().rename(columns={"Text": "Number"})
    statistic_per = round(
        100 * statistic_num.set_index(["dataset_name", "Word_type"]) / tweet.set_index(["dataset_name"]),
        2).reset_index()
    statistic_per = statistic_per.sort_values(by="Number", ascending=False).reset_index(drop=True)
    baselines = statistic_per.groupby(["Word_type"])["Number"].mean().to_dict()
    
    if format1 == "number":
        fig1 = go.Figure()
        for i,j in palette_map.items():
            fig1.add_trace(go.Bar(x=statistic_num[statistic_num["dataset_name"] == i]["Word_type"],
                                  y=statistic_num[statistic_num["dataset_name"] == i]["Number"],
                                  name=i,
                                  marker_color=j,
                                  hovertemplate="Word type: %{x} <br> Number: %{y}",
                                  texttemplate='%{y}',
                                  textfont=dict(
                                      family="Arial",  # Font family
                                      size=annotation_fontsize,  # Font size
                                      color="black"  # Font color
                                  ),
                                  textposition='outside',
                                  marker_pattern_shape=pattern_dict[i]))

        fig1.update_layout(barmode='group',
                           width=width,
                           height=height,
                           title={
                               'text': "{}".format(title_option),
                               'y': 0.9,
                               'x': 0.5,
                               'xanchor': 'center',
                               'yanchor': 'top'},
                           legend=dict(font=dict(
                               family='Arial',  # Specify your desired font family
                               size=legend_fontsize,  # Specify your desired font size
                               color='black'  # Specify your desired font color
                           )),
                           xaxis={'categoryorder': 'array', 'categoryarray': order_list})
        fig1.update_yaxes(type=num_scale, ticklabelstep=2,
                              titlefont=dict(
                                  family="Arial",
                                  color="black",  # Change font color
                                  size=axis_fontsize  # Change font size
                              ),
                              tickfont=dict(family="Arial",  # Change font family
                                            size=tick_fontsize,  # Change font size
                                            color="black"  # Change font color
                                            ))
        fig1.update_xaxes(title_text='Moral Foundation',
                             titlefont=dict(
                                  family="Arial",
                                  color="black",  # Change font color
                                  size=axis_fontsize  # Change font size
                              ),
                              tickfont=dict(family="Arial",  # Change font family
                                            size=tick_fontsize,  # Change font size
                                            color="black"  # Change font color
                                            ))
        fig1.update_yaxes(title_text='Number')

    else:
        fig1 = go.Figure()
        for i,j in palette_map.items():
            fig1.add_trace(go.Bar(x=statistic_per[statistic_per["dataset_name"] == i]["Word_type"],
                                  y=statistic_per[statistic_per["dataset_name"] == i]["Number"],
                                  name=i,
                                  marker_color=j,
                                  hovertemplate="Word type: %{x} <br> Percentage: %{y}%",
                                  texttemplate='%{y}%',
                                  textposition='outside',
                                  textfont=dict(
                                      family="Arial",  # Font family
                                      size=annotation_fontsize,  # Font size
                                      color="black"  # Font color
                                  ),
                                 marker_pattern_shape=pattern_dict[i]))
        # for type, baseline in baselines.items():
        #         fig1.add_shape(
        #             type='line',
        #             x0=-0.5,  # Adjust the x-axis position of the line
        #             #x1=len(df_user_per["dataset_name"].unique()) - 0.5,  # Adjust the x-axis position of the line
        #             y0=baseline,
        #             y1=baseline,
        #             line=dict(
        #                 #color="black",  # Color of the dotted line
        #                 width=2,  # Width of the dotted line
        #                 dash='dot'  # Style of the line (dotted)
        #             )
        #         )

        fig1.update_layout(barmode='group',
                           width=width,
                           height=height,
                           title={
                               'text': "{}".format(title_option),
                               'y': 0.9,
                               'x': 0.5,
                               'xanchor': 'center',
                               'yanchor': 'top'},
                           legend=dict(font=dict(
                               family='Arial',  # Specify your desired font family
                               size=legend_fontsize,  # Specify your desired font size
                               color='black'  # Specify your desired font color
                           )),
                           xaxis={'categoryorder': 'array', 'categoryarray': order_list})
        fig1.update_yaxes(type=num_scale, ticklabelstep=2,
                             titlefont=dict(
                                  family="Arial",
                                  color="black",  # Change font color
                                  size=axis_fontsize  # Change font size
                              ),
                              tickfont=dict(family="Arial",  # Change font family
                                            size=tick_fontsize,  # Change font size
                                            color="black"  # Change font color
                                            ))
        fig1.update_xaxes(title_text='Moral Foundation',
                              titlefont=dict(
                                  family="Arial",
                                  color="black",  # Change font color
                                  size=axis_fontsize  # Change font size
                              ),
                              tickfont=dict(family="Arial",  # Change font family
                                            size=tick_fontsize,  # Change font size
                                            color="black"  # Change font color
                                            ))
        fig1.update_yaxes(title_text='Percentage')
    return fig1

def Compare_Average_User_Concern_Score(df):
    df["care"] = ((df["contains_care_vice"] != 0) | (df["contains_care_virtue"] != 0)).astype(int)
    df["fairness"] = ((df["contains_fairness_vice"] != 0) | (df["contains_fairness_virtue"] != 0)).astype(int)
    df["loyalty"] = ((df["contains_loyalty_vice"] != 0) | (df["contains_loyalty_virtue"] != 0)).astype(int)
    df["authority"] = ((df["contains_authority_vice"] != 0) | (df["contains_authority_virtue"] != 0)).astype(int)
    df["sanctity"] = ((df["contains_sanctity_vice"] != 0) | (df["contains_sanctity_virtue"] != 0)).astype(int)
    df_user = df.groupby(["speaker", "dataset_name"])[
        ["care", "fairness", "loyalty", "authority", "sanctity"]].mean().reset_index()
    df_user = df_user.groupby(["dataset_name"])[["care", "fairness", "loyalty", "authority", "sanctity"]].mean()
    df_user_use = df_user.reset_index()
    import plotly.graph_objects as go

    fig = go.Figure()
    color = ["red", "green","blueviolet","blue","orange"]
    symbol = ['square',"circle-dot","diamond","star","cross"]
    max_value = df_user_use[["care", "fairness", "loyalty", "authority", "sanctity"]].max().max()
    max_value = np.ceil(max_value * 10) / 10
    for i, j in enumerate(df_user_use["dataset_name"]):
        fig.add_trace(go.Scatterpolar(r=df_user_use[["care", "fairness", "loyalty", "authority", "sanctity"]].values[i],
                                      theta=["Care", "Fairness", "Loyalty", "Authority", "Sanctity"],
                                      fill='toself',
                                      line_color=color[i],
                                      name=j,
                                      marker=dict(size=8, symbol=symbol[i])))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_value]
            )),
        width=500,
        height=500)
    return fig


def Compare_User_Distribution_Heatmap(df, dataset, format,customisation):
    width = customisation[0]
    height = customisation[1]
    df_user = df.groupby(["speaker", "dataset_name"])[["contains_sanctity_virtue", "contains_sanctity_vice",
                                                             "contains_loyalty_virtue", "contains_loyalty_vice",
                                                             "contains_authority_virtue", "contains_authority_vice",
                                                             "contains_care_virtue", "contains_care_vice",
                                                             "contains_fairness_virtue",
                                                             "contains_fairness_vice"]].sum().reset_index()
    df_user["mixed care"] = ((df_user["contains_care_virtue"] > 0) & (df_user["contains_care_vice"] > 0)).astype(int)
    df_user["mixed fairness"] = (
            (df_user["contains_fairness_virtue"] > 0) & (df_user["contains_fairness_vice"] > 0)).astype(int)
    df_user["mixed loyalty"] = (
            (df_user["contains_loyalty_virtue"] > 0) & (df_user["contains_loyalty_vice"] > 0)).astype(int)
    df_user["mixed authority"] = (
            (df_user["contains_authority_virtue"] > 0) & (df_user["contains_authority_vice"] > 0)).astype(int)
    df_user["mixed sanctity"] = (
            (df_user["contains_sanctity_virtue"] > 0) & (df_user["contains_sanctity_vice"] > 0)).astype(int)
    df_user["no care"] = (
            (df_user["contains_care_virtue"].values == 0) & (df_user["contains_care_vice"].values == 0)).astype(int)
    df_user["no fairness"] = (
            (df_user["contains_fairness_virtue"].values == 0) & (df_user["contains_fairness_vice"] == 0)).astype(
        int)
    df_user["no loyalty"] = (
            (df_user["contains_loyalty_virtue"] == 0) & (df_user["contains_loyalty_vice"] == 0)).astype(int)
    df_user["no authority"] = (
            (df_user["contains_authority_virtue"] == 0) & (df_user["contains_authority_vice"] == 0)).astype(int)
    df_user["no sanctity"] = (
            (df_user["contains_sanctity_virtue"] == 0) & (df_user["contains_sanctity_vice"] == 0)).astype(int)

    df_user["only care+"] = ((df_user["contains_care_virtue"] > 0) & (df_user["contains_care_vice"] == 0)).astype(
        int)
    df_user["only fairness+"] = (
            (df_user["contains_fairness_virtue"] > 0) & (df_user["contains_fairness_vice"] == 0)).astype(int)
    df_user["only loyalty+"] = (
            (df_user["contains_loyalty_virtue"] > 0) & (df_user["contains_loyalty_vice"] == 0)).astype(int)
    df_user["only authority+"] = (
            (df_user["contains_authority_virtue"] > 0) & (df_user["contains_authority_vice"] == 0)).astype(int)
    df_user["only sanctity+"] = (
            (df_user["contains_sanctity_virtue"] > 0) & (df_user["contains_sanctity_vice"] == 0)).astype(int)
    df_user["only care-"] = ((df_user["contains_care_virtue"] == 0) & (df_user["contains_care_vice"] > 0)).astype(
        int)
    df_user["only fairness-"] = (
            (df_user["contains_fairness_virtue"] == 0) & (df_user["contains_fairness_vice"] > 0)).astype(int)
    df_user["only loyalty-"] = (
            (df_user["contains_loyalty_virtue"] == 0) & (df_user["contains_loyalty_vice"] > 0)).astype(int)
    df_user["only sanctity-"] = (
            (df_user["contains_sanctity_virtue"] == 0) & (df_user["contains_sanctity_vice"] > 0)).astype(int)
    df_user["only authority-"] = (
            (df_user["contains_authority_virtue"] == 0) & (df_user["contains_authority_vice"] > 0)).astype(int)
    df_user = df_user[["dataset_name",
                        "mixed care",
                        "mixed fairness",
                        "mixed loyalty",
                        "mixed authority",
                        "mixed sanctity",
                        "no care",
                        "no fairness",
                        "no loyalty",
                        "no authority",
                        "no sanctity",
                        "only care+",
                        "only fairness+",
                        "only loyalty+",
                        "only authority+",
                        "only sanctity+",
                        "only care-",
                        "only fairness-",
                        "only loyalty-",
                        "only sanctity-",
                        "only authority-"]]
    df_user = df_user.groupby(["dataset_name"]).sum()
    df_user = df_user.unstack().reset_index()
    df_user.columns = ["moral valence","dataset_name","freq"]
    df_user['foundation'] = df_user['moral valence'].str.extract('(care|fairness|loyalty|authority|sanctity)')
    # Remove the 'mixed' string from the original column
    df_user['moral valence'] = df_user['moral valence'].str.replace('care', '')
    df_user['moral valence'] = df_user['moral valence'].str.replace('fairness', '')
    df_user['moral valence'] = df_user['moral valence'].str.replace('loyalty', '')
    df_user['moral valence'] = df_user['moral valence'].str.replace('authority', '')
    df_user['moral valence'] = df_user['moral valence'].str.replace('sanctity', '')
    if format == "percentage":
        df_user = (100*round(df_user.set_index(["foundation","dataset_name",'moral valence'])["freq"]/df_user.groupby(["foundation","dataset_name"])["freq"].sum(),4)).reset_index()
        color_name = "Percentage"
    else:
        color_name = "Number"

    df_user = df_user[df_user["dataset_name"] == dataset][["foundation",'moral valence',"freq"]]
    df_user = df_user.set_index(["foundation",'moral valence']).unstack()
    df_user.columns = df_user.columns.get_level_values(1)
    df_user = df_user[["no ", "only -", "mixed ", "only +"]]
    df_user.index = df_user.index.str.capitalize()
    df_user = df_user.reindex(["Care","Fairness","Loyalty","Authority","Sanctity"])
    fig = px.imshow(df_user,
                    labels=dict(x="Valence Category", y="Moral Foundation", color=color_name),
                    x=["Non",
                       "Only-",
                       "Mixed",
                       "Only+"],
                    y=df_user.index, text_auto=True, aspect="auto")
    fig.update_xaxes(side="top")
    fig.update_layout(width=width, height=height,title="{}".format(dataset),
                      title_font_color="black",  # change to your desired color
                      xaxis_tickfont_color="black",  # change to your desired color
                      yaxis_tickfont_color="black",  # change to your desired color
                      xaxis_title_font_color="black",  # change to your desired color
                      yaxis_title_font_color="black", # change to your desired color
                      title_font_size=17,  # change to your desired color
                      xaxis_tickfont_size=17,  # change to your desired color
                      yaxis_tickfont_size=17,  # change to your desired color
                      xaxis_title_font_size=17,  # change to your desired color
                      yaxis_title_font_size=17, # change to your desired color
                      legend_font=dict(color='black', size=17)
                      )
    fig.update_traces(textfont=dict(size=17, color='black'))
    return fig

def Compare_User_Distribution_Stack(df, moral_distribution, format,num_scale,customisation):
    palette = customisation[0]
    width = customisation[1]
    height = customisation[2]
    annotation_fontsize = customisation[3]
    legend_fontsize = customisation[4]
    tick_fontsize = customisation[5]
    axis_fontsize = customisation[6]
    title_option = customisation[7]

    pattern_list = ["-","/", "+","x", "."]
    pattern_dict = {}
    for index, (key, value) in enumerate(palette.items()):
        pattern_dict[key] = pattern_list[index]
    
    if moral_distribution == "care":
        moral_list = ["no care", "only care+", "mixed care", "only care-"]
    elif moral_distribution == "fairness":
        moral_list = ["no fairness", "only fairness+", "mixed fairness", "only fairness-"]
    elif moral_distribution == "loyalty":
        moral_list = ["no loyalty", "only loyalty+", "mixed loyalty", "only loyalty-"]
    elif moral_distribution == "sanctity":
        moral_list = ["no sanctity", "only sanctity+", "mixed sanctity", "only sanctity-"]
    else:
        moral_list = ["no authority", "only authority+", "mixed authority", "only authority-"]

    df_user = df.groupby(["speaker", "dataset_name"])[["contains_sanctity_virtue", "contains_sanctity_vice",
                                                             "contains_loyalty_virtue", "contains_loyalty_vice",
                                                             "contains_authority_virtue", "contains_authority_vice",
                                                             "contains_care_virtue", "contains_care_vice",
                                                             "contains_fairness_virtue",
                                                             "contains_fairness_vice"]].sum().reset_index()
    df_user["mixed care"] = ((df_user["contains_care_virtue"] > 0) & (df_user["contains_care_vice"] > 0)).astype(int)
    df_user["mixed fairness"] = (
            (df_user["contains_fairness_virtue"] > 0) & (df_user["contains_fairness_vice"] > 0)).astype(int)
    df_user["mixed loyalty"] = (
            (df_user["contains_loyalty_virtue"] > 0) & (df_user["contains_loyalty_vice"] > 0)).astype(int)
    df_user["mixed authority"] = (
            (df_user["contains_authority_virtue"] > 0) & (df_user["contains_authority_vice"] > 0)).astype(int)
    df_user["mixed sanctity"] = (
            (df_user["contains_sanctity_virtue"] > 0) & (df_user["contains_sanctity_vice"] > 0)).astype(int)
    df_user["no care"] = (
            (df_user["contains_care_virtue"].values == 0) & (df_user["contains_care_vice"].values == 0)).astype(int)
    df_user["no fairness"] = (
            (df_user["contains_fairness_virtue"].values == 0) & (df_user["contains_fairness_vice"] == 0)).astype(
        int)
    df_user["no loyalty"] = (
            (df_user["contains_loyalty_virtue"] == 0) & (df_user["contains_loyalty_vice"] == 0)).astype(int)
    df_user["no authority"] = (
            (df_user["contains_authority_virtue"] == 0) & (df_user["contains_authority_vice"] == 0)).astype(int)
    df_user["no sanctity"] = (
            (df_user["contains_sanctity_virtue"] == 0) & (df_user["contains_sanctity_vice"] == 0)).astype(int)

    df_user["only care+"] = ((df_user["contains_care_virtue"] > 0) & (df_user["contains_care_vice"] == 0)).astype(
        int)
    df_user["only fairness+"] = (
            (df_user["contains_fairness_virtue"] > 0) & (df_user["contains_fairness_vice"] == 0)).astype(int)
    df_user["only loyalty+"] = (
            (df_user["contains_loyalty_virtue"] > 0) & (df_user["contains_loyalty_vice"] == 0)).astype(int)
    df_user["only authority+"] = (
            (df_user["contains_authority_virtue"] > 0) & (df_user["contains_authority_vice"] == 0)).astype(int)
    df_user["only sanctity+"] = (
            (df_user["contains_sanctity_virtue"] > 0) & (df_user["contains_sanctity_vice"] == 0)).astype(int)
    df_user["only care-"] = ((df_user["contains_care_virtue"] == 0) & (df_user["contains_care_vice"] > 0)).astype(
        int)
    df_user["only fairness-"] = (
            (df_user["contains_fairness_virtue"] == 0) & (df_user["contains_fairness_vice"] > 0)).astype(int)
    df_user["only loyalty-"] = (
            (df_user["contains_loyalty_virtue"] == 0) & (df_user["contains_loyalty_vice"] > 0)).astype(int)
    df_user["only sanctity-"] = (
            (df_user["contains_sanctity_virtue"] == 0) & (df_user["contains_sanctity_vice"] > 0)).astype(int)
    df_user["only authority-"] = (
            (df_user["contains_authority_virtue"] == 0) & (df_user["contains_authority_vice"] > 0)).astype(int)
    df_user = df_user[['dataset_name'] + moral_list]
    df_user_number = df_user.groupby(["dataset_name"])[moral_list].sum()
    df_user_per = 100 * df_user_number.div(df_user_number.sum(axis=1), axis='rows')
    df_user_number = df_user_number.unstack().reset_index().round(2).rename(columns={"level_0": "mf", 0: "number"})
    df_user_per = df_user_per.unstack().reset_index().round(2).rename(columns={"level_0": "mf", 0: "percentage"})

    if format == "number":
        import plotly.graph_objects as go
        mf = df_user_number["mf"].unique()
        fig = go.Figure()
        for i, j in palette.items():
            fig.add_trace(go.Bar(
                y=df_user_number[df_user_number["mf"] == i]["dataset_name"].values,
                x=df_user_number[df_user_number["mf"] == i]["number"].values,
                name=i,
                marker_color=j,
                textposition='outside', texttemplate="%{x}",
                orientation="h",textfont=dict(
                                      family="Arial",  # Font family
                                      size=annotation_fontsize,  # Font size
                                      color="black"  # Font color
                                  ),
                marker_pattern_shape=pattern_dict[i]
            ))
            fig.update_layout(barmode='stack', width=width, height=height,
                              title={
                               'text': "{}".format(title_option),
                               'y': 0.9,
                               'x': 0.5,
                               'xanchor': 'center',
                               'yanchor': 'top'},
                           legend=dict(font=dict(
                               family='Arial',  # Specify your desired font family
                               size=legend_fontsize,  # Specify your desired font size
                               color='black'  # Specify your desired font color
                           )))
        if num_scale == "log":
            fig.update_xaxes(type="log", ticklabelstep=2, title_text='Number',
                          titlefont=dict(
                              family="Arial",
                              color="black",  # Change font color
                              size=axis_fontsize  # Change font size
                          ),
                          tickfont=dict(family="Arial",  # Change font family
                                        size=tick_fontsize,  # Change font size
                                        color="black"  # Change font color
                                        )
                          )
        else:
            fig.update_xaxes(ticklabelstep=2, title_text='Number',
                          titlefont=dict(
                              family="Arial",
                              color="black",  # Change font color
                              size=axis_fontsize  # Change font size
                          ),
                          tickfont=dict(family="Arial",  # Change font family
                                        size=tick_fontsize,  # Change font size
                                        color="black"  # Change font color
                                        )
                          )
        fig.update_yaxes(
                      categoryorder='category descending',
                      titlefont=dict(
                          family="Arial",
                          color="black",  # Change font color
                          size=axis_fontsize  # Change font size
                      ),
                      tickfont=dict(family="Arial",  # Change font family
                                    size=tick_fontsize,  # Change font size
                                    color="black"  # Change font color
                                    )
                      )
    elif format == "percentage":
        import plotly.graph_objects as go
        mf = df_user_per["mf"].unique()
        fig = go.Figure()
        for i, j in palette.items():
            fig.add_trace(go.Bar(
                y=df_user_per[df_user_per["mf"] == i]["dataset_name"].values,
                x=df_user_per[df_user_per["mf"] == i]["percentage"].values,
                name=i,
                marker_color=j,
                textposition='outside', texttemplate="%{x}%",
                orientation="h",
                textfont=dict(
                                      family="Arial",  # Font family
                                      size=annotation_fontsize,  # Font size
                                      color="black"  # Font color
                                  ),
                marker_pattern_shape=pattern_dict[i]
            ))
            fig.update_layout(barmode='stack', width=width, height=height,
                             title={
                               'text': "{}".format(title_option),
                               'y': 0.9,
                               'x': 0.5,
                               'xanchor': 'center',
                               'yanchor': 'top'},
                           legend=dict(font=dict(
                               family='Arial',  # Specify your desired font family
                               size=legend_fontsize,  # Specify your desired font size
                               color='black'  # Specify your desired font color
                           )))
        if num_scale == "log":
            fig.update_xaxes(type="log", ticklabelstep=2, title_text='Percentage',
                          titlefont=dict(
                              family="Arial",
                              color="black",  # Change font color
                              size=axis_fontsize  # Change font size
                          ),
                          tickfont=dict(family="Arial",  # Change font family
                                        size=tick_fontsize,  # Change font size
                                        color="black"  # Change font color
                                        )
                          )
        else:
            fig.update_xaxes(ticklabelstep=2, title_text='Percentage',
                          titlefont=dict(
                              family="Arial",
                              color="black",  # Change font color
                              size=axis_fontsize  # Change font size
                          ),
                          tickfont=dict(family="Arial",  # Change font family
                                        size=tick_fontsize,  # Change font size
                                        color="black"  # Change font color
                                        )
                          )
        fig.update_yaxes(
                      categoryorder='category descending',
                      titlefont=dict(
                          family="Arial",
                          color="black",  # Change font color
                          size=axis_fontsize  # Change font size
                      ),
                      tickfont=dict(family="Arial",  # Change font family
                                    size=tick_fontsize,  # Change font size
                                    color="black"  # Change font color
                                    )
                      )
    return fig

def Compare_User_Distribution_Group(df, moral_distribution, format,num_scale, customisation):
    palette = customisation[0]
    width = customisation[1]
    height = customisation[2]
    add_baseline = customisation[3]
    annotation_fontsize = customisation[4]
    legend_fontsize = customisation[5]
    tick_fontsize = customisation[6]
    axis_fontsize = customisation[7]
    title_option = customisation[8]
    
    if moral_distribution == "care":
        moral_list = ["no care", "only care+", "mixed care", "only care-"]
    elif moral_distribution == "fairness":
        moral_list = ["no fairness", "only fairness+", "mixed fairness", "only fairness-"]
    elif moral_distribution == "loyalty":
        moral_list = ["no loyalty", "only loyalty+", "mixed loyalty", "only loyalty-"]
    elif moral_distribution == "sanctity":
        moral_list = ["no sanctity", "only sanctity+", "mixed sanctity", "only sanctity-"]
    else:
        moral_list = ["no authority", "only authority+", "mixed authority", "only authority-"]

    df_user = df.groupby(["speaker", "dataset_name"])[["contains_sanctity_virtue", "contains_sanctity_vice",
                                                             "contains_loyalty_virtue", "contains_loyalty_vice",
                                                             "contains_authority_virtue", "contains_authority_vice",
                                                             "contains_care_virtue", "contains_care_vice",
                                                             "contains_fairness_virtue",
                                                             "contains_fairness_vice"]].sum().reset_index()
    df_user["mixed care"] = ((df_user["contains_care_virtue"] > 0) & (df_user["contains_care_vice"] > 0)).astype(int)
    df_user["mixed fairness"] = (
            (df_user["contains_fairness_virtue"] > 0) & (df_user["contains_fairness_vice"] > 0)).astype(int)
    df_user["mixed loyalty"] = (
            (df_user["contains_loyalty_virtue"] > 0) & (df_user["contains_loyalty_vice"] > 0)).astype(int)
    df_user["mixed authority"] = (
            (df_user["contains_authority_virtue"] > 0) & (df_user["contains_authority_vice"] > 0)).astype(int)
    df_user["mixed sanctity"] = (
            (df_user["contains_sanctity_virtue"] > 0) & (df_user["contains_sanctity_vice"] > 0)).astype(int)
    df_user["no care"] = (
            (df_user["contains_care_virtue"].values == 0) & (df_user["contains_care_vice"].values == 0)).astype(int)
    df_user["no fairness"] = (
            (df_user["contains_fairness_virtue"].values == 0) & (df_user["contains_fairness_vice"] == 0)).astype(
        int)
    df_user["no loyalty"] = (
            (df_user["contains_loyalty_virtue"] == 0) & (df_user["contains_loyalty_vice"] == 0)).astype(int)
    df_user["no authority"] = (
            (df_user["contains_authority_virtue"] == 0) & (df_user["contains_authority_vice"] == 0)).astype(int)
    df_user["no sanctity"] = (
            (df_user["contains_sanctity_virtue"] == 0) & (df_user["contains_sanctity_vice"] == 0)).astype(int)

    df_user["only care+"] = ((df_user["contains_care_virtue"] > 0) & (df_user["contains_care_vice"] == 0)).astype(
        int)
    df_user["only fairness+"] = (
            (df_user["contains_fairness_virtue"] > 0) & (df_user["contains_fairness_vice"] == 0)).astype(int)
    df_user["only loyalty+"] = (
            (df_user["contains_loyalty_virtue"] > 0) & (df_user["contains_loyalty_vice"] == 0)).astype(int)
    df_user["only authority+"] = (
            (df_user["contains_authority_virtue"] > 0) & (df_user["contains_authority_vice"] == 0)).astype(int)
    df_user["only sanctity+"] = (
            (df_user["contains_sanctity_virtue"] > 0) & (df_user["contains_sanctity_vice"] == 0)).astype(int)
    df_user["only care-"] = ((df_user["contains_care_virtue"] == 0) & (df_user["contains_care_vice"] > 0)).astype(
        int)
    df_user["only fairness-"] = (
            (df_user["contains_fairness_virtue"] == 0) & (df_user["contains_fairness_vice"] > 0)).astype(int)
    df_user["only loyalty-"] = (
            (df_user["contains_loyalty_virtue"] == 0) & (df_user["contains_loyalty_vice"] > 0)).astype(int)
    df_user["only sanctity-"] = (
            (df_user["contains_sanctity_virtue"] == 0) & (df_user["contains_sanctity_vice"] > 0)).astype(int)
    df_user["only authority-"] = (
            (df_user["contains_authority_virtue"] == 0) & (df_user["contains_authority_vice"] > 0)).astype(int)
    df_user = df_user[['dataset_name'] + moral_list]
    df_user_number = df_user.groupby(["dataset_name"])[moral_list].sum()
    df_user_per = 100 * df_user_number.div(df_user_number.sum(axis=1), axis='rows')
    df_user_number = df_user_number.unstack().reset_index().round(2).rename(columns={"level_0": "mf", 0: "number"})
    df_user_per = df_user_per.unstack().reset_index().round(2).rename(columns={"level_0": "mf", 0: "percentage"})
    baselines = df_user_per.groupby(["mf"])["percentage"].mean().to_dict()

    if format == "number":
        import plotly.graph_objects as go
        mf = df_user_number["mf"].unique()
        fig = go.Figure()
        for i,j in palette.items():
            fig.add_trace(go.Bar(
                y=df_user_number[df_user_number["mf"] == i]["dataset_name"].values,
                x=df_user_number[df_user_number["mf"] == i]["number"].values,
                name=i,
                marker_color = j,
                textposition='outside', texttemplate="%{x}",
                textfont=dict(
                                      family="Arial",  # Font family
                                      size=annotation_fontsize,  # Font size
                                      color="black"  # Font color
                                  ),
                orientation="h",
            ))
            fig.update_layout(barmode='group', width=width, height=height,
                             title={
                               'text': "{}".format(title_option),
                               'y': 0.9,
                               'x': 0.5,
                               'xanchor': 'center',
                               'yanchor': 'top'},
                           legend=dict(font=dict(
                               family='Arial',  # Specify your desired font family
                               size=legend_fontsize,  # Specify your desired font size
                               color='black'  # Specify your desired font color
                           )))
        if num_scale == "log":
            fig.update_xaxes(type="log", ticklabelstep=2, title_text='Number',
                          titlefont=dict(
                              family="Arial",
                              color="black",  # Change font color
                              size=axis_fontsize  # Change font size
                          ),
                          tickfont=dict(family="Arial",  # Change font family
                                        size=tick_fontsize,  # Change font size
                                        color="black"  # Change font color
                                        )
                          )
        else:
            fig.update_xaxes(ticklabelstep=2, title_text='Number',
                          titlefont=dict(
                              family="Arial",
                              color="black",  # Change font color
                              size=axis_fontsize  # Change font size
                          ),
                          tickfont=dict(family="Arial",  # Change font family
                                        size=tick_fontsize,  # Change font size
                                        color="black"  # Change font color
                                        )
                          )
        fig.update_yaxes(
                      categoryorder='category descending',
                      titlefont=dict(
                          family="Arial",
                          color="black",  # Change font color
                          size=axis_fontsize  # Change font size
                      ),
                      tickfont=dict(family="Arial",  # Change font family
                                    size=tick_fontsize,  # Change font size
                                    color="black"  # Change font color
                                    )
                      )
    elif format == "percentage":
        import plotly.graph_objects as go
        mf = df_user_per["mf"].unique()
        fig = go.Figure()
        for i,j in palette.items():
            fig.add_trace(go.Bar(
                y=df_user_per[df_user_per["mf"] == i]["dataset_name"].values,
                x=df_user_per[df_user_per["mf"] == i]["percentage"].values,
                name=i,
                marker_color = j,
                textposition='outside', texttemplate="%{x}%",textfont=dict(
                                      family="Arial",  # Font family
                                      size=annotation_fontsize,  # Font size
                                      color="black"  # Font color
                                      ),
                                    orientation="h"))
            fig.update_layout(barmode='group', width=width, height=height,
                             title={
                               'text': "{}".format(title_option),
                               'y': 0.9,
                               'x': 0.5,
                               'xanchor': 'center',
                               'yanchor': 'top'},
                           legend=dict(font=dict(
                               family='Arial',  # Specify your desired font family
                               size=legend_fontsize,  # Specify your desired font size
                               color='black'  # Specify your desired font color
                           )))
        if add_baseline == True:
            for type, baseline in baselines.items():
                    fig.add_shape(
                        type='line',
                        y0=-0.5,  # Adjust the x-axis position of the line
                        y1=len(df_user_per["dataset_name"].unique()) - 0.5,  # Adjust the x-axis position of the line
                        x0=baseline,
                        x1=baseline,
                        line=dict(
                            color=palette[type],  # Color of the dotted line
                            width=2,  # Width of the dotted line
                            dash='dot'  # Style of the line (dotted)
                        ))                    
        if num_scale == "log":
            fig.update_xaxes(type="log", ticklabelstep=2, title_text='Percentage',
                              titlefont=dict(
                                  family="Arial",
                                  color="black",  # Change font color
                                  size=axis_fontsize  # Change font size
                              ),
                              tickfont=dict(family="Arial",  # Change font family
                                            size=tick_fontsize,  # Change font size
                                            color="black"  # Change font color
                                            )
                              )
        else:
            fig.update_xaxes(ticklabelstep=2, title_text='Percentage',
                              titlefont=dict(
                                  family="Arial",
                                  color="black",  # Change font color
                                  size=axis_fontsize  # Change font size
                              ),
                              tickfont=dict(family="Arial",  # Change font family
                                            size=tick_fontsize,  # Change font size
                                            color="black"  # Change font color
                                            )
                              )
        fig.update_yaxes(
                          categoryorder='category descending',
                          titlefont=dict(
                              family="Arial",
                              color="black",  # Change font color
                              size=axis_fontsize  # Change font size
                          ),
                          tickfont=dict(family="Arial",  # Change font family
                                        size=tick_fontsize,  # Change font size
                                        color="black"  # Change font color
                                        ))    
    return fig


def Compare_User_Distribution_Deviation(df, moral_distribution, format,num_scale, customisation):
    
    palette = customisation[0]
    if moral_distribution == "care":
        moral_list = ["no care", "only care+", "mixed care", "only care-"]
    elif moral_distribution == "fairness":
        moral_list = ["no fairness", "only fairness+", "mixed fairness", "only fairness-"]
    elif moral_distribution == "loyalty":
        moral_list = ["no loyalty", "only loyalty+", "mixed loyalty", "only loyalty-"]
    elif moral_distribution == "sanctity":
        moral_list = ["no sanctity", "only sanctity+", "mixed sanctity", "only sanctity-"]
    else:
        moral_list = ["no authority", "only authority+", "mixed authority", "only authority-"]

    df_user = df.groupby(["speaker", "dataset_name"])[["contains_sanctity_virtue", "contains_sanctity_vice",
                                                             "contains_loyalty_virtue", "contains_loyalty_vice",
                                                             "contains_authority_virtue", "contains_authority_vice",
                                                             "contains_care_virtue", "contains_care_vice",
                                                             "contains_fairness_virtue",
                                                             "contains_fairness_vice"]].sum().reset_index()
    df_user["mixed care"] = ((df_user["contains_care_virtue"] > 0) & (df_user["contains_care_vice"] > 0)).astype(int)
    df_user["mixed fairness"] = (
            (df_user["contains_fairness_virtue"] > 0) & (df_user["contains_fairness_vice"] > 0)).astype(int)
    df_user["mixed loyalty"] = (
            (df_user["contains_loyalty_virtue"] > 0) & (df_user["contains_loyalty_vice"] > 0)).astype(int)
    df_user["mixed authority"] = (
            (df_user["contains_authority_virtue"] > 0) & (df_user["contains_authority_vice"] > 0)).astype(int)
    df_user["mixed sanctity"] = (
            (df_user["contains_sanctity_virtue"] > 0) & (df_user["contains_sanctity_vice"] > 0)).astype(int)
    df_user["no care"] = (
            (df_user["contains_care_virtue"].values == 0) & (df_user["contains_care_vice"].values == 0)).astype(int)
    df_user["no fairness"] = (
            (df_user["contains_fairness_virtue"].values == 0) & (df_user["contains_fairness_vice"] == 0)).astype(
        int)
    df_user["no loyalty"] = (
            (df_user["contains_loyalty_virtue"] == 0) & (df_user["contains_loyalty_vice"] == 0)).astype(int)
    df_user["no authority"] = (
            (df_user["contains_authority_virtue"] == 0) & (df_user["contains_authority_vice"] == 0)).astype(int)
    df_user["no sanctity"] = (
            (df_user["contains_sanctity_virtue"] == 0) & (df_user["contains_sanctity_vice"] == 0)).astype(int)

    df_user["only care+"] = ((df_user["contains_care_virtue"] > 0) & (df_user["contains_care_vice"] == 0)).astype(
        int)
    df_user["only fairness+"] = (
            (df_user["contains_fairness_virtue"] > 0) & (df_user["contains_fairness_vice"] == 0)).astype(int)
    df_user["only loyalty+"] = (
            (df_user["contains_loyalty_virtue"] > 0) & (df_user["contains_loyalty_vice"] == 0)).astype(int)
    df_user["only authority+"] = (
            (df_user["contains_authority_virtue"] > 0) & (df_user["contains_authority_vice"] == 0)).astype(int)
    df_user["only sanctity+"] = (
            (df_user["contains_sanctity_virtue"] > 0) & (df_user["contains_sanctity_vice"] == 0)).astype(int)
    df_user["only care-"] = ((df_user["contains_care_virtue"] == 0) & (df_user["contains_care_vice"] > 0)).astype(
        int)
    df_user["only fairness-"] = (
            (df_user["contains_fairness_virtue"] == 0) & (df_user["contains_fairness_vice"] > 0)).astype(int)
    df_user["only loyalty-"] = (
            (df_user["contains_loyalty_virtue"] == 0) & (df_user["contains_loyalty_vice"] > 0)).astype(int)
    df_user["only sanctity-"] = (
            (df_user["contains_sanctity_virtue"] == 0) & (df_user["contains_sanctity_vice"] > 0)).astype(int)
    df_user["only authority-"] = (
            (df_user["contains_authority_virtue"] == 0) & (df_user["contains_authority_vice"] > 0)).astype(int)
    df_user = df_user[['dataset_name'] + moral_list]
    df_user_number = df_user.groupby(["dataset_name"])[moral_list].sum()
    df_user_per = 100 * df_user_number.div(df_user_number.sum(axis=1), axis='rows')
    df_user_per = df_user_per.unstack().reset_index().round(2).rename(columns={"level_0": "mf", 0: "percentage"})
    baselines = df_user_per.groupby(["mf"])["percentage"].mean().reset_index()
    deviation = (df_user_per.set_index(["mf","dataset_name"]) - baselines.set_index(["mf"])).reset_index()
    deviation.columns = ["mf","dataset_name","deviation"]
    deviation = pd.merge(deviation,df_user_per, on=["mf","dataset_name"])
    return baselines, deviation
    
def Compare_User_Distribution_Layer(df, moral_distribution, format,num_scale,customisation):
    palette = customisation[0]
    width = customisation[1]
    height = customisation[2]
    annotation_fontsize = customisation[3]
    legend_fontsize = customisation[4]
    tick_fontsize = customisation[5]
    axis_fontsize = customisation[6]
    title_option = customisation[7]
    
    if moral_distribution == "care":
        moral_list = ["no care", "only care+", "mixed care", "only care-"]
    elif moral_distribution == "fairness":
        moral_list = ["no fairness", "only fairness+", "mixed fairness", "only fairness-"]
    elif moral_distribution == "loyalty":
        moral_list = ["no loyalty", "only loyalty+", "mixed loyalty", "only loyalty-"]
    elif moral_distribution == "sanctity":
        moral_list = ["no sanctity", "only sanctity+", "mixed sanctity", "only sanctity-"]
    else:
        moral_list = ["no authority", "only authority+", "mixed authority", "only authority-"]

    df_user = df.groupby(["speaker", "dataset_name"])[["contains_sanctity_virtue", "contains_sanctity_vice",
                                                             "contains_loyalty_virtue", "contains_loyalty_vice",
                                                             "contains_authority_virtue", "contains_authority_vice",
                                                             "contains_care_virtue", "contains_care_vice",
                                                             "contains_fairness_virtue",
                                                             "contains_fairness_vice"]].sum().reset_index()
    df_user["mixed care"] = ((df_user["contains_care_virtue"] > 0) & (df_user["contains_care_vice"] > 0)).astype(int)
    df_user["mixed fairness"] = (
            (df_user["contains_fairness_virtue"] > 0) & (df_user["contains_fairness_vice"] > 0)).astype(int)
    df_user["mixed loyalty"] = (
            (df_user["contains_loyalty_virtue"] > 0) & (df_user["contains_loyalty_vice"] > 0)).astype(int)
    df_user["mixed authority"] = (
            (df_user["contains_authority_virtue"] > 0) & (df_user["contains_authority_vice"] > 0)).astype(int)
    df_user["mixed sanctity"] = (
            (df_user["contains_sanctity_virtue"] > 0) & (df_user["contains_sanctity_vice"] > 0)).astype(int)
    df_user["no care"] = (
            (df_user["contains_care_virtue"].values == 0) & (df_user["contains_care_vice"].values == 0)).astype(int)
    df_user["no fairness"] = (
            (df_user["contains_fairness_virtue"].values == 0) & (df_user["contains_fairness_vice"] == 0)).astype(
        int)
    df_user["no loyalty"] = (
            (df_user["contains_loyalty_virtue"] == 0) & (df_user["contains_loyalty_vice"] == 0)).astype(int)
    df_user["no authority"] = (
            (df_user["contains_authority_virtue"] == 0) & (df_user["contains_authority_vice"] == 0)).astype(int)
    df_user["no sanctity"] = (
            (df_user["contains_sanctity_virtue"] == 0) & (df_user["contains_sanctity_vice"] == 0)).astype(int)

    df_user["only care+"] = ((df_user["contains_care_virtue"] > 0) & (df_user["contains_care_vice"] == 0)).astype(
        int)
    df_user["only fairness+"] = (
            (df_user["contains_fairness_virtue"] > 0) & (df_user["contains_fairness_vice"] == 0)).astype(int)
    df_user["only loyalty+"] = (
            (df_user["contains_loyalty_virtue"] > 0) & (df_user["contains_loyalty_vice"] == 0)).astype(int)
    df_user["only authority+"] = (
            (df_user["contains_authority_virtue"] > 0) & (df_user["contains_authority_vice"] == 0)).astype(int)
    df_user["only sanctity+"] = (
            (df_user["contains_sanctity_virtue"] > 0) & (df_user["contains_sanctity_vice"] == 0)).astype(int)
    df_user["only care-"] = ((df_user["contains_care_virtue"] == 0) & (df_user["contains_care_vice"] > 0)).astype(
        int)
    df_user["only fairness-"] = (
            (df_user["contains_fairness_virtue"] == 0) & (df_user["contains_fairness_vice"] > 0)).astype(int)
    df_user["only loyalty-"] = (
            (df_user["contains_loyalty_virtue"] == 0) & (df_user["contains_loyalty_vice"] > 0)).astype(int)
    df_user["only sanctity-"] = (
            (df_user["contains_sanctity_virtue"] == 0) & (df_user["contains_sanctity_vice"] > 0)).astype(int)
    df_user["only authority-"] = (
            (df_user["contains_authority_virtue"] == 0) & (df_user["contains_authority_vice"] > 0)).astype(int)
    df_user = df_user[['dataset_name'] + moral_list]
    df_user_number = df_user.groupby(["dataset_name"])[moral_list].sum()
    df_user_per = 100 * df_user_number.div(df_user_number.sum(axis=1), axis='rows')
    df_user_number = df_user_number.reset_index().round(2)
    df_user_per = df_user_per.reset_index().round(2)
    baselines = df_user_per[moral_list].mean().to_dict()
    
    if format == "number":
        import plotly.graph_objects as go

        platform_dataset_name = df_user_number.set_index("dataset_name").columns

        fig = go.Figure()
        color = palette
        for i in np.arange(len(df_user_number)):
            fig.add_trace(go.Bar(
                x=platform_dataset_name,
                y=df_user_number.set_index("dataset_name").iloc[i],
                name=df_user_number["dataset_name"][i],
                marker_color=color[df_user_number["dataset_name"][i]],
                textposition='outside', texttemplate="%{y}",
                textfont=dict(
                            family="Arial",  # Font family
                            size=annotation_fontsize,  # Font size
                            color="black"  # Font color
                                  )
            ))
            fig.update_layout(barmode='group', xaxis_tickangle=-45, width=width, height=height,
                             title={
                               'text': "{}".format(title_option),
                               'y': 0.9,
                               'x': 0.5,
                               'xanchor': 'center',
                               'yanchor': 'top'},
                           legend=dict(font=dict(
                               family='Arial',  # Specify your desired font family
                               size=legend_fontsize,  # Specify your desired font size
                               color='black'  # Specify your desired font color
                           )))
        if num_scale == "log":
            fig.update_yaxes(type="log", ticklabelstep=2, title_text='Number',
                          titlefont=dict(
                              family="Arial",
                              color="black",  # Change font color
                              size=axis_fontsize  # Change font size
                          ),
                          tickfont=dict(family="Arial",  # Change font family
                                        size=tick_fontsize,  # Change font size
                                        color="black"  # Change font color
                                        )
                          )
        else:
            fig.update_yaxes(ticklabelstep=2, title_text='Number',
                          titlefont=dict(
                              family="Arial",
                              color="black",  # Change font color
                              size=axis_fontsize  # Change font size
                          ),
                          tickfont=dict(family="Arial",  # Change font family
                                        size=tick_fontsize,  # Change font size
                                        color="black"  # Change font color
                                        )
                          )
        fig.update_xaxes(categoryorder='category descending',
                      titlefont=dict(
                          family="Arial",
                          color="black",  # Change font color
                          size=axis_fontsize  # Change font size
                      ),
                      tickfont=dict(family="Arial",  # Change font family
                                    size=tick_fontsize,  # Change font size
                                    color="black"  # Change font color
                                    )
                      )
    elif format == "percentage":
        import plotly.graph_objects as go

        platform_dataset_name = df_user_per.set_index("dataset_name").columns

        fig = go.Figure()
        color = palette
        for i in np.arange(len(df_user_per)):
            fig.add_trace(go.Bar(
                x=platform_dataset_name,
                y=df_user_per.set_index("dataset_name").iloc[i],
                name=df_user_per["dataset_name"][i],
                marker_color=color[df_user_per["dataset_name"][i]],
                textposition='outside', texttemplate="%{y}%",
                textfont=dict(
                             family="Arial",  # Font family
                            size=annotation_fontsize,  # Font size
                            color="black"  # Font color
                                  )
            ))
            fig.update_layout(barmode='group', xaxis_tickangle=-45, width=width, height=height,
                             title={
                               'text': "{}".format(title_option),
                               'y': 0.9,
                               'x': 0.5,
                               'xanchor': 'center',
                               'yanchor': 'top'},
                           legend=dict(font=dict(
                               family='Arial',  # Specify your desired font family
                               size=legend_fontsize,  # Specify your desired font size
                               color='black'  # Specify your desired font color
                           )))
        if num_scale == "log":
            fig.update_yaxes(type="log", ticklabelstep=2, title_text='Percentage',
                          titlefont=dict(
                              family="Arial",
                              color="black",  # Change font color
                              size=axis_fontsize  # Change font size
                          ),
                          tickfont=dict(family="Arial",  # Change font family
                                        size=tick_fontsize,  # Change font size
                                        color="black"  # Change font color
                                        )
                          )
        else:
            fig.update_yaxes(ticklabelstep=2, title_text='Percentage',
                          titlefont=dict(
                              family="Arial",
                              color="black",  # Change font color
                              size=axis_fontsize  # Change font size
                          ),
                          tickfont=dict(family="Arial",  # Change font family
                                        size=tick_fontsize,  # Change font size
                                        color="black"  # Change font color
                                        )
                          )
        fig.update_xaxes(categoryorder='category descending',
                      titlefont=dict(
                          family="Arial",
                          color="black",  # Change font color
                          size=axis_fontsize  # Change font size
                      ),
                      tickfont=dict(family="Arial",  # Change font family
                                    size=tick_fontsize,  # Change font size
                                    color="black"  # Change font color
                                    )
                      )
        # for type, baseline in baselines.items():
        #         fig.add_shape(
        #             type='line',
        #             x0=-0.5,  # Adjust the x-axis position of the line
        #             #x1=len(df_user_per["dataset_name"].unique()) - 0.5,  # Adjust the x-axis position of the line
        #             y0=baseline,
        #             y1=baseline,
        #             line=dict(
        #                 #color="black",  # Color of the dotted line
        #                 width=2,  # Width of the dotted line
        #                 dash='dot'  # Style of the line (dotted)
        #             )
        #         )
            
    return fig

@st.cache_data
def ADU_dataloader(file_list):
    dfs = [pd.read_excel(file) for file in file_list]
    data = pd.concat(dfs, ignore_index=True)
    data["speaker"] = data["Text"].apply(lambda x: speak_speration(x))
    data["Arg-ADUs"] = data["support"] | data["attack"]
    data["Nonarg-ADUs"] = (data[['support', 'attack']].sum(axis=1) == 0)
    data["Support"] = data["support"]
    data["Attack"] = data["attack"]
    data["Input"] = data["attack_input"] | data["support_input"]
    data["Output"] = data["attack_output"] | data["support_output"]
    return data

@st.cache_data
def Arg_dataloader(file_list):
    dfs = [pd.read_excel(file) for file in file_list]
    data = pd.concat(dfs, ignore_index=True)
    print("You have selected {} arguments".format(len(data)))
    data = data[(data["connection"].values == "RA")|(data["connection"].values == "CA")]
    data["Support"] = (data["connection"].values == "RA")
    data["Attack"] = (data["connection"].values == "CA")
    data["Same speaker"] = data["same_speakers"]
    data["Different speakers"] = data["different_speakers"]
    return data


def Comparative_Moral_Foundation_Word_Cloud(df,category_type,valence_type):
        if valence_type == "Positive":
            valence_type = "+"
        else:
            valence_type = "-"
        word_type_ = category_type + valence_type
        if word_type_ == "Care+":
            word_type = "care.virtue"
        elif word_type_ == "Fairness+":
            word_type = "fairness.virtue"
        elif word_type_ == "Loyalty+":
            word_type = "loyalty.virtue"
        elif word_type_ == "Authority+":
            word_type = "authority.virtue"
        elif word_type_ == "Sanctity+":
            word_type = "sanctity.virtue"
        elif word_type_ == "Care-":
            word_type = "care.vice"
        elif word_type_ == "Fairness-":
            word_type = "fairness.vice"
        elif word_type_ == "Loyalty-":
            word_type = "loyalty.vice"
        elif word_type_ == "Authority-":
            word_type = "authority.vice"
        elif word_type_ == "Sanctity-":
            word_type = "sanctity.vice"

        st.write("##### 1. Shared {} Word Cloud".format(word_type_))
        f3 = Compare_Shared_Words_Frequency(data=df, word_type=word_type)
        if len(f3) != 0:
            wordcloud = Word_Cloud(f3.reset_index(), word_type)
        else:
            st.info(
                "No shared {} words".format(word_type_.lower()))

        st.write("##### 2. Shared {} Words Frequency".format(word_type_))
        if len(f3) != 0:
            st.table(f3.astype(str))
            st.write(" Shared {} Word Frequency (Total) is {}".format(word_type_, f3.sum(axis=1).sum()))
        else:
            st.info(
                "No shared {} words".format(word_type_.lower()))

        st.write("##### 3. Unique {} Word Frequency".format(word_type_))
        f4, index = Compare_Unique_Words_Frequency(data=df, word_type=word_type)
        for i in index:
            if f4[i].sum() != 0:
                st.table(f4[i].astype(str))
                st.write(" Unique {} Word Frequency (Total) in *{}* is {}".format(word_type_, i, f4[i].sum()))

        return None






def word_cloud_ADU_module():
    with st.chat_message("assistant"):
        st.write(
            "Awesome! Welcome to the *****Word Cloud feature*****. Please **choose the analytical angle for ADUs and Morals**.")
        col1, col_, col2 = st.columns([3,1,3])
        col1.write("#### ADU Property")
        col2.write("#### Moral Dimension")
        with col1:
            ADUsel = st.multiselect("Arg/Non Arg 👇", ["Arg-ADUs", "Nonarg-ADUs"],default=["Arg-ADUs"],placeholder="Choose one or more options")
            if "Arg-ADUs" in ADUsel:
                polaritysel = st.multiselect("Support/Attack 👇", ["Support", "Attack"],placeholder="Choose one or more options")
                structuresel = st.multiselect("Input/Output 👇", ["Input", "Output"],placeholder="Choose one or more options")
            else:
                polaritysel = []
                structuresel = []

        with col2:
                MFsel = st.selectbox("Moral Foundation 👇", ["Care", "Fairness", "Loyalty", "Authority", "Sanctity"])
                sentisel = st.selectbox("Moral Valence 👇", ["Positive", "Negative"])

        if (len(ADUsel)!=0) & (len(polaritysel)!=0) & (len(structuresel)!=0):
            if "Nonarg-ADUs" in ADUsel:
                index2 = (data[polaritysel].sum(axis=1)>0)
                index3 = (data[structuresel].sum(axis=1)>0)
                df = data[(data["Nonarg-ADUs"]) == 1]
                df = pd.concat([df,data[index2&index3]])
                #st.write(df)

            else:
                index2 = (data[polaritysel].sum(axis=1) > 0)
                index3 = (data[structuresel].sum(axis=1) > 0)
                df = data[index2 & index3]
                #st.write(df)
        elif (ADUsel == ["Nonarg-ADUs"]) and (len(polaritysel)==0) and (len(structuresel)==0):
            index1 = (data[ADUsel].sum(axis=1) > 0)
            df = data[index1]
            #st.write(df)
        else:
            df = pd.DataFrame()

    if (len(df) != 0) and (len(MFsel) != 0) and (len(sentisel) != 0):
        with st.chat_message("assistant"):
            st.write(
                "Fantastic!🤗 Following your selections, the analysis outcomes will be displayed below.")
            Comparative_Moral_Foundation_Word_Cloud(df, MFsel, sentisel)
    else:
        st.write(" ")

    return None


def color_scheme_dataset(df, key):
    if len(df["dataset_name"].unique()) == 1:
        m_color = st.color_picker('***Choose the color scheme: {} 👇***'.format(df["dataset_name"].unique()[0]), '#f5ee03', key="{}_00".format(key))     
        palette_map = {df["dataset_name"].unique()[0]:m_color}
    elif len(df["dataset_name"].unique()) == 2:
        m_color = st.color_picker('***Choose the color scheme: {} 👇***'.format(df["dataset_name"].unique()[0]), '#03F525', key="{}_01".format(key)) 
        n_color = st.color_picker('***Choose the color scheme: {} 👇***'.format(df["dataset_name"].unique()[1]), '#f5ee03', key="{}_02".format(key))
        palette_map = {df["dataset_name"].unique()[0]:m_color,df["dataset_name"].unique()[1]:n_color}
    elif len(df["dataset_name"].unique()) == 3:
        k_color = st.color_picker('***Choose the color scheme: {} 👇***'.format(df["dataset_name"].unique()[0]), '#03F525', key="{}_03".format(key)) 
        l_color = st.color_picker('***Choose the color scheme: {} 👇***'.format(df["dataset_name"].unique()[1]), '#f5ee03', key="{}_04".format(key))
        n_color = st.color_picker('***Choose the color scheme: {} 👇***'.format(df["dataset_name"].unique()[2]), '#D35400', key="{}_05".format(key)) 
        palette_map = {df["dataset_name"].unique()[0]:k_color,df["dataset_name"].unique()[1]:l_color,df["dataset_name"].unique()[2]:n_color}
    elif len(df["dataset_name"].unique()) == 4:
        k_color = st.color_picker('***Choose the color scheme: {} 👇***'.format(df["dataset_name"].unique()[0]), '#03F525', key="{}_06".format(key)) 
        l_color = st.color_picker('***Choose the color scheme: {} 👇***'.format(df["dataset_name"].unique()[1]), '#f5ee03', key="{}_07".format(key))
        n_color = st.color_picker('***Choose the color scheme: {} 👇***'.format(df["dataset_name"].unique()[2]), '#D35400', key="{}_08".format(key)) 
        m_color = st.color_picker('***Choose the color scheme: {} 👇***'.format(df["dataset_name"].unique()[3]), '#F503F1', key="{}_09".format(key)) 
        palette_map = {df["dataset_name"].unique()[0]:k_color,df["dataset_name"].unique()[1]:l_color,df["dataset_name"].unique()[2]:n_color,df["dataset_name"].unique()[3]:m_color}
    else: 
        k_color = st.color_picker('***Choose the color scheme: {} 👇***'.format(df["dataset_name"].unique()[0]), '#03F525', key="{}_10".format(key)) 
        l_color = st.color_picker('***Choose the color scheme: {} 👇***'.format(df["dataset_name"].unique()[1]), '#f5ee03', key="{}_11".format(key))
        n_color = st.color_picker('***Choose the color scheme: {} 👇***'.format(df["dataset_name"].unique()[2]), '#D35400', key="{}_12".format(key)) 
        m_color = st.color_picker('***Choose the color scheme: {} 👇***'.format(df["dataset_name"].unique()[3]), '#F503F1', key="{}_13".format(key)) 
        d_color = st.color_picker('***Choose the color scheme: {} 👇***'.format(df["dataset_name"].unique()[4]), '#E29059', key="{}_14".format(key)) 
        palette_map = {df["dataset_name"].unique()[0]:k_color,df["dataset_name"].unique()[1]:l_color,df["dataset_name"].unique()[2]:n_color,df["dataset_name"].unique()[3]:m_color,df["dataset_name"].unique()[4]:d_color}
    return palette_map


def color_scheme(moral_scale, key):
    if moral_scale == 'Moral vs No moral':
        col1,col2 = st.columns(2)
        with col1:
            m_color = st.color_picker('***Choose the color scheme: the morals 👇***', '#f5ee03', key="{}_00".format(key))
        with col2:
            nm_color = st.color_picker('***Change the color scheme: no morals 👇***', '#0393F5',key="{}_01".format(key))        
        palette_map = {"Morals":m_color,
                       "No morals":nm_color}
    elif moral_scale == "2 Moral Valences":
        col1,col2 = st.columns(2)
        with col1:
            p_color = st.color_picker('***Choose the color scheme: Positive 👇***', '#17B965',key="{}_1".format(key))
        with col2:
            n_color = st.color_picker('***Choose the color scheme: Negative 👇***', '#F50307',key="{}_2".format(key))
        palette_map = {"Positive": p_color,
                       "Negative": n_color}
    elif moral_scale == '5 Moral Foundations':
        col1,col2,col3 = st.columns(3)
        with col1:
            c_color = st.color_picker('***Choose the color scheme: Care 👇***', '#76D7C4',key="{}_3".format(key))
            f_color = st.color_picker('***Choose the color scheme: Fairness 👇***', '#5DADE2',key="{}_4".format(key))
        with col2:
            l_color = st.color_picker('***Choose the color scheme: Loyalty 👇***', '#F7DC6F',key="{}_5".format(key))
            a_color = st.color_picker('***Choose the color scheme: Authority 👇***', '#BB8FCE',key="{}_6".format(key))
        with col3:
            s_color = st.color_picker('***Choose the color scheme: Sanctity 👇***', '#D0D3D4',key="{}_7".format(key))
        palette_map = {"care": c_color,
           "fairness": f_color,
           "loyalty": l_color,
           "authority": a_color,
           "sanctity": s_color}
    elif moral_scale == "5 Moral Foundations * 2 Moral Valences":
        col1,col2 = st.columns(2)
        with col1:
            c_color = st.color_picker('***Choose the color scheme: Care+ 👇***', '#76D7C4',key="{}_8".format(key))
            f_color = st.color_picker('***Choose the color scheme: Fairness+ 👇***', '#45B39D',key="{}_9".format(key))
            l_color = st.color_picker('***Choose the color scheme: Loyalty+ 👇***', '#1E8449',key="{}_10".format(key))
            a_color = st.color_picker('***Choose the color scheme: Authority+ 👇***', '#28B463',key="{}_11".format(key))
            s_color = st.color_picker('***Choose the color scheme: Sanctity+ 👇***', '#27AE60',key="{}_12".format(key))
        with col2:
            c_color_ = st.color_picker('***Choose the color scheme: Care- 👇***', '#F1C40F',key="{}_13".format(key))
            f_color_ = st.color_picker('***Choose the color scheme: Fairness- 👇***', '#F5B041',key="{}_14".format(key))
            l_color_ = st.color_picker('***Choose the color scheme: Loyalty- 👇***', '#E74C3C',key="{}_15".format(key))
            a_color_ = st.color_picker('***Choose the color scheme: Authority- 👇***', '#D35400',key="{}_16".format(key))
            s_color_ = st.color_picker('***Choose the color scheme: Sanctity- 👇***', '#EB984E',key="{}_17".format(key))
        palette_map = {"care+": c_color,
                       "care-": c_color_,
                       "fairness+": f_color,
                       "fairness-": f_color_,
                       "loyalty+": l_color,
                       "loyalty-": l_color_,
                       "authority+": a_color,
                       "authority-": a_color_,
                       "sanctity+": s_color,
                       "sanctity-": s_color_}
    elif moral_scale == "care":
        col1,col2 = st.columns(2)
        with col1:
            no_color = st.color_picker('***Choose the color scheme: no care 👇***', '#0963F1',key="{}_21".format(key))
            color_p = st.color_picker('***Choose the color scheme: only care+ 👇***', '#0FC103',key="{}_22".format(key))
        with col2:
            mixed_color = st.color_picker('***Choose the color scheme: mixed care 👇***', '#FFFB08',key="{}_23".format(key))
            color_n = st.color_picker('***Choose the color scheme: only care- 👇***', '#D35400',key="{}_24".format(key))
        palette_map = {
            "no care": no_color,
            "only care+": color_p,
            "mixed care": mixed_color,
            "only care-": color_n}
    elif moral_scale == "fairness":
        col1,col2 = st.columns(2)
        with col1:
            no_color = st.color_picker('***Choose the color scheme: no fairness 👇***', '#0963F1',key="{}_25".format(key))
            color_p = st.color_picker('***Choose the color scheme: only fairness+ 👇***', '#0FC103',key="{}_26".format(key))
        with col2:
            mixed_color = st.color_picker('***Choose the color scheme: mixed fairness 👇***', '#FFFB08',key="{}_27".format(key))
            color_n = st.color_picker('***Choose the color scheme: only fairness- 👇***', '#D35400',key="{}_28".format(key))
        palette_map = {
            "no fairness": no_color,
            "only fairness+": color_p,
            "mixed fairness": mixed_color,
            "only fairness-": color_n}
    elif moral_scale == "loyalty":
        col1,col2 = st.columns(2)
        with col1:
            no_color = st.color_picker('***Choose the color scheme: no loyalty 👇***', '#0963F1',key="{}_29".format(key))
            color_p = st.color_picker('***Choose the color scheme: only loyalty+ 👇***', '#0FC103',key="{}_211".format(key))
        with col2:
            mixed_color = st.color_picker('***Choose the color scheme: mixed loyalty 👇***', '#FFFB08',key="{}_212".format(key))
            color_n = st.color_picker('***Choose the color scheme: only loyalty- 👇***', '#D35400',key="{}_213".format(key))
        palette_map = {
            "no loyalty": no_color,
            "only loyalty+": color_p,
            "mixed loyalty": mixed_color,
            "only loyalty-": color_n}
    elif moral_scale == "sanctity":
        col1,col2 = st.columns(2)
        with col1:
            no_color = st.color_picker('***Choose the color scheme: no sanctity 👇***', '#0963F1',key="{}_214".format(key))
            color_p = st.color_picker('***Choose the color scheme: only sanctity+ 👇***', '#0FC103',key="{}_215".format(key))
        with col2:
            mixed_color = st.color_picker('***Choose the color scheme: mixed sanctity 👇***', '#FFFB08',key="{}_216".format(key))
            color_n = st.color_picker('***Choose the color scheme: only sanctity- 👇***', '#D35400',key="{}_217".format(key))
        palette_map = {
            "no sanctity": no_color,
            "only sanctity+": color_p,
            "mixed sanctity": mixed_color,
            "only sanctity-": color_n}
    elif moral_scale == "authority":
        col1,col2 = st.columns(2)
        with col1:
            no_color = st.color_picker('***Choose the color scheme: no authority 👇***', '#0963F1',key="{}_218".format(key))
            color_p = st.color_picker('***Choose the color scheme: only authority+ 👇***', '#0FC103',key="{}_219".format(key))
        with col2:
            mixed_color = st.color_picker('***Choose the color scheme: mixed authority 👇***', '#FFFB08',key="{}_220".format(key))
            color_n = st.color_picker('***Choose the color scheme: only authority- 👇***', '#D35400',key="{}_221".format(key))
        palette_map = {
            "no authority": no_color,
            "only authority+": color_p,
            "mixed authority": mixed_color,
            "only authority-": color_n}
        
    return palette_map

def moral_value_distribution_ADU_module():
    with st.chat_message("assistant"):
        st.write(
           "Wonderful! Welcome to the *****Moral Foundation Distribution***** feature. Kindly **select the analytical perspective for ADUs** and the **evaluation scale for Morals**.")
        col1, col_, col2 = st.columns([3,1,3])
        col1.write("#### ADU Property")
        with col1:
            ADUsel = st.multiselect("Arg/Non Arg 👇", ["Arg-ADUs", "Nonarg-ADUs"],default="Arg-ADUs",placeholder="Choose one or more options")
            if "Arg-ADUs" in ADUsel:
                polaritysel = st.multiselect("Support/Attack 👇", ["Support", "Attack"],placeholder="Choose one or more options")
                structuresel = st.multiselect("Input/Output 👇", ["Input", "Output"],placeholder="Choose one or more options")
            else:
                polaritysel = list()
                structuresel = list()
        with col2:
            moral_scale = st.radio("Moral scale 👇",['Moral vs No moral',
                                         '2 Moral Valences',
                                         '5 Moral Foundations',
                                         '5 Moral Foundations * 2 Moral Valences'],key="moral_scale_mv")
            n_format = st.radio("Numerical representations 👇",['number','percentage'],key="moral_scale_nformat")
        if (len(ADUsel)!=0) & (len(polaritysel)!=0) & (len(structuresel)!=0):
            if "Nonarg-ADUs" in ADUsel:
                index2 = (data[polaritysel].sum(axis=1)>0)
                index3 = (data[structuresel].sum(axis=1)>0)
                df = data[(data["Nonarg-ADUs"]) == 1]
                df = pd.concat([df,data[index2&index3]])
                #st.write(df)

            else:
                index2 = (data[polaritysel].sum(axis=1) > 0)
                index3 = (data[structuresel].sum(axis=1) > 0)
                df = data[index2 & index3]
        elif (ADUsel == ["Nonarg-ADUs"]) and (len(polaritysel)==0) and (len(structuresel)==0):
            index1 = (data[ADUsel].sum(axis=1) > 0)
            df = data[index1]
            #st.write(df)
        else:
            df = pd.DataFrame()
    if len(df)!=0:
        with st.chat_message("assistant"):
            st.write("Fantastic!🤗 Following your selections, the analysis outcomes will be displayed below. Remember, you can always modify the presentation format of the statistics.")
            with st.expander("***Analysis Result***",expanded=True):
                tab1,tab2,tab3 = st.tabs(["Grouped Bar Chart","Layered Bar Chart","Statistic Table"])
                with tab1:
                    num_scale = st.radio("***Axis scale 👇***", ("linear", "log"),key="tab1",horizontal=True)
                    key = "graphic"
                    palette_map = color_scheme(moral_scale, key)
                    width = st.slider('Figure width 👇', 0, 2000, 550)
                    height = st.slider('Figure height 👇', 0, 2000, 550)
                    if n_format == "percentage":
                        add_baseline = st.checkbox('Add baseline 👈',key="xxx1")
                    else:
                        add_baseline = False
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        annotation_fontsize = st.number_input('Annotation fontsize 👇', value=13,
                                                              key="adu_annotation_1")
                    with col2:
                        legend_fontsize = st.number_input('Legend fontsize 👇', value=13, key="adu_annotation_2")
                    with col3:
                        tick_fontsize = st.number_input('Tick fontsize 👇', value=13, key="adu_annotation_3")
                    with col4:
                        axis_fontsize = st.number_input('Axis label fontsize 👇', value=13,
                                                        key="adu_annotation_4")
    
                    title_option = st.text_input('please add the picture title 👈', key="adu_annotation_5")
                    customisation = [palette_map,width,height,add_baseline,annotation_fontsize,legend_fontsize,tick_fontsize,axis_fontsize,title_option]
                    fig,table = Compare_Moral_Foundation_Word_In_Tweet_Group(df, n_format, moral_scale,num_scale,customisation)
                    st.plotly_chart(fig)
                    if add_baseline == True:
                        baselines, deviation = Compare_Moral_Foundation_Word_In_Tweet_Group_Deviation(df, n_format, moral_scale,num_scale,customisation)
                        Metric_description(baselines, deviation, "Word_type", "dataset_name","Percentage","deviation")
    
                    
                with tab2:
                    num_scale = st.radio("***Axis scale 👇***", ("linear", "log"),key="tab2",horizontal=True)
                    key="adu_layer"
                    palette_map = color_scheme_dataset(data,key)
                    width = st.slider('Figure width 👇', 0, 2000, 550, key="{}_1".format(key))
                    height = st.slider('Figure height 👇', 0, 2000, 550, key="{}_2".format(key))
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        annotation_fontsize = st.number_input('Annotation fontsize 👇', value=13,
                                                              key="adu_annotation_11")
                    with col2:
                        legend_fontsize = st.number_input('Legend fontsize 👇', value=13, key="adu_annotation_22")
                    with col3:
                        tick_fontsize = st.number_input('Tick fontsize 👇', value=13, key="adu_annotation_33")
                    with col4:
                        axis_fontsize = st.number_input('Axis label fontsize 👇', value=13,
                                                        key="adu_annotation_44")
    
                    title_option = st.text_input('please add the picture title 👈', key="adu_annotation_55")
                    customisation = [palette_map,width,height,annotation_fontsize,legend_fontsize,tick_fontsize,axis_fontsize,title_option]
                    fig = Compare_Moral_Foundation_Word_In_Tweet_Layer(df, n_format, moral_scale,num_scale,customisation)
                    st.plotly_chart(fig)
                with tab3:
                    st.dataframe(table)
                    csv = convert_df(table)
    
                    st.download_button(
                        label="Download data as CSV",
                        data=csv,
                        file_name='comparative_moral_value_distribution.csv',
                        mime='text/csv',
                    )
    else:
        st.write(" ")

    return None


def user_distribution_ADU_module():
    with st.chat_message("assistant"):
        st.write(
            "Awesome! Welcome to the *****Interlocutors Distribution feature*****. Please **choose the Moral Foundations**.")
        col1, col_, col2 = st.columns([3,1,3])
        col1.write("#### Moral Foundations")
        with col1:
            MFsel = st.selectbox("Moral Foundation 👇", ["all","care", "fairness", "loyalty", "authority", "sanctity"])
        with col2:
            n_format = st.radio("Numerical representations 👇", ['number', 'percentage'], key="moral_scale_nformat")
        df = data

    if len(df)!=0:
        with st.chat_message("assistant"):
            st.write(
                "Fantastic!🤗 Following your selections, the analysis outcomes will be displayed below.")
            if MFsel == "all":
                p_width = st.slider("Adjust the width 👇", 1, 1000, 500, key="w")
                p_height = st.slider("Adjust the height 👇", 1, 1000, 500, key="h")
                for dataset in df.dataset_name.unique():
                    customisation = [p_width, p_height]
                    fig = Compare_User_Distribution_Heatmap(df, dataset, n_format,customisation)
                    st.plotly_chart(fig)
            else:
                tab1,tab2,tab3 = st.tabs(["Stacked bar chart","Grouped bar chart","Layered bar chart"])
                with tab1:
                    num_scale = st.radio("***Axis scale 👇***",("linear","log"), key="tab1",horizontal=True)
                    key = "user_stack"
                    palette_map = color_scheme(MFsel, key)
                    width = st.slider('Figure width 👇', 0, 2000, 550)
                    height = st.slider('Figure height 👇', 0, 2000, 550)
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        annotation_fontsize = st.number_input('Annotation fontsize 👇', value=13,
                                                              key="stack_annotation_11")
                    with col2:
                        legend_fontsize = st.number_input('Legend fontsize 👇', value=13, key="stack_annotation_22")
                    with col3:
                        tick_fontsize = st.number_input('Tick fontsize 👇', value=13, key="stack_annotation_33")
                    with col4:
                        axis_fontsize = st.number_input('Axis label fontsize 👇', value=13,
                                                        key="stack_annotation_44")
    
                    title_option = st.text_input('please add the picture title 👈', key="stack_annotation_55")
                    customisation = [palette_map,width,height,annotation_fontsize,legend_fontsize,tick_fontsize,axis_fontsize,title_option]
                    fig = Compare_User_Distribution_Stack(df,MFsel,n_format,num_scale,customisation)
                    st.plotly_chart(fig)
                with tab2:
                    num_scale = st.radio("***Axis scale 👇***",("linear","log"), key="tab2",horizontal=True)
                    key = "user_group"
                    palette_map = color_scheme(MFsel, key)
                    width = st.slider('Figure width 👇', 0, 2000, 550, key="group0")
                    height = st.slider('Figure height 👇', 0, 2000, 550, key="group1")
                    if n_format == "percentage":
                        add_baseline = st.checkbox('Add baseline 👈',key="xxx8")
                    else:
                        add_baseline = False
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        annotation_fontsize = st.number_input('Annotation fontsize 👇', value=13,
                                                              key="group_annotation_11")
                    with col2:
                        legend_fontsize = st.number_input('Legend fontsize 👇', value=13, key="group_annotation_22")
                    with col3:
                        tick_fontsize = st.number_input('Tick fontsize 👇', value=13, key="group_annotation_33")
                    with col4:
                        axis_fontsize = st.number_input('Axis label fontsize 👇', value=13,
                                                        key="group_annotation_44")
    
                    title_option = st.text_input('please add the picture title 👈', key="group_annotation_55")
                    customisation = [palette_map,width,height,add_baseline,annotation_fontsize,legend_fontsize,tick_fontsize,axis_fontsize,title_option]
                    fig = Compare_User_Distribution_Group(df,MFsel,n_format,num_scale,customisation)
                    st.plotly_chart(fig)
                    if add_baseline == True:
                        baselines, deviation = Compare_User_Distribution_Deviation(df,MFsel,format,num_scale, customisation)
                        Metric_description(baselines, deviation, "mf", "dataset_name","percentage","deviation")

                with tab3:
                    num_scale = st.radio("***Axis scale 👇***",("linear","log"), key="tab3", horizontal=True)
                    key="user_layer"
                    palette_map = color_scheme_dataset(data,key)
                    width = st.slider('Figure width 👇', 0, 2000, 550, key="{}_1".format(key))
                    height = st.slider('Figure height 👇', 0, 2000, 550, key="{}_2".format(key))
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        annotation_fontsize = st.number_input('Annotation fontsize 👇', value=13,
                                                              key="layer_annotation_11")
                    with col2:
                        legend_fontsize = st.number_input('Legend fontsize 👇', value=13, key="layer_annotation_22")
                    with col3:
                        tick_fontsize = st.number_input('Tick fontsize 👇', value=13, key="layer_annotation_33")
                    with col4:
                        axis_fontsize = st.number_input('Axis label fontsize 👇', value=13,
                                                        key="layer_annotation_44")
    
                    title_option = st.text_input('please add the picture title 👈', key="layer_annotation_55")
                    customisation = [palette_map,width,height,annotation_fontsize,legend_fontsize,tick_fontsize,axis_fontsize,title_option]
                    
                    fig = Compare_User_Distribution_Layer(df,MFsel,n_format,num_scale,customisation)
                    st.plotly_chart(fig)
    else:
        st.write("")
    return None

def Metric_description(baselines, deviation, col1, col2, col3, col4):
    if len(baselines[col1].values)==10:
        cols = st.columns(5)
        for i,name in enumerate(baselines[col1].values[:5]):
            with cols[i]:
                with elements("{}".format(name)):
                    st.metric("{} baseline".format(name), value = round(baselines[baselines[col1] == name][col3].values[0],2))
                    for dataset in deviation[(deviation[col1]==name)][col2].unique():
                        st.metric("{} {}".format(name, dataset), value=round(deviation[(deviation[col1] == name) & (deviation[col2] == dataset)][col3].values[0],2),delta=round(deviation[(deviation[col1] == name) & (deviation[col2] == dataset)][col4].values[0],2))
        for i,name in enumerate(baselines[col1].values[5:]):
            with cols[i]:
                with elements("{}".format(name)):
                    st.metric("{} baseline".format(name), value = round(baselines[baselines[col1] == name][col3].values[0],2))
                    for dataset in deviation[(deviation[col1]==name)][col2].unique():
                        st.metric("{} {}".format(name, dataset), value=round(deviation[(deviation[col1] == name) & (deviation[col2] == dataset)][col3].values[0],2),delta=round(deviation[(deviation[col1] == name) & (deviation[col2] == dataset)][col4].values[0],2))                  
    else:
        cols = st.columns(len(baselines[col1].values))
        for i,name in enumerate(baselines[col1].values):
            with cols[i]:
                with elements("{}".format(name)):
                    st.metric("{} baseline".format(name), value = round(baselines[baselines[col1] == name][col3].values[0],2))
                    for dataset in deviation[(deviation[col1]==name)][col2].unique():
                        st.metric("{} {}".format(name, dataset), value=round(deviation[(deviation[col1] == name) & (deviation[col2] == dataset)][col3].values[0],2),delta=round(deviation[(deviation[col1] == name) & (deviation[col2] == dataset)][col4].values[0],2))
    return None

def user_moral_score_ADU_module():

    df = data
    if len(df)!=0:
        with st.chat_message("assistant"):
            st.write(
                "Awesome! Welcome to the *****Interlocutors Moral Scores feature*****. The analysis present the average moral scores for interlocutors in each constructed dataset.")
            fig = Compare_Average_User_Concern_Score(df)
            st.plotly_chart(fig)
    else:
        st.write("")
    return None

def word_cloud_Arg_module():
    with st.chat_message("assistant"):
        st.write(
            "Awesome! Welcome to the *****Word Cloud feature*****. Please **choose the analytical angle for Arguments and Morals**.")
        col1, col_, col2 = st.columns([3,1,3])
        col1.write("#### Argument Property")
        col2.write("#### Moral Dimension")
        with col1:
            polaritysel = st.multiselect("Support/Attack 👇", ["Support", "Attack"],
                                         placeholder="Choose one or more options")
            speakersel = st.multiselect("Arguments constructed by the same speaker/Different speakers 👇",
                                        ["Same speaker", "Different speakers"],
                                        placeholder="Choose one or more options")
        with col2:
            MFsel = st.selectbox("Moral Foundation 👇", ["Care", "Fairness", "Loyalty", "Authority", "Sanctity"])
            sentisel = st.selectbox("Moral Valence 👇", ["Positive", "Negative"])

        if (len(polaritysel) != 0) and (len(speakersel) != 0):
            index1 = (data[polaritysel].sum(axis=1) > 0)
            index2 = (data[speakersel].sum(axis=1) > 0)
            df = data[index1&index2]
        else:
            df = pd.DataFrame()

    if len(df) != 0 and len(MFsel) != 0 and len(sentisel) != 0:
        with st.chat_message("assistant"):
            st.write(
                "Fantastic!🤗 Following your selections, the analysis outcomes will be displayed below.")
            Comparative_Moral_Foundation_Word_Cloud(df, MFsel, sentisel)
    else:
        st.write(" ")
    return None

def moral_value_distribution_Arg_module():
    with st.chat_message("assistant"):
        st.write(
           "Wonderful! Welcome to the *****Moral Foundation Distribution***** feature. Kindly **select the analytical perspective for Arguments** and the **evaluation scale for Morals**.")
        col1, col_, col2 = st.columns([3,1,3])
        col1.write("#### Argument Property")
        with col1:
            polaritysel = st.multiselect("Support/Attack 👇", ["Support", "Attack"],placeholder="Choose one or more options")
            speakersel = st.multiselect("Arguments constructed by the same speaker/Different speakers 👇", ["Same speaker", "Different speakers"],placeholder="Choose one or more options")
        with col2:
            moral_scale = st.radio("Moral Dimensions 👇", ['Moral vs No moral',
                                        '2 Moral Valences',
                                        '5 Moral Foundations',
                                        '5 Moral Foundations * 2 Moral Valences'], key="moral_scale_mv_arg")
            n_format = st.radio("Numerical representations 👇", ['number', 'percentage'], key="moral_scale_nformat_arg")

        if (len(polaritysel) != 0) and (len(speakersel) != 0):
            index1 = (data[polaritysel].sum(axis=1) > 0)
            index2 = (data[speakersel].sum(axis=1) > 0)
            df = data[index1&index2]
        else:
            df = pd.DataFrame()

    if len(df)!=0:
        with st.chat_message("assistant"):
            st.write("Fantastic!🤗 Following your selections, the analysis outcomes will be displayed below. Remember, you can always modify the presentation format of the statistics.")
            with st.expander("Analysis",expanded=True):
                tab1,tab2,tab3 = st.tabs(["Grouped Bar Chart","Layered Bar Chart","Statistic Table"])
                with tab1:
                    num_scale = st.radio("***Axis scale 👇***",("linear","log"), key="tab1",horizontal=True)
                    key="arg"
                    palette_map = color_scheme(moral_scale, key)
                    width = st.slider('Figure width 👇', 0, 2000, 550)
                    height = st.slider('Figure height 👇', 0, 2000, 550)
                    if n_format == "percentage":
                        add_baseline = st.checkbox('Add baseline 👈',key="xxx4")
                    else:
                        add_baseline = False
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        annotation_fontsize = st.number_input('Annotation fontsize 👇', value=13,
                                                              key="arg_annotation_1")
                    with col2:
                        legend_fontsize = st.number_input('Legend fontsize 👇', value=13, key="arg_annotation_2")
                    with col3:
                        tick_fontsize = st.number_input('Tick fontsize 👇', value=13, key="arg_annotation_3")
                    with col4:
                        axis_fontsize = st.number_input('Axis label fontsize 👇', value=13,
                                                        key="arg_annotation_4")

                    title_option = st.text_input('please add the picture title 👈', key="arg_annotation_5")
                    customisation = [palette_map,width,height,add_baseline,annotation_fontsize,legend_fontsize,tick_fontsize,axis_fontsize,title_option]
                    fig,table = Compare_Moral_Foundation_Word_In_Tweet_Group(df, n_format, moral_scale,num_scale,customisation)
                    st.plotly_chart(fig)
                    if add_baseline == True:
                        baselines, deviation = Compare_Moral_Foundation_Word_In_Tweet_Group_Deviation(df, n_format, moral_scale,num_scale,customisation)
                        Metric_description(baselines, deviation, "Word_type", "dataset_name","Percentage","deviation")
                with tab2:
                    num_scale = st.radio("Axis scale 👇",("linear","log"), key="tab2",horizontal=True)
                    key="arg_layer"
                    palette_map = color_scheme_dataset(data,key)
                    width = st.slider('Figure width 👇', 0, 2000, 550, key="{}_1".format(key))
                    height = st.slider('Figure height 👇', 0, 2000, 550, key="{}_2".format(key))
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        annotation_fontsize = st.number_input('Annotation fontsize 👇', value=13,
                                                              key="arg_annotation_11")
                    with col2:
                        legend_fontsize = st.number_input('Legend fontsize 👇', value=13, key="arg_annotation_22")
                    with col3:
                        tick_fontsize = st.number_input('Tick fontsize 👇', value=13, key="arg_annotation_33")
                    with col4:
                        axis_fontsize = st.number_input('Axis label fontsize 👇', value=13,
                                                        key="arg_annotation_44")
    
                    title_option = st.text_input('please add the picture title 👈', key="arg_annotation_55")
                    customisation = [palette_map,width,height,annotation_fontsize,legend_fontsize,tick_fontsize,axis_fontsize,title_option]
                    fig = Compare_Moral_Foundation_Word_In_Tweet_Layer(df, n_format, moral_scale,num_scale,customisation)
                    st.plotly_chart(fig)
                with tab3:
                    st.dataframe(table)
                    csv = convert_df(table)
    
                    st.download_button(
                        label="Download data as CSV",
                        data=csv,
                        file_name='comparative_moral_value_distribution.csv',
                        mime='text/csv',
                    )
    else:
        st.write(" ")

    return None


def sidebar():
    unit = st.sidebar.radio(
        "Analysis units",
        ("ADU-based", "Relation-based", "Entity-based")
    )

    if unit != "Entity-based":
        module = st.sidebar.radio(
            "Analytics module",
            ("WordCloud", "Moral Foundation Distribution")
        )
    else:
        module = st.sidebar.radio(
            "Analytics module",
            ("Interlocutors Distribution","Interlocutors' Moral Foundation Scores",)
        )
    return unit, module

with st.chat_message("assistant"):
    st.write("Hi there! 👋 Excited to see your enthusiasm for comparative corpora analysis. Ready to dive in? Choose from the datasets listed below to kick off your analysis. Note: You can define up to 5 datasets simultaneously.")
    col1,col2,col3,col4,col5 = st.columns([4,4,4,4,4])
    with col1:
        dataset1_on = st.toggle("📂Dataset1")
    with col2:
        dataset2_on = st.toggle("📂Dataset2")
    with col3:
        dataset3_on = st.toggle("📂Dataset3")
    with col4:
        dataset4_on = st.toggle("📂Dataset4")
    with col5:
        dataset5_on = st.toggle("📂Dataset5")

    if dataset1_on:
        col1, col2,col3 = st.columns([1, 4, 2])
        with col1:
            st.write("##### Dataset1")
        with col2:
            selection1 = select_corpora(i=1)
        with col3:
            title1 = st.text_input('You can define the name for the collected dataset', 'Dataset1')
        st.divider()
    else:
        selection1 = dict()
        selection1["checked"] = []
        title1 = 'Dataset1'
    if dataset2_on:
        col1, col2, col3 = st.columns([1, 4, 2])
        with col1:
            st.write("##### Dataset2")
        with col2:
            selection2 = select_corpora(i=2)
        with col3:
            title2 = st.text_input('You can define the name for the collected dataset', 'Dataset2')
        st.divider()
    else:
        selection2 = dict()
        selection2["checked"] = []
        title2 = 'Dataset2'
    if dataset3_on:
        col1, col2, col3 = st.columns([1, 4, 2])
        with col1:
            st.write("##### Dataset3")
        with col2:
            selection3 = select_corpora(i=3)
        with col3:
            title3 = st.text_input('You can define the name for the collected dataset', 'Dataset3')
        st.divider()
    else:
        selection3 = dict()
        selection3["checked"] = []
        title3 = 'Dataset3'
    if dataset4_on:
        col1, col2, col3 = st.columns([1, 4, 2])
        with col1:
            st.write("##### Dataset4")
        with col2:
            selection4 = select_corpora(i=4)
        with col3:
            title4 = st.text_input('You can define the name for the collected dataset', 'Dataset4')
        st.divider()
    else:
        selection4 = dict()
        selection4["checked"] = []
        title4 = 'Dataset4'
    if dataset5_on:
        col1, col2, col3 = st.columns([1, 4, 2])
        with col1:
            st.write("##### Dataset5")
        with col2:
            selection5 = select_corpora(i=5)
        with col3:
            title5 = st.text_input('You can define the name for the collected dataset', 'Dataset5')
        st.divider()
    else:
        selection5 = dict()
        selection5["checked"] = []
        title5 = 'Dataset5'

    unit, module = sidebar()


adu_filename_map = {
                "US2016r1D":"data/US2016rD1_ADU_Moral.xlsx",
                "US2016r1G":"data/US2016rG1_ADU_Moral.xlsx",
                "US2016r1R":"data/US2016rR1_ADU_Moral.xlsx",
                "US2016tvD":"data/US2016D1tv_ADU_Moral.xlsx",
                "US2016tvG":"data/US2016G1tv_ADU_Moral.xlsx",
                "US2016tvR":"data/US2016R1tv_ADU_Moral.xlsx",
                "British Empire":"data/BritishEmpire_ADU_Moral.xlsx",
                "DDay":"data/DDay_ADU_Moral.xlsx",
                "Hypocrisy":"data/Hypocrisy_ADU_Moral.xlsx",
                "Money":"data/Money_ADU_Moral.xlsx",
                "Welfare":"data/Welfare_ADU_Moral.xlsx"}

arg_filename_map = {
                "US2016r1D":"data/US2016rD1_Arg_Moral.xlsx",
                "US2016r1G":"data/US2016rG1_Arg_Moral.xlsx",
                "US2016r1R":"data/US2016rR1_Arg_Moral.xlsx",
                "US2016tvD":"data/US2016D1tv_Arg_Moral.xlsx",
                "US2016tvG":"data/US2016G1tv_Arg_Moral.xlsx",
                "US2016tvR":"data/US2016R1tv_Arg_Moral.xlsx",
                "British Empire":"data/BritishEmpire_Arg_Moral.xlsx",
                "DDay":"data/DDay_Arg_Moral.xlsx",
                "Hypocrisy":"data/Hypocrisy_Arg_Moral.xlsx",
                "Money":"data/Money_Arg_Moral.xlsx",
                "Welfare":"data/Welfare_Arg_Moral.xlsx"}

if unit == "Relation-based":
    selection1_files = [arg_filename_map.get(corpora_name, 'Unknown') for corpora_name in selection1["checked"]]
    selection2_files = [arg_filename_map.get(corpora_name, 'Unknown') for corpora_name in selection2["checked"]]
    selection3_files = [arg_filename_map.get(corpora_name, 'Unknown') for corpora_name in selection3["checked"]]
    selection4_files = [arg_filename_map.get(corpora_name, 'Unknown') for corpora_name in selection4["checked"]]
    selection5_files = [arg_filename_map.get(corpora_name, 'Unknown') for corpora_name in selection5["checked"]]
else:
    selection1_files = [adu_filename_map.get(corpora_name, 'Unknown') for corpora_name in selection1["checked"]]
    selection2_files = [adu_filename_map.get(corpora_name, 'Unknown') for corpora_name in selection2["checked"]]
    selection3_files = [adu_filename_map.get(corpora_name, 'Unknown') for corpora_name in selection3["checked"]]
    selection4_files = [adu_filename_map.get(corpora_name, 'Unknown') for corpora_name in selection4["checked"]]
    selection5_files = [adu_filename_map.get(corpora_name, 'Unknown') for corpora_name in selection5["checked"]]


data_selection = {title1:selection1_files,
                  title2:selection2_files,
                  title3:selection3_files,
                  title4:selection4_files,
                  title5:selection5_files}
if unit == "ADU-based":
    if module == "WordCloud":
        if at_least_number_not_empty(data_selection,2)[0]:
            ##########################
            dataframe_dict = dict()
            for i in at_least_number_not_empty(data_selection,2)[1]:
                dataframe_dict[i] = ADU_dataloader(data_selection[i])
            data = add_datasetname(dataframe_dict)
            #st.write(data)
            word_cloud_ADU_module()
            ##########################
    elif module == "Moral Foundation Distribution":
        if at_least_number_not_empty(data_selection,1)[0]:
            ###########################
            dataframe_dict = dict()
            for i in at_least_number_not_empty(data_selection,1)[1]:
                dataframe_dict[i] = ADU_dataloader(data_selection[i])
            data = add_datasetname(dataframe_dict)
            moral_value_distribution_ADU_module()
            ###########################
elif unit == "Relation-based":
    if module == "WordCloud":
        if at_least_number_not_empty(data_selection,2)[0]:
            dataframe_dict = dict()
            for i in at_least_number_not_empty(data_selection, 2)[1]:
                dataframe_dict[i] = Arg_dataloader(data_selection[i])
            data = add_datasetname(dataframe_dict)
            word_cloud_Arg_module()
    elif module == "Moral Foundation Distribution":
        if at_least_number_not_empty(data_selection,1)[0]:
            dataframe_dict = dict()
            for i in at_least_number_not_empty(data_selection, 1)[1]:
                dataframe_dict[i] = Arg_dataloader(data_selection[i])
            data = add_datasetname(dataframe_dict)
            moral_value_distribution_Arg_module()
else:
    if module == "Interlocutors Distribution":
        if at_least_number_not_empty(data_selection,1)[0]:
            dataframe_dict = dict()
            for i in at_least_number_not_empty(data_selection, 1)[1]:
                dataframe_dict[i] = ADU_dataloader(data_selection[i])
            data = add_datasetname(dataframe_dict)
            user_distribution_ADU_module()
    elif module == "Interlocutors' Moral Foundation Scores":
        if at_least_number_not_empty(data_selection,1)[0]:
            dataframe_dict = dict()
            for i in at_least_number_not_empty(data_selection, 1)[1]:
                dataframe_dict[i] = ADU_dataloader(data_selection[i])
            data = add_datasetname(dataframe_dict)
            user_moral_score_ADU_module()

















