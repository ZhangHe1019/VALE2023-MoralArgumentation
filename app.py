import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import re
from PIL import Image
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from math import log10
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.patches import Patch
import networkx as nx

import matplotlib.pyplot as plt
global df

# -*- coding: utf-8 -*-

sns.set_theme(style="white")
plt.style.use("seaborn-talk")
st.set_option('deprecation.showPyplotGlobalUse', False)


# functions

def Word_Cloud(df, category):
    unique_words = list(set(",".join(df[category.lower()].dropna().values).split(",")))
    if len(unique_words) == 1 and len(unique_words[0]) == 0:
        f = st.info("No word cloud because of no {} in this corpus".format(category.lower()))
    else:
        f = st.pyplot(Make_Word_Cloud(unique_words))
    return f


def Make_Word_Cloud(lexicon):
    wordcloud = WordCloud(background_color="#493E38", colormap='YlOrRd', width=1500, height=800,
                          normalize_plurals=False).generate(" ".join(lexicon))
    fig, ax = plt.subplots(figsize=(10, 10), facecolor=None)
    ax.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    return fig


def Find_MFW(data, word_type):
    if len(data[word_type.lower()].dropna()) != 0:
        df = pd.DataFrame(list(",".join(data[word_type.lower()].dropna().values).split(",")), columns=["words"])
        df = df["words"].value_counts().reset_index()
        df.columns = ["words", "frequency"]
    else:
        df = pd.DataFrame()
    return df


def Top_10_MFW(MFW_dataframe,with_slash):
    if with_slash == True:
        df1 = MFW_dataframe.copy()
        df1["words"] = df1.groupby(["frequency"])["words"].transform(lambda x: '/'.join(x))
        df1["percentage%"] = round(100 * (df1["frequency"] / MFW_dataframe["frequency"].sum()), 2)
        df1 = df1.drop_duplicates()[:10]
        # top 10 offensive language words
        df1 = df1.set_index(np.arange(1, len(df1) + 1, 1))
    else:
        df1 = MFW_dataframe.copy()
        df1["words"] = df1.groupby(["frequency"])["words"].transform(lambda x: '/'.join(x))
        df1["percentage%"] = round(100 * (df1["frequency"] / MFW_dataframe["frequency"].sum()), 2)
        df1 = df1.drop_duplicates()[:10]
        # top 10 offensive language words
        top_10 = df1['words'].str.split('/', expand=True)
        top_10 = top_10.stack().reset_index(level=1, drop=True)
        top_10.name = "words"
        df1 = df1.drop(["words"], axis=1).join(top_10)
        df1 = df1.set_index(np.arange(1, len(df1) + 1, 1))

    return df1


def Total_Word_Frequency(data, category):
    if category == "care.virtue":
        contains_type = "contains_care_virtue"
        num_type = "num care.virtue"
    elif category == "care.vice":
        contains_type = "contains_care_vice"
        num_type = "num care.vice"
    elif category == "fairness.virtue":
        contains_type = "contains_fairness_virtue"
        num_type = "num fairness.virtue"
    elif category == "fairness.vice":
        contains_type = "contains_fairness_vice"
        num_type = "num fairness.vice"
    elif category == "loyalty.virtue":
        contains_type = "contains_loyalty_virtue"
        num_type = "num loyalty.virtue"
    elif category == "loyalty.vice":
        contains_type = "contains_loyalty_vice"
        num_type = "num loyalty.vice"
    elif category == "authority.virtue":
        contains_type = "contains_authority_virtue"
        num_type = "num authority.virtue"
    elif category == "authority.vice":
        contains_type = "contains_authority_vice"
        num_type = "num authority.vice"
    elif category == "sanctity.virtue":
        contains_type = "contains_sanctity_virtue"
        num_type = "num sanctity.virtue"
    elif category == "sanctity.vice":
        contains_type = "contains_sanctity_vice"
        num_type = "num sanctity.vice"
    Total_Frequency = data.groupby([contains_type])[num_type].sum().reset_index()
    Total_Frequency.columns = ["words", "frequency"]
    Total_Frequency["words"] = Total_Frequency["words"].map({1: "Total"})
    Total_Frequency = Total_Frequency.dropna()
    return Total_Frequency


def Texts_With_MFW(data, category, possible_word, word_selection):
    if category == "care.virtue":
        num_type = "num care.virtue"
    elif category == "care.vice":
        num_type = "num care.vice"
    elif category == "fairness.virtue":
        num_type = "num fairness.virtue"
    elif category == "fairness.vice":
        num_type = "num fairness.vice"
    elif category == "loyalty.virtue":
        num_type = "num loyalty.virtue"
    elif category == "loyalty.vice":
        num_type = "num loyalty.vice"
    elif category == "authority.virtue":
        num_type = "num loyalty.virtue"
    elif category == "authority.vice":
        num_type = "num authority.vice"
    elif category == "sanctity.virtue":
        num_type = "num sanctity.virtue"
    elif category == "sanctity.vice":
        num_type = "num sanctity.vice"
    data = data[data[num_type] != 0]
    df2 = data[category.lower()].str.split(",", expand=True).stack().reset_index(level=1, drop=True)
    df2 = pd.DataFrame(df2, columns=[category.lower()])
    data = data.drop([category.lower()], axis=1).join(df2)
    if word_selection == "all":
        Text = data[["Text",category]].reset_index(drop=True).rename(columns={category:"MF words"})
    else:
        Text = data[data[category.lower()] == possible_word][["Text",category]].reset_index(drop=True).rename(columns={category:"MF words"})
    return Text


def Compare_Shared_Words_Frequency(data, word_type):
    column_name = word_type.lower()

    df1 = data.copy(deep=True)
    df = df1.fillna("nan")
    df = df[df[column_name] != "nan"][["Text", "topic", column_name]]
    df["topic"] = df["topic"].astype(str)
    df = df[[column_name, "Text", "topic"]].apply(lambda x: x.str.split(","), axis=0).explode(column_name)
    df["topic"] = df["topic"].apply(lambda x: "".join(x))
    shared_words = df.groupby(['topic', column_name])["Text"].count().unstack()
    shared_words1 = pd.DataFrame(shared_words.T, columns=shared_words.index,
                                 index=shared_words.columns).dropna().reset_index()
    if ((len(shared_words) == len(options)) and (len(shared_words1) != 0)):
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
    df = df[df[column_name] != "nan"][["Text", "topic", column_name]]
    df = df[[column_name, "Text", "topic"]].apply(lambda x: x.str.split(","), axis=0).explode(column_name)
    df["topic"] = df["topic"].apply(lambda x: "".join(x))
    shared_words = df.groupby(['topic', column_name])["Text"].count().unstack()
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


def add_spacelines(number=2):
    for i in range(number):
        st.write("\n")


def Moral_Foundation_Word_In_Tweet(df, format1):
    df["no moral foundation words"] = (df["loyalty.vice"].isnull().astype(int)) & \
                                      (df["loyalty.virtue"].isnull().astype(int)) & \
                                      (df["authority.vice"].isnull().astype(int)) & \
                                      (df["authority.virtue"].isnull().astype(int)) & \
                                      (df["sanctity.virtue"].isnull().astype(int)) & \
                                      (df["sanctity.vice"].isnull().astype(int)) & \
                                      (df["care.virtue"].isnull().astype(int)) & \
                                      (df["care.vice"].isnull().astype(int)) & \
                                      (df["fairness.virtue"].isnull().astype(int)) & \
                                      (df["fairness.virtue"].isnull().astype(int))
    statistic_df = df[["contains_care_virtue",
                       "contains_care_vice",
                       "contains_fairness_virtue",
                       "contains_fairness_vice",
                       "contains_sanctity_virtue",
                       "contains_sanctity_vice",
                       "contains_authority_virtue",
                       "contains_authority_vice",
                       "contains_loyalty_virtue",
                       "contains_loyalty_vice",
                       "no moral foundation words"]].sum().reset_index().rename(columns={"index": "Words Type",
                                                                                         0: "Number"})
    statistic_df["Words Type"] = statistic_df["Words Type"].map({"contains_care_virtue": "care+",
                                                                 "contains_care_vice": "care-",
                                                                 "contains_fairness_virtue": "fairness+",
                                                                 "contains_fairness_vice": "fairness-",
                                                                 "contains_sanctity_virtue": "sanctity+",
                                                                 "contains_sanctity_vice": "sanctity-",
                                                                 "contains_authority_virtue": "authority+",
                                                                 "contains_authority_vice": "authority-",
                                                                 "contains_loyalty_virtue": "loyalty+",
                                                                 "contains_loyalty_vice": "loyalty-",
                                                                 "no moral foundation words": "no morals"},
                                                                na_action='ignore')
    statistic_df["Percentage"] = 100 * round(statistic_df["Number"] / len(df), 4)
    statistic_df = statistic_df.sort_values(by="Number", ascending=False).reset_index(drop=True)

    if format1 == "number":
        import seaborn as sns
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.barplot(data=statistic_df, y="Words Type", x="Number", palette={"care+": "#76D7C4",
               "care-": "#F1C40F",
               "fairness+": "#45B39D",
               "fairness-": "#F5B041",
               "sanctity+": "#27AE60",
               "sanctity-": "#EB984E",
               "authority+": "#28B463",
               "authority-": "#D35400",
               "loyalty+": "#1E8449",
               "loyalty-": "#E74C3C",
               "no morals": "#85C1E9"},
                ax=ax)
        plt.tick_params(labelsize=15)
        ax.set_xlabel("number", fontsize=15)
        ax.set_ylabel("MF types", fontsize=15)
        ax.set_title("The number of ADUs containing different moral values", fontsize=15)
        for i, j in enumerate(statistic_df["Number"].values.tolist()):
            plt.text(j, i, j, fontsize=12)
    else:
        import seaborn as sns
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.barplot(data=statistic_df, y="Words Type", x="Percentage", palette={"care+": "#76D7C4",
               "care-": "#F1C40F",
               "fairness+": "#45B39D",
               "fairness-": "#F5B041",
               "sanctity+": "#27AE60",
               "sanctity-": "#EB984E",
               "authority+": "#28B463",
               "authority-": "#D35400",
               "loyalty+": "#1E8449",
               "loyalty-": "#E74C3C",
               "no morals": "#85C1E9"},
                ax=ax, )
        plt.tick_params(labelsize=15)
        ax.set_xlabel("percentage", fontsize=15)
        ax.set_ylabel("MF types", fontsize=15)
        ax.set_title("The percentage of ADUs containing different moral values", fontsize=15)
        for i, j in enumerate(statistic_df["Percentage"].values.tolist()):
            plt.text(j, i, "{}%".format(round(j, 2)), fontsize=12)
    return fig


def Compare_Moral_Foundation_Word_In_Tweet_Group(df, format1, valence):
    if valence == "negative":
        column_list = ["contains_care_vice",
                       "contains_fairness_vice",
                       "contains_sanctity_vice",
                       "contains_authority_vice",
                       "contains_loyalty_vice",
                       "topic"]
        column_renamemap = {"contains_care_vice": "care-",
                            "contains_fairness_vice": "fairness-",
                            "contains_sanctity_vice": "sanctity-",
                            "contains_authority_vice": "authority-",
                            "contains_loyalty_vice": "loyalty-"}
        palette_map = {"care-": "#F1C40F",
                       "fairness-": "#F5B041",
                       "sanctity-": "#EB984E",
                       "authority-": "#D35400",
                       "loyalty-": "#E74C3C"}
        pattern = {"care-": "-",
                       "fairness-": "/",
                       "sanctity-": ".",
                       "authority-": "x",
                       "loyalty-": "+"}

    elif valence == "positive":
        column_list = ["contains_care_virtue",
                       "contains_fairness_virtue",
                       "contains_sanctity_virtue",
                       "contains_authority_virtue",
                       "contains_loyalty_virtue",
                       "topic"]
        column_renamemap = {"contains_care_virtue": "care+",
                            "contains_fairness_virtue": "fairness+",
                            "contains_sanctity_virtue": "sanctity+",
                            "contains_authority_virtue": "authority+",
                            "contains_loyalty_virtue": "loyalty+"}

        palette_map = {"care+": "#76D7C4",
               "fairness+": "#45B39D",
               "sanctity+": "#27AE60",
               "authority+": "#28B463",
               "loyalty+": "#1E8449"}
        pattern = {"care+": "-",
                       "fairness+": "/",
                       "sanctity+": ".",
                       "authority+": "x",
                       "loyalty+": "+"}
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
                       "contains_loyalty_vice",
                       "no moral foundation words", "topic"]
        column_renamemap = {"contains_care_virtue": "care+",
                            "contains_care_vice": "care-",
                            "contains_fairness_virtue": "fairness+",
                            "contains_fairness_vice": "fairness-",
                            "contains_sanctity_virtue": "sanctity+",
                            "contains_sanctity_vice": "sanctity-",
                            "contains_authority_virtue": "authority+",
                            "contains_authority_vice": "authority-",
                            "contains_loyalty_virtue": "loyalty+",
                            "contains_loyalty_vice": "loyalty-",
                            "no moral foundation words": "no morals"}
        palette_map = {"care+": "#76D7C4",
                       "care-": "#F1C40F",
                       "fairness+": "#45B39D",
                       "fairness-": "#F5B041",
                       "sanctity+": "#27AE60",
                       "sanctity-": "#EB984E",
                       "authority+": "#28B463",
                       "authority-": "#D35400",
                       "loyalty+": "#1E8449",
                       "loyalty-": "#E74C3C",
                       "no morals": "#85C1E9"}
        pattern = {"care+": "-",
                       "care-": "-",
                       "fairness+": "/",
                       "fairness-": "/",
                       "sanctity+": ".",
                       "sanctity-": ".",
                       "authority+": "x",
                       "authority-": "x",
                       "loyalty+": "+",
                       "loyalty-": "+",
                       "no morals": "|"}
        df["no moral foundation words"] = (df["loyalty.vice"].isnull().astype(int)) & \
                                          (df["loyalty.virtue"].isnull().astype(int)) & \
                                          (df["authority.vice"].isnull().astype(int)) & \
                                          (df["authority.virtue"].isnull().astype(int)) & \
                                          (df["sanctity.virtue"].isnull().astype(int)) & \
                                          (df["sanctity.vice"].isnull().astype(int)) & \
                                          (df["care.virtue"].isnull().astype(int)) & \
                                          (df["care.vice"].isnull().astype(int)) & \
                                          (df["fairness.virtue"].isnull().astype(int)) & \
                                          (df["fairness.virtue"].isnull().astype(int))

    statistic_num = df[column_list].groupby(["topic"]).sum().unstack().reset_index().rename(
        columns={"level_0": "Word_type", 0: "Number"})
    statistic_num["Word_type"] = statistic_num["Word_type"].map(column_renamemap)
    tweet = df.groupby(["topic"])["Text"].count().reset_index().rename(columns={"Text": "Number"})
    statistic_per = round(
        100 * statistic_num.set_index(["topic", "Word_type"]) / tweet.set_index(["topic"]),
        2).reset_index()
    statistic_per = statistic_per.sort_values(by="Number", ascending=False).reset_index(drop=True)
    palette = palette_map
    if format1 == "number":
        fig1 = go.Figure()
        for i, j in palette.items():
            fig1.add_trace(go.Bar(x=statistic_num[statistic_num["Word_type"] == i]["topic"],
                                  y=statistic_num[statistic_num["Word_type"] == i]["Number"],
                                  name=i,
                                  marker_color=j,
                                  hovertemplate="Topic: %{x} <br> Number: %{y}",
                                  texttemplate='%{y}',
                                  textposition='outside',
                                  textfont = {'family': "Times", 'size': [30]},
                                  marker_pattern_shape=pattern[i]))

        fig1.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1),barmode='group',
                           width=1000,
                           height=450,
            font=dict(size=15))
        fig1.update_yaxes(type="log", ticklabelstep=2)
        fig1.update_xaxes(title_text='Topic', categoryorder='category ascending')
        fig1.update_yaxes(title_text='Number (log scale)')
    else:
        fig1 = go.Figure()
        for i, j in palette.items():
            fig1.add_trace(go.Bar(x=statistic_per[statistic_per["Word_type"] == i]["topic"],
                                  y=statistic_per[statistic_per["Word_type"] == i]["Number"],
                                  name=i,
                                  marker_color=j,
                                  hovertemplate="Topic: %{x} <br> Percentage: %{y}%",
                                  texttemplate='%{y}%',
                                  textposition='outside',
                                  textfont = {'family': "Times", 'size': [30]},
                                  marker_pattern_shape=pattern[i]))

        fig1.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1),barmode='group',
                           width=1000,
                           height=450,
            font=dict(size=15))
        fig1.update_yaxes(type="log", ticklabelstep=2)
        fig1.update_xaxes(title_text='Topic', categoryorder='category ascending')
        fig1.update_yaxes(title_text='Percentage (log scale)')
    return fig1


def Compare_Moral_Foundation_Word_In_Tweet_Layer(df, format1, valence):
    if valence == "negative":
        column_list = ["contains_care_vice",
                       "contains_fairness_vice",
                       "contains_sanctity_vice",
                       "contains_authority_vice",
                       "contains_loyalty_vice",
                       "topic"]
        column_renamemap = {"contains_care_vice": "care-",
                            "contains_fairness_vice": "fairness-",
                            "contains_sanctity_vice": "sanctity-",
                            "contains_authority_vice": "authority-",
                            "contains_loyalty_vice": "loyalty-"}
        order_list = ["care-",
                      "fairness-",
                      "loyalty-",
                      "authority-",
                      "sanctity-"]

    elif valence == "positive":
        column_list = ["contains_care_virtue",
                       "contains_fairness_virtue",
                       "contains_sanctity_virtue",
                       "contains_authority_virtue",
                       "contains_loyalty_virtue",
                       "topic"]
        column_renamemap = {"contains_care_virtue": "care+",
                            "contains_fairness_virtue": "fairness+",
                            "contains_sanctity_virtue": "sanctity+",
                            "contains_authority_virtue": "authority+",
                            "contains_loyalty_virtue": "loyalty+"}
        order_list = ["care+",
                      "fairness+",
                      "loyalty+",
                      "authority+",
                      "sanctity+"]

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
                       "contains_loyalty_vice",
                       "no moral foundation words", "topic"]

        order_list = ["care+",
                      "care-",
                      "fairness+",
                      "fairness-",
                      "loyalty+",
                      "loyalty-",
                      "authority+",
                      "authority-",
                      "sanctity+",
                      "sanctity-",
                      "no morals"]

        column_renamemap = {"contains_care_virtue": "care+",
                            "contains_care_vice": "care-",
                            "contains_fairness_virtue": "fairness+",
                            "contains_fairness_vice": "fairness-",
                            "contains_sanctity_virtue": "sanctity+",
                            "contains_sanctity_vice": "sanctity-",
                            "contains_authority_virtue": "authority+",
                            "contains_authority_vice": "authority-",
                            "contains_loyalty_virtue": "loyalty+",
                            "contains_loyalty_vice": "loyalty-",
                            "no moral foundation words": "no morals"}
        df["no moral foundation words"] = (df["loyalty.vice"].isnull().astype(int)) & \
                                          (df["loyalty.virtue"].isnull().astype(int)) & \
                                          (df["authority.vice"].isnull().astype(int)) & \
                                          (df["authority.virtue"].isnull().astype(int)) & \
                                          (df["sanctity.virtue"].isnull().astype(int)) & \
                                          (df["sanctity.vice"].isnull().astype(int)) & \
                                          (df["care.virtue"].isnull().astype(int)) & \
                                          (df["care.vice"].isnull().astype(int)) & \
                                          (df["fairness.virtue"].isnull().astype(int)) & \
                                          (df["fairness.virtue"].isnull().astype(int))

    statistic_num = df[column_list].groupby(["topic"]).sum().unstack().reset_index().rename(
        columns={"level_0": "Word_type", 0: "Number"})
    statistic_num["Word_type"] = statistic_num["Word_type"].map(column_renamemap)
    tweet = df.groupby(["topic"])["Text"].count().reset_index().rename(columns={"Text": "Number"})
    statistic_per = round(
        100 * statistic_num.set_index(["topic", "Word_type"]) / tweet.set_index(["topic"]),
        2).reset_index()
    statistic_per = statistic_per.sort_values(by="Number", ascending=False).reset_index(drop=True)
    if format1 == "number":
        fig1 = go.Figure()
        for i in statistic_num["topic"].unique():
            fig1.add_trace(go.Bar(x=statistic_num[statistic_num["topic"] == i]["Word_type"],
                                  y=statistic_num[statistic_num["topic"] == i]["Number"],
                                  name=i,
                                  hovertemplate="Word type: %{x} <br> Number: %{y}",
                                  texttemplate='%{y}',
                                  textposition='outside'))

        fig1.update_layout(barmode='group',
                           width=1000,
                           height=450,
                           xaxis={'categoryorder': 'array', 'categoryarray': order_list})
        fig1.update_yaxes(type="log", ticklabelstep=2)
        fig1.update_xaxes(title_text='Moral Foundation')
        fig1.update_yaxes(title_text='Number (log scale)')

    else:
        fig1 = go.Figure()
        for i in statistic_per["topic"].unique():
            fig1.add_trace(go.Bar(x=statistic_per[statistic_per["topic"] == i]["Word_type"],
                                  y=statistic_per[statistic_per["topic"] == i]["Number"],
                                  name=i,
                                  hovertemplate="Word type: %{x} <br> Percentage: %{y}%",
                                  texttemplate='%{y}%',
                                  textposition='outside'))

        fig1.update_layout(barmode='group',
                           width=1000,
                           height=450,
                           xaxis={'categoryorder': 'array', 'categoryarray': order_list})
        fig1.update_yaxes(type="log", ticklabelstep=2)
        fig1.update_xaxes(title_text='Moral Foundation')
        fig1.update_yaxes(title_text='Percentage (log scale)')
    return fig1


def Top10_MF_Word_In_Tweet(df, format1):
    def classfunc(x):
        if x in care_virtue:
            label = "care.virtue"
        elif x in care_vice:
            label = "care.vice"
        elif x in loyalty_virtue:
            label = "loyalty.virtue"
        elif x in loyalty_vice:
            label = "loyalty.vice"
        elif x in fairness_virtue:
            label = "fairness.virtue"
        elif x in fairness_vice:
            label = "fairness.vice"
        elif x in sanctity_vice:
            label = "sanctity.vice"
        elif x in sanctity_virtue:
            label = "sanctity.virtue"
        elif x in authority_vice:
            label = "authority.vice"
        else:
            label = "authority.virtue"
        return label

    care_virtue = [i for i in ",".join(df[df["care.virtue"].notnull()]["care.virtue"].values.tolist()).split(",")]
    care_vice = [i for i in ",".join(df[df["care.vice"].notnull()]["care.vice"].values.tolist()).split(",")]
    loyalty_virtue = [i for i in
                      ",".join(df[df["loyalty.virtue"].notnull()]["loyalty.virtue"].values.tolist()).split(",")]
    loyalty_vice = [i for i in ",".join(df[df["loyalty.vice"].notnull()]["loyalty.vice"].values.tolist()).split(",")]
    fairness_virtue = [i for i in
                       ",".join(df[df["fairness.virtue"].notnull()]["fairness.virtue"].values.tolist()).split(",")]
    fairness_vice = [i for i in ",".join(df[df["fairness.vice"].notnull()]["fairness.vice"].values.tolist()).split(",")]
    authority_virtue = [i for i in
                        ",".join(df[df["authority.virtue"].notnull()]["authority.virtue"].values.tolist()).split(",")]
    authority_vice = [i for i in
                      ",".join(df[df["authority.vice"].notnull()]["authority.vice"].values.tolist()).split(",")]
    sanctity_virtue = [i for i in
                       ",".join(df[df["sanctity.virtue"].notnull()]["sanctity.virtue"].values.tolist()).split(",")]
    sanctity_vice = [i for i in ",".join(df[df["sanctity.vice"].notnull()]["sanctity.vice"].values.tolist()).split(",")]
    word_list = care_vice + fairness_vice + care_virtue + fairness_virtue + sanctity_vice + sanctity_virtue + loyalty_vice + loyalty_virtue + authority_vice + authority_virtue
    result = pd.value_counts(word_list).reset_index().rename(columns={"index": "words", 0: "frequency"})
    result = result.groupby(["frequency"])["words"].apply(lambda x: "/".join(x)).reset_index().sort_values(
        by="frequency", ascending=False).reset_index(drop=True)
    result = result[:10]
    result["percentage%"] = 100 * result["frequency"] / len(word_list)
    top_10 = result['words'].str.split('/', expand=True)
    top_10 = top_10.stack().reset_index(level=1, drop=True)
    top_10.name = "words"
    result = result.drop(["words"], axis=1).join(top_10)
    result["category"] = result.words.apply(lambda x: classfunc(x))
    if format1 == "number":
        import plotly.express as px
        fig = px.scatter(result, x="words", y="frequency", color='category', color_discrete_map={
            "care.virtue": "#F86D07",
            "care.vice": "#F7D19F",
            "fairness.virtue": "#079F02",
            "fairness.vice": "#B4FCB2",
            "loyalty.virtue": "#01A195",
            "loyalty.vice": "#C5F9F5",
            "authority.virtue": "#3804B0",
            "authority.vice": "#C1A8F9",
            "sanctity.virtue": "#9C018C",
            "sanctity.vice": "#FCCDF5",
            "no moral foundation words": "#85C1E9"},
                         category_orders={'category': np.sort(result["category"].unique())})
        fig.update_traces(hovertemplate='words: %{x} <br>frequency: %{y}')
        fig.update_layout(barmode='group',
                          width=800,
                          height=400,
                          title={
                              'text': "Top 10 most frequent moral foundation words in {} (frequency)".format(
                                  options[0]),
                              'y': 0.9,
                              'x': 0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'}
                          )
        fig.update_xaxes(title_text='Moral foundation words', categoryorder='total descending')
        fig.update_yaxes(title_text='Frequency')
    else:
        import plotly.express as px
        fig = px.bar(result, x="words", y="percentage%", color='category', color_discrete_map={
            "care.virtue": "#F86D07",
            "care.vice": "#F7D19F",
            "fairness.virtue": "#079F02",
            "fairness.vice": "#B4FCB2",
            "loyalty.virtue": "#01A195",
            "loyalty.vice": "#C5F9F5",
            "authority.virtue": "#3804B0",
            "authority.vice": "#C1A8F9",
            "sanctity.virtue": "#9C018C",
            "sanctity.vice": "#FCCDF5",
            "no moral foundation words": "#85C1E9"},
                     category_orders={'category': np.sort(result["category"].unique())})
        fig.update_traces(hovertemplate='words: %{x} <br>percentage: %{y}%')
        fig.update_layout(barmode='group',
                          width=800,
                          height=400,
                          title={
                              'text': "Top 10 most frequent moral foundation words in {} (percentage)".format(
                                  options[0]),
                              'y': 0.9,
                              'x': 0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'},
                          )
        fig.update_xaxes(title_text='Moral foundation words', categoryorder='total descending')
        fig.update_yaxes(title_text='Percentage')

    return result, fig


def Moral_Concern_In_Social_Network(df, moral_foundation, options):
    if moral_foundation == "care":
        virtue = "contains_care_virtue"
        vice = "contains_care_vice"
        legend_title = ["only care+", "mixed care", "only care-", "no care"]
        color_dict = {"only care+": "green", "only care-": "red", "mixed care": "yellow", "no care": "blue"}
    elif moral_foundation == "fairness":
        virtue = "contains_fairness_virtue"
        vice = "contains_fairness_vice"
        legend_title = ["only fairness+", "mixed fairness", "only fairness-", "no fairness"]
        color_dict = {"only fairness+": "green", "only fairness-": "red", "mixed fairness": "yellow",
                      "no fairness": "blue"}
    elif moral_foundation == "loyalty":
        virtue = "contains_loyalty_virtue"
        vice = "contains_loyalty_vice"
        legend_title = ["only loyalty+", "mixed loyalty", "only loyalty-", "no loyalty"]
        color_dict = {"only loyalty+": "green", "only loyalty-": "red", "mixed loyalty": "yellow",
                      "no loyalty": "blue"}
    elif moral_foundation == "authority":
        virtue = "contains_authority_virtue"
        vice = "contains_authority_vice"
        legend_title = ["only authority+", "mixed authority", "only authority-", "no authority"]
        color_dict = {"only authority+": "green", "only authority-": "red", "mixed authority": "yellow",
                      "no authority": "blue"}
    else:
        virtue = "contains_sanctity_virtue"
        vice = "contains_sanctity_vice"
        legend_title = ["only sanctity+", "mixed sanctity", "only sanctity-", "no sanctity"]
        color_dict = {"only sanctity+": "green", "only sanctity-": "red", "mixed sanctity": "yellow",
                      "no sanctity": "blue"}
    df_user = df.groupby(["speaker"])[["contains_sanctity_virtue", "contains_sanctity_vice",
                                           "contains_loyalty_virtue", "contains_loyalty_vice",
                                           "contains_authority_virtue", "contains_authority_vice",
                                           "contains_care_virtue", "contains_care_vice",
                                           "contains_fairness_virtue", "contains_fairness_vice"]].sum().reset_index()

    connection = [(i, j) for (i, j) in list(zip(df_pair.speaker_premise.values, df_pair.dropna().speaker_conclusion.values))
                  if
                  i != j]

    def color_palette(node):
        if df_user[df_user["speaker"] == node][virtue].values == 0 and \
                df_user[df_user["speaker"] == node][vice].values == 0:
            color = "blue"
        elif df_user[df_user["speaker"] == node][virtue].values > 0 and \
                df_user[df_user["speaker"] == node][vice].values == 0:
            color = "green"
        elif df_user[df_user["speaker"] == node][virtue].values == 0 and \
                df_user[df_user["speaker"] == node][vice].values > 0:
            color = "red"
        else:
            color = "yellow"
        return color

    f = plt.figure(figsize=(20, 20))
    G = nx.DiGraph(directed=True)
    G.add_nodes_from(df_user.dropna().speaker.values.tolist())
    G.add_edges_from(connection)
    color_map = [color_palette(node) for node in G]
    pos = nx.circular_layout(G, scale=2)
    nx.draw_networkx(G,pos,with_labels=True, node_size=100, node_color=color_map)
    for v in set(legend_title):
        plt.scatter([], [], c=color_dict[v], label='{}'.format(v))
    handles, labels = plt.gca().get_legend_handles_labels()
    res = {labels[i]: handles[i] for i in range(len(labels))}
    plt.legend([res[label] for label in legend_title], legend_title, loc="upper left", title="User Types",
               title_fontsize="20", fontsize="20", ncol=4)
    plt.show()


def Compare_Average_User_Concern_Score(df):
    df["care"] = ((df["contains_care_vice"] != 0) | (df["contains_care_virtue"] != 0)).astype(int)
    df["fairness"] = ((df["contains_fairness_vice"] != 0) | (df["contains_fairness_virtue"] != 0)).astype(int)
    df["loyalty"] = ((df["contains_loyalty_vice"] != 0) | (df["contains_loyalty_virtue"] != 0)).astype(int)
    df["authority"] = ((df["contains_authority_vice"] != 0) | (df["contains_authority_virtue"] != 0)).astype(int)
    df["sanctity"] = ((df["contains_sanctity_vice"] != 0) | (df["contains_sanctity_virtue"] != 0)).astype(int)
    df_user = df.groupby(["speaker", "topic"])[
        ["care", "fairness", "loyalty", "authority", "sanctity"]].mean().reset_index()
    df_user = df_user.groupby(["topic"])[["care", "fairness", "loyalty", "authority", "sanctity"]].mean()
    df_user_use = df_user.reset_index()
    import plotly.graph_objects as go

    fig = go.Figure()
    color = ["red", "green","blueviolet","blue","orange"]
    symbol = ['square',"circle-dot","diamond","star","cross"]
    for i, j in enumerate(df_user_use["topic"]):
        fig.add_trace(go.Scatterpolar(r=df_user_use[["care", "fairness", "loyalty", "authority", "sanctity"]].values[i],
                                      theta=["care", "fairness", "loyalty", "authority", "sanctity"],
                                      fill='toself',
                                      line_color=color[i],
                                      name=j,
                                      marker=dict(size=8, symbol=symbol[i])))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 0.5]
            )),
        width=500,
        height=500)
    return fig


def Compare_User_Distribution_Stack(df, moral_distribution, format):
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

    df_user = df.groupby(["speaker", "topic"])[["contains_sanctity_virtue", "contains_sanctity_vice",
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
    df_user = df_user[['topic'] + moral_list]
    df_user_number = df_user.groupby(["topic"])[moral_list].sum()
    df_user_per = 100 * df_user_number.div(df_user_number.sum(axis=1), axis='rows')
    df_user_number = df_user_number.unstack().reset_index().round(2).rename(columns={"level_0": "mf", 0: "number"})
    df_user_per = df_user_per.unstack().reset_index().round(2).rename(columns={"level_0": "mf", 0: "percentage"})

    if format == "number":
        import plotly.graph_objects as go
        mf = df_user_number["mf"].unique()
        fig = go.Figure()
        for i in mf:
            fig.add_trace(go.Bar(
                y=df_user_number[df_user_number["mf"] == i]["topic"].values,
                x=df_user_number[df_user_number["mf"] == i]["number"].values,
                name=i,
                textposition='auto', texttemplate="%{x}",
                orientation="h",
            ))
            fig.update_layout(barmode='stack', width=500, height=500, )
    elif format == "percentage":
        import plotly.graph_objects as go
        mf = df_user_per["mf"].unique()
        fig = go.Figure()
        for i in mf:
            fig.add_trace(go.Bar(
                y=df_user_per[df_user_per["mf"] == i]["topic"].values,
                x=df_user_per[df_user_per["mf"] == i]["percentage"].values,
                name=i,
                textposition='auto', texttemplate="%{x}%",
                orientation="h",
            ))
            fig.update_layout(barmode='stack', width=1000,
                              height=500, )
    return fig


def Compare_User_Distribution_Group(df, moral_distribution, format):
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

    df_user = df.groupby(["speaker", "topic"])[["contains_sanctity_virtue", "contains_sanctity_vice",
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
    df_user = df_user[['topic'] + moral_list]
    df_user_number = df_user.groupby(["topic"])[moral_list].sum()
    df_user_per = 100 * df_user_number.div(df_user_number.sum(axis=1), axis='rows')
    df_user_number = df_user_number.unstack().reset_index().round(2).rename(columns={"level_0": "mf", 0: "number"})
    df_user_per = df_user_per.unstack().reset_index().round(2).rename(columns={"level_0": "mf", 0: "percentage"})

    if format == "number":
        import plotly.graph_objects as go
        mf = df_user_number["mf"].unique()
        fig = go.Figure()
        for i in mf:
            fig.add_trace(go.Bar(
                y=df_user_number[df_user_number["mf"] == i]["topic"].values,
                x=df_user_number[df_user_number["mf"] == i]["number"].values,
                name=i,
                textposition='auto', texttemplate="%{x}",
                orientation="h",
            ))
            fig.update_layout(barmode='group', width=1000, height=500)
    elif format == "percentage":
        import plotly.graph_objects as go
        mf = df_user_per["mf"].unique()
        fig = go.Figure()
        for i in mf:
            fig.add_trace(go.Bar(
                y=df_user_per[df_user_per["mf"] == i]["topic"].values,
                x=df_user_per[df_user_per["mf"] == i]["percentage"].values,
                name=i,
                textposition='auto', texttemplate="%{x}%",
                orientation="h",
            ))
            fig.update_layout(barmode='group', width=1000, height=500)
    return fig


def Compare_User_Distribution_Layer(df, moral_distribution, format):
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

    df_user = df.groupby(["speaker", "topic"])[["contains_sanctity_virtue", "contains_sanctity_vice",
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
    df_user = df_user[['topic'] + moral_list]
    df_user_number = df_user.groupby(["topic"])[moral_list].sum()
    df_user_per = 100 * df_user_number.div(df_user_number.sum(axis=1), axis='rows')
    df_user_number = df_user_number.reset_index().round(2)
    df_user_per = df_user_per.reset_index().round(2)
    if format == "number":
        import plotly.graph_objects as go

        platform_topic = df_user_number.set_index("topic").columns

        fig = go.Figure()
        color = ["red", "green", "yellow", "blueviolet", "orange"]
        for i in np.arange(len(df_user_number)):
            fig.add_trace(go.Bar(
                x=platform_topic,
                y=df_user_number.set_index("topic").iloc[i],
                name=df_user_number["topic"][i],
                marker_color=color[i],
                textposition='auto', texttemplate="%{y}"
            ))
            fig.update_layout(barmode='group', xaxis_tickangle=-45, width=1000, height=500)
    elif format == "percentage":
        import plotly.graph_objects as go

        platform_topic = df_user_per.set_index("topic").columns

        fig = go.Figure()
        color = ["red", "green", "yellow", "blueviolet","orange"]
        for i in np.arange(len(df_user_per)):
            fig.add_trace(go.Bar(
                x=platform_topic,
                y=df_user_per.set_index("topic").iloc[i],
                name=df_user_per["topic"][i],
                marker_color=color[i],
                textposition='auto', texttemplate="%{y}%"
            ))
            fig.update_layout(barmode='group', xaxis_tickangle=-45, width=1000, height=500)
    return fig
    

def MainPage():
    st.title("MorArgAn: Moral Argument Analytics")
    add_spacelines(2)
    st.write("#### Moral Value Detection")
    st.write("###### Lexicon-Based Method")
    with st.expander("Definition"):
        add_spacelines(1)
        st.write("Moral Value Detection in this study makes use of Moral Foundation Words Dictionary. ")
        st.write("The Lexicon consists of:")
        col1, col2, col3 = st.columns([2, 2, 2])
        col2.write("**570 care.virtue words**")
        col2.write("**289 care.vice words**")
        col2.write("**116 fairness.virtue**")
        col2.write("**237 fairness.vice**")
        col2.write("**144 loyalty.virtue**")
        col2.write("**50 loyalty.vice**")
        col2.write("**302 authority.virtue**")
        col2.write("**131 authority.vice**")
        col2.write("**273 sanctity.virtue**")
        col2.write("**389 sanctity.vice**")
        st.write(
            "Referece: **Frimer, J. A., Boghrati, R., Haidt, J., Graham, J., & Dehgani, M. (2019). Moral foundations dictionary for linguistic analyses 2.0. Unpublished manuscript.**")
    add_spacelines(1)

    st.write("#### Moral Value Metrics")
    st.write("###### Sequence Type")
    with st.expander("Definition"):
        add_spacelines(1)
        st.write(""" Drawing from the moral foundation dictionary, 
        we assess the presence of specific moral values within text sequences. 
        Additionally, we've introduced the **No Morals** category to **designate sequences that lack moral values**. 
        Taking into account the five moral foundations, each associated with two valence categories, we distinguish between **10 moral value categories**, 
        plus the category **No Morals** designed for sequences devoid of moral values.""")
    st.write("###### Moral Value Score")
    with st.expander("Definition"):
        add_spacelines(1)
        st.write("""We compute the proportion of sequences that encompass specific moral values across the 10 predefined moral value categories to obtain interlocutors' moral value score.""")
    st.write("###### Moral Valence Degree")
    with st.expander("Definition"):
        add_spacelines(1)
        st.write("""For each moral foundation attributed to an interlocutor within the corpora, we categorise it into one of four moral valence categories: 'only virtue', 'only vice', 'mixed', and 'no specific moral values'.""")
        st.write("Taking the 'care' foundation as an example:")
        st.write("***only care+***: This signifies that when referencing the care foundation, speakers exclusively utilise care virtues in their discourse.")
        st.write("***only care-***: This implies that when mentioning the care foundation, speakers solely incorporate care vices.")
        st.write("***mixed care***: Under this classification, when the care foundation is mentioned, speakers employ both care virtues and vices.")
        st.write("***no care***: Here, speakers abstain from integrating care foundation into their speech.")
    st.write("#### User Manual")
    with open("data/User_Manual.pdf", "rb") as file:
        st.download_button(
            label="Download User Manual",
            data=file,
            file_name="User_Manual.pdf",
            mime="application/pdf"
        )
    with st.container():
        hide_footer_style = """
            <style>
            footer {visibility: visible;
                    color : white;
                    background-color: #d2cdcd;}

            footer:after{
            visibility: visible;
            content : 'Interface developed by He Zhang. For more technical support, please contact zhanghe1019@hotmail.com';
            display : block;
            positive : relative;
            color : white;
            background-color: #d2cdcd;
            padding: 5px;
            top: 3px;
            font-weight: bold;
            }
            </style>
        """
        st.markdown(hide_footer_style, unsafe_allow_html=True)
    st.write('<style>div.row-widget.stRadio > div{flex-direction:column;font-size=18px;}</style>',
             unsafe_allow_html=True)



def Top10_Moral_Foundation_Word():
    st.subheader("Top 10 Most Frequent Moral Foundation Word")
    add_spacelines(2)
    format1 = st.radio("Choose y-aix unit", ("number", "percentage",))
    if len(options) != 0:
        table, fig = Top10_MF_Word_In_Tweet(df, format1)
        st.plotly_chart(fig)
        with st.expander("elaboration"):
            "If you pick number button, you will get the frequency of each word of top 10 most frequent moral foundation words.\n" \
 \
            "If you pick percentage button, you will get the percentage of each word frequency of top 10 most frequent moral foundation words" \
            " base on total moral foundation words frequency."

            "Details can be seen in the table below:"

            st.table(table)
    else:
        st.write("")


def Top_10_Moral_Foundation_Words_Visualisation(df, category_valence, format1):
    palette_map = {"care.virtue": "#76D7C4",
                   "care.vice": "#F1C40F",
                   "fairness.virtue": "#45B39D",
                   "fairness.vice": "#F5B041",
                   "sanctity.virtue": "#27AE60",
                   "sanctity.vice": "#EB984E",
                   "authority.virtue": "#28B463",
                   "authority.vice": "#D35400",
                   "loyalty.virtue": "#1E8449",
                   "loyalty.vice": "#E74C3C",
                   "no moral foundation words": "#85C1E9"}
    df = df.round(2)
    if format1 == "percentage":
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df.words.values,
            y=df["percentage%"].values,
            marker_color=palette_map[category_valence],
            textposition='outside',
            texttemplate="%{y}%"))
        fig.update_layout(barmode='group', xaxis_tickangle=-45)
    elif format1 == "number":
        import plotly.express as px
        fig = px.scatter(df, x="words", y="frequency")
        fig.update_traces(hovertemplate='words: %{x} <br>frequency: %{y}')
        fig.update_layout(barmode='group',
                          width=800,
                          height=400,
                          title={
                              'text': "Top 10 most frequent moral foundation words in {} (frequency)".format(
                                  options[0]),
                              'y': 0.9,
                              'x': 0.5,
                              'xanchor': 'center',
                              'yanchor': 'top'}
                          )
        fig.update_xaxes(title_text='Moral foundation words', categoryorder='total descending')
        fig.update_yaxes(title_text='Frequency')
    return fig


def User_Distribution_Group(df, format):
    df_user = df.groupby(["speaker"])[["contains_sanctity_virtue", "contains_sanctity_vice",
                                           "contains_loyalty_virtue", "contains_loyalty_vice",
                                           "contains_authority_virtue", "contains_authority_vice",
                                           "contains_care_virtue", "contains_care_vice",
                                           "contains_fairness_virtue", "contains_fairness_vice"]].sum().reset_index()
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
    df_user["only authority-"] = (
                (df_user["contains_authority_virtue"] == 0) & (df_user["contains_authority_vice"] > 0)).astype(int)
    df_user["only sanctity-"] = (
                (df_user["contains_sanctity_virtue"] == 0) & (df_user["contains_sanctity_vice"] > 0)).astype(int)
    df_user_mixed = df_user[
        ["speaker", "mixed care", "mixed fairness", "mixed authority", "mixed loyalty", "mixed sanctity"]].set_index(
        ["speaker"]).sum().reset_index().rename(columns={"index": "mixed", 0: "number"}).round(2)
    df_user_vice = df_user[
        ["speaker", "only care-", "only fairness-", "only authority-", "only loyalty-",
         "only sanctity-"]].set_index(["speaker"]).sum().reset_index().rename(
        columns={"index": "vice", 0: "number"}).round(2)
    df_user_virtue = df_user[
        ["speaker", "only care+", "only fairness+", "only authority+", "only loyalty+",
         "only sanctity+"]].set_index(
        ["speaker"]).sum().reset_index().rename(columns={"index": "virtue", 0: "number"}).round(2)
    df_user_no = df_user[
        ["speaker", "no care", "no fairness", "no authority", "no loyalty",
         "no sanctity"]].set_index(["speaker"]).sum().reset_index().rename(columns={"index": "no", 0: "number"}).round(2)
    if format == "number":
        import plotly.graph_objects as go
        moral_foundation = ['care', 'fairness', 'authority', "loyalty", "sanctity"]
        fig = go.Figure(data=[
            go.Bar(name='Only -', x=moral_foundation, y=df_user_vice["number"].values, marker_color='red'),
            go.Bar(name='Mixed', x=moral_foundation, y=df_user_mixed["number"].values, marker_color='yellow'),
            go.Bar(name='Only +', x=moral_foundation, y=df_user_virtue["number"].values, marker_color='Green'),
            go.Bar(name='No morals', x=moral_foundation, y=df_user_no["number"].values,
                   marker_color='Blue')])
        # Change the bar mode
        fig.update_layout(barmode='group', width=1000, height=450)
        fig.update_traces(textposition='outside', texttemplate="%{y}")
        return fig
    else:
        import plotly.graph_objects as go
        moral_foundation = ['care', 'fairness', 'authority', "loyalty", "sanctity"]
        fig = go.Figure(data=[
            go.Bar(name='Only -', x=moral_foundation,
                   y=(100 * df_user_vice["number"].values / len(df_user)).round(2), marker_color='red'),
            go.Bar(name='Mixed', x=moral_foundation, y=(100 * df_user_mixed["number"].values / len(df_user)).round(2),
                   marker_color='yellow'),
            go.Bar(name='Only +', x=moral_foundation,
                   y=(100 * df_user_virtue["number"].values / len(df_user)).round(2), marker_color='Green'),
            go.Bar(name='No morals', x=moral_foundation,
                   y=(100 * df_user_no["number"].values / len(df_user)).round(2), marker_color='Blue')])
        # Change the bar mode
        fig.update_layout(barmode='group', width=1000, height=450)
        fig.update_traces(textposition='outside', texttemplate="%{y}%")
        return fig


def User_Moral_Concern_Score_Heatmap(df):
    df = df.groupby(['speaker'])[["contains_care_virtue",
                       "contains_care_vice",
                       "contains_fairness_virtue",
                       "contains_fairness_vice",
                       "contains_sanctity_virtue",
                       "contains_sanctity_vice",
                       "contains_authority_virtue",
                       "contains_authority_vice",
                       "contains_loyalty_virtue",
                       "contains_loyalty_vice"]].mean()
    import plotly.express as px
    data = df.to_numpy().T.round(2)
    fig = px.imshow(data,
                    labels=dict(y="Moral Foundation", x="Speaker", color="Interlocutor Score"),
                    y=["care+",
                       "care-",
                       "fairness+",
                       "fairness-",
                       "sanctity+",
                       "sanctity-",
                       "authority+",
                       "authority-",
                       "loyalty+",
                       "loyalty-"],
                    x=df.index,text_auto=True,aspect="auto")
    fig.update_xaxes(side="top")
    fig.update_layout(width=1200, height=800)
    return fig


def User_Distribution_Stack(df, format):
    df_user = df.groupby(["speaker"])[["contains_sanctity_virtue", "contains_sanctity_vice",
                                           "contains_loyalty_virtue", "contains_loyalty_vice",
                                           "contains_authority_virtue", "contains_authority_vice",
                                           "contains_care_virtue", "contains_care_vice",
                                           "contains_fairness_virtue", "contains_fairness_vice"]].sum().reset_index()
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
    df_user["only authority-"] = (
                (df_user["contains_authority_virtue"] == 0) & (df_user["contains_authority_vice"] > 0)).astype(int)
    df_user["only sanctity-"] = (
                (df_user["contains_sanctity_virtue"] == 0) & (df_user["contains_sanctity_vice"] > 0)).astype(int)
    df_user_mixed = df_user[
        ["speaker", "mixed care", "mixed fairness", "mixed authority", "mixed loyalty", "mixed sanctity"]].set_index(
        ["speaker"]).sum().reset_index().rename(columns={"index": "mixed", 0: "number"}).round(2)
    df_user_vice = df_user[
        ["speaker", "only care-", "only fairness-", "only authority-", "only loyalty-",
         "only sanctity-"]].set_index(["speaker"]).sum().reset_index().rename(
        columns={"index": "vice", 0: "number"}).round(2)
    df_user_virtue = df_user[
        ["speaker", "only care+", "only fairness+", "only authority+", "only loyalty+",
         "only sanctity+"]].set_index(
        ["speaker"]).sum().reset_index().rename(columns={"index": "virtue", 0: "number"}).round(2)
    df_user_no = df_user[
        ["speaker", "no care", "no fairness", "no authority", "no loyalty",
         "no sanctity"]].set_index(["speaker"]).sum().reset_index().rename(columns={"index": "no", 0: "number"}).round(2)
    if format == "number":
        import plotly.graph_objects as go
        moral_foundation = ['care', 'fairness', 'authority', "loyalty", "sanctity"]
        fig = go.Figure(data=[
            go.Bar(name='Only -', y=moral_foundation, x=df_user_vice["number"].values, orientation='h',
                   marker_color='red'),
            go.Bar(name='Mixed', y=moral_foundation, x=df_user_mixed["number"].values, orientation='h',
                   marker_color='yellow'),
            go.Bar(name='Only +', y=moral_foundation, x=df_user_virtue["number"].values, orientation='h',
                   marker_color='Green'),
            go.Bar(name='No morals', y=moral_foundation, x=df_user_no["number"].values, orientation='h',
                   marker_color='Blue')])
        # Change the bar mode
        fig.update_layout(barmode='stack', width=1000, height=450)
        return fig
    else:
        import plotly.graph_objects as go
        moral_foundation = ['care', 'fairness', 'authority', "loyalty", "sanctity"]
        fig = go.Figure(data=[
            go.Bar(name='Only -', y=moral_foundation, x=100 * df_user_vice["number"].values / len(df_user),
                   orientation='h', marker_color='red'),
            go.Bar(name='Mixed', y=moral_foundation, x=100 * df_user_mixed["number"].values / len(df_user),
                   orientation='h', marker_color='yellow'),
            go.Bar(name='Only +', y=moral_foundation, x=100 * df_user_virtue["number"].values / len(df_user),
                   orientation='h', marker_color='Green'),
            go.Bar(name='No morals', y=moral_foundation,
                   x=100 * df_user_no["number"].values / len(df_user), orientation='h', marker_color='Blue')])
        # Change the bar mode
        fig.update_layout(barmode='stack', width=1000, height=450)
        return fig


def User_Distribution_Layered(df, format):
    df_user = df.groupby(["speaker"])[["contains_sanctity_virtue", "contains_sanctity_vice",
                                           "contains_loyalty_virtue", "contains_loyalty_vice",
                                           "contains_authority_virtue", "contains_authority_vice",
                                           "contains_care_virtue", "contains_care_vice",
                                           "contains_fairness_virtue", "contains_fairness_vice"]].sum().reset_index()
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
    df_user["only authority-"] = (
                (df_user["contains_authority_virtue"] == 0) & (df_user["contains_authority_vice"] > 0)).astype(int)
    df_user["only sanctity-"] = (
                (df_user["contains_sanctity_virtue"] == 0) & (df_user["contains_sanctity_vice"] > 0)).astype(int)
    df_user_mixed = df_user[
        ["speaker", "mixed care", "mixed fairness", "mixed authority", "mixed loyalty", "mixed sanctity"]].set_index(
        ["speaker"]).sum().reset_index().rename(columns={"index": "mixed", 0: "number"})
    df_user_vice = df_user[
        ["speaker", "only care-", "only fairness-", "only authority-", "only loyalty-",
         "only sanctity-"]].set_index(["speaker"]).sum().reset_index().rename(columns={"index": "vice", 0: "number"})
    df_user_virtue = df_user[
        ["speaker", "only care+", "only fairness+", "only authority+", "only loyalty+",
         "only sanctity+"]].set_index(
        ["speaker"]).sum().reset_index().rename(columns={"index": "virtue", 0: "number"})
    df_user_no = df_user[
        ["speaker", "no care", "no fairness", "no authority", "no loyalty",
         "no sanctity"]].set_index(["speaker"]).sum().reset_index().rename(columns={"index": "no", 0: "number"})
    if format == "number":
        import plotly.graph_objects as go
        df_user_new = pd.DataFrame()
        df_user_new["moral foundation"] = ['care', 'fairness', 'authority', "loyalty", "sanctity"]
        df_user_new["Only -"] = (df_user_vice["number"].values).round(2)
        df_user_new["mixed"] = (df_user_mixed["number"].values).round(2)
        df_user_new["Only +"] = (df_user_virtue["number"].values).round(2)
        df_user_new["No morals"] = (df_user_no["number"].values).round(2)
        df_user_new = df_user_new.set_index("moral foundation").unstack().reset_index().rename(
            columns={"level_0": "user type", 0: "number"})
        import plotly.express as px
        fig = px.bar(df_user_new,
                     y="moral foundation",
                     x="number",
                     color="moral foundation",
                     barmode="group",
                     facet_col="user type",
                     color_discrete_map={"care": "#F86D07",
                                         "fairness": "#079F02",
                                         "loyalty": "#01A195",
                                         "authority": "#3804B0",
                                         "sanctity": "#9C018C"},
                     orientation='h')
        fig.update_layout(barmode='group', width=1000, height=450, legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        fig.update_traces(textposition='auto', texttemplate="%{x}", width=0.5)
        return fig
    else:
        import plotly.express as px
        df_user_new = pd.DataFrame()
        df_user_new["moral foundation"] = ['care', 'fairness', 'authority', "loyalty", "sanctity"]
        df_user_new["Only -"] = (100 * df_user_vice["number"].values / len(df_user)).round(2)
        df_user_new["mixed"] = (100 * df_user_mixed["number"].values / len(df_user)).round(2)
        df_user_new["Only +"] = (100 * df_user_virtue["number"].values / len(df_user)).round(2)
        df_user_new["No morals"] = (100 * df_user_no["number"].values / len(df_user)).round(2)
        df_user_new = df_user_new.set_index("moral foundation").unstack().reset_index().rename(
            columns={"level_0": "user type", 0: "percentage"})
        fig = px.bar(df_user_new,
                     y="moral foundation",
                     x="percentage",
                     color="moral foundation",
                     barmode="group",
                     facet_col="user type",
                     color_discrete_map={"care": "#F86D07",
                                         "fairness": "#079F02",
                                         "loyalty": "#01A195",
                                         "authority": "#3804B0",
                                         "sanctity": "#9C018C"},
                     orientation='h')
        fig.update_layout(barmode='stack', width=1000, height=450, legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        fig.update_traces(textposition='auto', texttemplate="%{x}%", width=0.5)
        return fig


def User_Interaction_Analysis(df, aspect, moral_foundation):
    if moral_foundation == "care":
        columns_name = ["no care", "only care-", "mixed care", "only care+"]
    elif moral_foundation == "fairness":
        columns_name = ["no fairness", "only fairness-", "mixed fairness", "only fairness+"]
    elif moral_foundation == "authority":
        columns_name = ["no authority", "only authority-", "mixed authority", "only authority+"]
    elif moral_foundation == "loyalty":
        columns_name = ["no loyalty", "only loyalty-", "mixed loyalty", "only loyalty+"]
    else:
        columns_name = ["no sanctity", "only sanctity-", "mixed sanctity", "only sanctity+"]

    df_user = df.groupby(["speaker"])[["contains_sanctity_virtue", "contains_sanctity_vice",
                                           "contains_loyalty_virtue", "contains_loyalty_vice",
                                           "contains_authority_virtue", "contains_authority_vice",
                                           "contains_care_virtue", "contains_care_vice",
                                           "contains_fairness_virtue", "contains_fairness_vice"]].sum().reset_index()
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
    df_user["only authority-"] = (
            (df_user["contains_authority_virtue"] == 0) & (df_user["contains_authority_vice"] > 0)).astype(int)
    df_user["only sanctity-"] = (
            (df_user["contains_sanctity_virtue"] == 0) & (df_user["contains_sanctity_vice"] > 0)).astype(int)
    df_user_care = df_user[["speaker"] + columns_name]
    df_user_care = df_user_care.set_index("speaker").stack().reset_index().rename(
        columns={"level_1": "care", 0: "selected"})
    df_user_care = df_user_care[df_user_care["selected"] == 1].reset_index(drop=True)[["speaker", "care"]]
    user_dict = dict([(index, category) for index, category in zip(df_user_care.speaker, df_user_care.care)])
    moral_dict = dict([(j, i) for i, j in enumerate(columns_name)])
    number_dict = dict([(i, j) for i, j in enumerate(columns_name)])
    pair = df_pair[["speaker_premise","speaker_conclusion"]].value_counts().reset_index().rename(columns={"count": "frequency"})
    pair["speaker_premise"] = pair.speaker_premise.map(user_dict).map(moral_dict)
    pair["speaker_conclusion"] = pair.speaker_conclusion.map(user_dict).map(moral_dict)
    pair = pair.dropna()
    if aspect == "ADU frequency":
        Table_speaker = pd.pivot_table(pair, index=['speaker_premise', 'speaker_conclusion'], values=['frequency'],
                                      aggfunc="mean").reset_index()
        Table_speaker = Table_speaker.pivot(index='speaker_premise', columns='speaker_conclusion', values='frequency')
        Table_speaker.index = Table_speaker.index.map(number_dict)
        Table_speaker.columns = Table_speaker.columns.map(number_dict)
        Table_speaker_list = np.array(round(Table_speaker, 2)).tolist()
        import plotly.express as px
        data = pd.DataFrame(Table_speaker_list,dtype=object).fillna(0).values.tolist()
        fig = px.imshow(data, labels=dict(x="Input speaker", y="Output speaker", color="average number of ADUs"),
                        x=Table_speaker.columns,
                        y=Table_speaker.index,
                        text_auto=True,
                        color_continuous_scale='greens')
        fig.update_layout(autosize=False, width=500, height=500)
        fig.update_xaxes(tickangle=90,side="top")
    return fig


def Moral_Foundation_Word_Cloud(mode):
    global df
    data = df.copy(deep=True)
    if mode == "ADU":
        st.subheader(f"Moral Foundation Word Cloud")
        if len(options) != 0:
            st.write("#### 1. Word Cloud Visualisation")
            category_type = st.selectbox(
                'choose an moral foundation category',
                ('Care',
                 'Fairness',
                 "Loyalty",
                 "Authority",
                 "Sanctity",))
            valence_type = st.selectbox(
                'choose the valence of moral foundation category',
                ('Positive',
                 'Negative',))
            add_spacelines(1)
            threshold_cloud = st.slider('Select a precision value (threshold) for a WordCloud',
                                        0, 100, 100, disabled=True)
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

            st.info(f'Selected precision: **{threshold_cloud}**')
            f1 = Word_Cloud(df, word_type)
            MFW_dataframe = Find_MFW(data=df, word_type=word_type)
            if len(MFW_dataframe) != 0:
                tab1, tab2 = st.tabs(["Graph", "Table"])
                with tab1:
                    format1 = st.radio("Choose the y-axis unit", ("number", "percentage"))
                    f2 = Top_10_MFW(MFW_dataframe,with_slash=False)
                    fig = Top_10_Moral_Foundation_Words_Visualisation(f2, word_type, format1)
                    st.plotly_chart(fig)
                with tab2:
                    st.write("**Top 10 {} Words Frequency**".format(word_type_))
                    f2 = Top_10_MFW(MFW_dataframe,with_slash=True)
                    st.table(f2)
                    st.write("**Total {} Words Frequency**".format(word_type_))
                    f3 = Total_Word_Frequency(df, word_type)
                    st.table(f3)

            else:
                st.info("No {} in this corpus".format(word_type.lower()))
            add_spacelines(2)
            st.write("#### 2. Qualitative Analysis")
            MFW_dataframe = Find_MFW(data=df, word_type=word_type)
            if len(MFW_dataframe) != 0:
                selectbox_name = MFW_dataframe.words.values.tolist()
                word_selection = st.radio("Choose the number of {} words you want to select".format(word_type_.lower()),
                                          ("single", "all",))
                if word_selection == "single":
                    possible_words = st.selectbox("Choose {} words you want to analyse".format(word_type_.lower()),
                                                  selectbox_name)
                else:
                    possible_words = np.nan
                f5 = Texts_With_MFW(df, word_type, possible_words, word_selection)
                st.table(f5)
            else:
                st.info("No {} words in this corpus".format(word_type_.lower()))

        else:
            st.write("")
    else:
        st.subheader(f"Moral Foundation Word Cloud")
        if len(options) != 0:
            st.write("#### 1. Word Cloud Visualisation")
            st.write('**Elements:**')
            column1, column2, column3, column4 = st.columns([2, 2, 2, 2])
            with column1:
                arg_element1 = st.checkbox("Input", value=False)
            with column2:
                arg_element2 = st.checkbox("Output", value=True)
            with column3:
                arg_element3 = st.checkbox("Initial Input", value=False)
            with column4:
                arg_element4 = st.checkbox("Final Output", value=True)
            arg_option = st.multiselect(
                'Input/Output Relations',
                ['Support', 'Attack'],
                ['Support', 'Attack'])

            if (len(arg_option)==1) and (arg_option[0] == "Support"):
                    data = data[data["RA"] == 1]
                    if arg_element1 == True and arg_element2 == False and arg_element3 == False and arg_element4 == False:
                        df = data[(data["RA_premise"] == 1)]
                    elif arg_element1 == False and arg_element2 == True and arg_element3 == False and arg_element4 == False:
                        df = data[(data["RA_conclusion"] == 1)]
                    elif arg_element1 == False and arg_element2 == False and arg_element3 == True and arg_element4 == False:
                        df = data[(data["RA final input"] == 1)]
                    elif arg_element1 == False and arg_element2 == False and arg_element3 == False and arg_element4 == True:
                        df = data[(data["RA final output"] == 1)]
                    elif arg_element1 == False and arg_element2 == False and arg_element3 == True and arg_element4 == True:
                        df = data[(data["RA final input"] == 1) | (data["RA final output"] == 1)]
                    elif arg_element1 == True and arg_element2 == False and arg_element3 == True and arg_element4 == True:
                        df = data[(data["RA final output"] == 1) | (data["RA_premise"] == 1)]
                    elif arg_element1 == False and arg_element2 == True and arg_element3 == True and arg_element4 == True:
                        df = data[(data["RA final output"] == 1) | (data["RA_conclusion"] == 1)]
                    elif arg_element1 == True and arg_element2 == False and arg_element3 == True and arg_element4 == False:
                        df = data[(data["RA_premise"] == 1)]
                    elif arg_element1 == False and arg_element2 == True and arg_element3 == False and arg_element4 == True:
                        df = data[(data["RA_conclusion"] == 1)]
                    elif arg_element1 == False and arg_element2 == True and arg_element3 == True and arg_element4 == False:
                        df = data[(data["RA final input"] == 1) | (data["RA_conclusion"] == 1)]
                    elif arg_element1 == True and arg_element2 == False and arg_element3 == False and arg_element4 == True:
                        df = data[(data["RA final output"] == 1) | (data["RA_premise"] == 1)]
                    elif arg_element1 == False and arg_element2 == True and arg_element3 == True and arg_element4 == False:
                        df = data[(data["RA final input"] == 1) | (data["RA_conclusion"] == 1)]
                    elif arg_element1 == True and arg_element2 == False and arg_element3 == False and arg_element4 == True:
                        df = data[(data["RA final output"] == 1) | (data["RA_premise"] == 1)]
                    else:
                        df = data
            elif (len(arg_option)==1) and (arg_option[0] == "Attack"):
                    data = data[data["CA"] == 1]
                    if arg_element1 == True and arg_element2 == False and arg_element3 == False and arg_element4 == False:
                        df = data[(data["CA_premise"] == 1)]
                    elif arg_element1 == False and arg_element2 == True and arg_element3 == False and arg_element4 == False:
                        df = data[(data["CA_conclusion"] == 1)]
                    elif arg_element1 == False and arg_element2 == False and arg_element3 == True and arg_element4 == False:
                        df = data[(data["CA final input"] == 1)]
                    elif arg_element1 == False and arg_element2 == False and arg_element3 == False and arg_element4 == True:
                        df = data[(data["CA final output"] == 1)]
                    elif arg_element1 == False and arg_element2 == False and arg_element3 == True and arg_element4 == True:
                        df = data[(data["CA final input"] == 1) | (data["CA final output"] == 1)]
                    elif arg_element1 == True and arg_element2 == False and arg_element3 == True and arg_element4 == True:
                        df = data[(data["CA final output"] == 1) | (data["CA_premise"] == 1)]
                    elif arg_element1 == False and arg_element2 == True and arg_element3 == True and arg_element4 == True:
                        df = data[(data["CA final output"] == 1) | (data["CA_conclusion"] == 1)]
                    elif arg_element1 == True and arg_element2 == False and arg_element3 == True and arg_element4 == False:
                        df = data[(data["CA_premise"] == 1)]
                    elif arg_element1 == False and arg_element2 == True and arg_element3 == False and arg_element4 == True:
                        df = data[(data["CA_conclusion"] == 1)]
                    elif arg_element1 == False and arg_element2 == True and arg_element3 == True and arg_element4 == False:
                        df = data[(data["CA final input"] == 1) | (data["CA_conclusion"] == 1)]
                    elif arg_element1 == True and arg_element2 == False and arg_element3 == False and arg_element4 == True:
                        df = data[(data["CA final output"] == 1) | (data["CA_premise"] == 1)]
                    else:
                        df = data
            elif (len(arg_option)==2):
                    data = data[(data["RA"] == 1) | (data["CA"] == 1)]
                    if arg_element1 == True and arg_element2 == False and arg_element3 == False and arg_element4 == False:
                        df = data[(data["CA_premise"] == 1)|(data["RA_premise"] == 1)]
                    elif arg_element1 == False and arg_element2 == True and arg_element3 == False and arg_element4 == False:
                        df = data[(data["CA_conclusion"] == 1)|(data["RA_conclusion"] == 1)]
                    elif arg_element1 == False and arg_element2 == False and arg_element3 == True and arg_element4 == False:
                        df = data[(data["CA final input"] == 1)|(data["RA final input"] == 1)]
                    elif arg_element1 == False and arg_element2 == False and arg_element3 == False and arg_element4 == True:
                        df = data[(data["CA final output"] == 1)|(data["RA final output"] == 1)]
                    elif arg_element1 == False and arg_element2 == False and arg_element3 == True and arg_element4 == True:
                        df = data[(data["CA final input"] == 1) | (data["CA final output"] == 1) | (data["RA final input"] == 1) | (data["RA final output"] == 1)]
                    elif arg_element1 == True and arg_element2 == False and arg_element3 == True and arg_element4 == True:
                        df = data[(data["CA final output"] == 1) | (data["CA_premise"] == 1) | (data["RA final output"] == 1) | (data["RA_premise"] == 1)]
                    elif arg_element1 == False and arg_element2 == True and arg_element3 == True and arg_element4 == True:
                        df = data[(data["CA final output"] == 1) | (data["CA_conclusion"] == 1) | (data["RA final output"] == 1) | (data["RA_conclusion"] == 1)]
                    elif arg_element1 == True and arg_element2 == False and arg_element3 == True and arg_element4 == False:
                        df = data[(data["CA_premise"] == 1) | (data["RA_premise"] == 1)]
                    elif arg_element1 == False and arg_element2 == True and arg_element3 == False and arg_element4 == True:
                        df = data[(data["CA_conclusion"] == 1) | (data["RA_conclusion"] == 1)]
                    elif arg_element1 == False and arg_element2 == True and arg_element3 == True and arg_element4 == False:
                        df = data[(data["RA final input"] == 1) | (data["RA_conclusion"] == 1) | (data["CA final input"] == 1) | (data["CA_conclusion"] == 1)]
                    elif arg_element1 == True and arg_element2 == False and arg_element3 == False and arg_element4 == True:
                        df = data[(data["RA final output"] == 1) | (data["RA_premise"] == 1) | (data["CA final output"] == 1) | (data["CA_premise"] == 1)]
                    else:
                        df = data


            category_type = st.selectbox(
                'choose an moral foundation category',
                ('Care',
                 'Fairness',
                 "Loyalty",
                 "Authority",
                 "Sanctity",))
            valence_type = st.selectbox(
                'choose the valence of moral foundation category',
                ('Positive',
                 'Negative',))
            add_spacelines(1)
            threshold_cloud = st.slider('Select a precision value (threshold) for a WordCloud',
                                        0, 100, 100, disabled=True)
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

            st.info(f'Selected precision: **{threshold_cloud}**')
            f1 = Word_Cloud(df, word_type)
            MFW_dataframe = Find_MFW(data=df, word_type=word_type)
            if len(MFW_dataframe) != 0:
                tab1, tab2 = st.tabs(["Graph", "Table"])
                with tab1:
                    format1 = st.radio("Choose the y-axis unit", ("number", "percentage"))
                    f2 = Top_10_MFW(MFW_dataframe, with_slash=False)
                    fig = Top_10_Moral_Foundation_Words_Visualisation(f2, word_type, format1)
                    st.plotly_chart(fig)
                with tab2:
                    st.write("**Top 10 {} Words Frequency**".format(word_type_))
                    f2 = Top_10_MFW(MFW_dataframe, with_slash=True)
                    st.table(f2)
                    st.write("**Total {} Words Frequency**".format(word_type_))
                    f3 = Total_Word_Frequency(df, word_type)
                    st.table(f3)

            else:
                st.info("No {} in this corpus".format(word_type.lower()))
            add_spacelines(2)
            st.write("#### 2. Qualitative Analysis")
            MFW_dataframe = Find_MFW(data=df, word_type=word_type)
            if len(MFW_dataframe) != 0:
                selectbox_name = MFW_dataframe.words.values.tolist()
                word_selection = st.radio(
                    "Choose the number of {} words you want to select".format(word_type_.lower()),
                    ("single", "all",))
                if word_selection == "single":
                    possible_words = st.selectbox("Choose {} words you want to analyse".format(word_type_.lower()),
                                                  selectbox_name)
                else:
                    possible_words = np.nan
                f5 = Texts_With_MFW(df, word_type, possible_words, word_selection)
                st.table(f5)
            else:
                st.info("No {} words in this corpus".format(word_type_.lower()))

        else:
            st.write("")


def Comparative_Moral_Foundation_Word_Cloud(mode):
    if mode == "ARG":
        global df
        data = df.copy(deep=True)
        st.subheader(f"Moral Foundation Word Cloud Visualisation")
        st.write('**Elements:**')
        column1, column2, column3, column4 = st.columns([2, 2, 2, 2])
        with column1:
            arg_element1 = st.checkbox("Input", value=False)
        with column2:
            arg_element2 = st.checkbox("Output", value=True)
        with column3:
            arg_element3 = st.checkbox("Initial Input", value=False)
        with column4:
            arg_element4 = st.checkbox("Final Output", value=True)
        arg_option = st.multiselect(
            'Input/Output Relations',
            ['Support', 'Attack'],
            ['Support', 'Attack'])
        if len(options) >= 2:
            if (len(arg_option) == 1) and (arg_option[0] == "Support"):
                data = data[data["RA"] == 1]
                if arg_element1 == True and arg_element2 == False and arg_element3 == False and arg_element4 == False:
                    df = data[(data["RA_premise"] == 1)]
                elif arg_element1 == False and arg_element2 == True and arg_element3 == False and arg_element4 == False:
                    df = data[(data["RA_conclusion"] == 1)]
                elif arg_element1 == False and arg_element2 == False and arg_element3 == True and arg_element4 == False:
                    df = data[(data["RA final input"] == 1)]
                elif arg_element1 == False and arg_element2 == False and arg_element3 == False and arg_element4 == True:
                    df = data[(data["RA final output"] == 1)]
                elif arg_element1 == False and arg_element2 == False and arg_element3 == True and arg_element4 == True:
                    df = data[(data["RA final input"] == 1) | (data["RA final output"] == 1)]
                elif arg_element1 == True and arg_element2 == False and arg_element3 == True and arg_element4 == True:
                    df = data[(data["RA final output"] == 1) | (data["RA_premise"] == 1)]
                elif arg_element1 == False and arg_element2 == True and arg_element3 == True and arg_element4 == True:
                    df = data[(data["RA final output"] == 1) | (data["RA_conclusion"] == 1)]
                elif arg_element1 == True and arg_element2 == False and arg_element3 == True and arg_element4 == False:
                    df = data[(data["RA_premise"] == 1)]
                elif arg_element1 == False and arg_element2 == True and arg_element3 == False and arg_element4 == True:
                    df = data[(data["RA_conclusion"] == 1)]
                elif arg_element1 == False and arg_element2 == True and arg_element3 == True and arg_element4 == False:
                    df = data[(data["RA final input"] == 1) | (data["RA_conclusion"] == 1)]
                elif arg_element1 == True and arg_element2 == False and arg_element3 == False and arg_element4 == True:
                    df = data[(data["RA final output"] == 1) | (data["RA_premise"] == 1)]
                elif arg_element1 == False and arg_element2 == True and arg_element3 == True and arg_element4 == False:
                    df = data[(data["RA final input"] == 1) | (data["RA_conclusion"] == 1)]
                elif arg_element1 == True and arg_element2 == False and arg_element3 == False and arg_element4 == True:
                    df = data[(data["RA final output"] == 1) | (data["RA_premise"] == 1)]
                else:
                    df = data
            elif (len(arg_option) == 1) and (arg_option[0] == "Attack"):
                data = data[data["CA"] == 1]
                if arg_element1 == True and arg_element2 == False and arg_element3 == False and arg_element4 == False:
                    df = data[(data["CA_premise"] == 1)]
                elif arg_element1 == False and arg_element2 == True and arg_element3 == False and arg_element4 == False:
                    df = data[(data["CA_conclusion"] == 1)]
                elif arg_element1 == False and arg_element2 == False and arg_element3 == True and arg_element4 == False:
                    df = data[(data["CA final input"] == 1)]
                elif arg_element1 == False and arg_element2 == False and arg_element3 == False and arg_element4 == True:
                    df = data[(data["CA final output"] == 1)]
                elif arg_element1 == False and arg_element2 == False and arg_element3 == True and arg_element4 == True:
                    df = data[(data["CA final input"] == 1) | (data["CA final output"] == 1)]
                elif arg_element1 == True and arg_element2 == False and arg_element3 == True and arg_element4 == True:
                    df = data[(data["CA final output"] == 1) | (data["CA_premise"] == 1)]
                elif arg_element1 == False and arg_element2 == True and arg_element3 == True and arg_element4 == True:
                    df = data[(data["CA final output"] == 1) | (data["CA_conclusion"] == 1)]
                elif arg_element1 == True and arg_element2 == False and arg_element3 == True and arg_element4 == False:
                    df = data[(data["CA_premise"] == 1)]
                elif arg_element1 == False and arg_element2 == True and arg_element3 == False and arg_element4 == True:
                    df = data[(data["CA_conclusion"] == 1)]
                elif arg_element1 == False and arg_element2 == True and arg_element3 == True and arg_element4 == False:
                    df = data[(data["CA final input"] == 1) | (data["CA_conclusion"] == 1)]
                elif arg_element1 == True and arg_element2 == False and arg_element3 == False and arg_element4 == True:
                    df = data[(data["CA final output"] == 1) | (data["CA_premise"] == 1)]
                else:
                    df = data
            elif (len(arg_option) == 2):
                data = data[(data["RA"] == 1) | (data["CA"] == 1)]
                if arg_element1 == True and arg_element2 == False and arg_element3 == False and arg_element4 == False:
                    df = data[(data["CA_premise"] == 1) | (data["RA_premise"] == 1)]
                elif arg_element1 == False and arg_element2 == True and arg_element3 == False and arg_element4 == False:
                    df = data[(data["CA_conclusion"] == 1) | (data["RA_conclusion"] == 1)]
                elif arg_element1 == False and arg_element2 == False and arg_element3 == True and arg_element4 == False:
                    df = data[(data["CA final input"] == 1) | (data["RA final input"] == 1)]
                elif arg_element1 == False and arg_element2 == False and arg_element3 == False and arg_element4 == True:
                    df = data[(data["CA final output"] == 1) | (data["RA final output"] == 1)]
                elif arg_element1 == False and arg_element2 == False and arg_element3 == True and arg_element4 == True:
                    df = data[
                        (data["CA final input"] == 1) | (data["CA final output"] == 1) | (data["RA final input"] == 1) | (
                                    data["RA final output"] == 1)]
                elif arg_element1 == True and arg_element2 == False and arg_element3 == True and arg_element4 == True:
                    df = data[
                        (data["CA final output"] == 1) | (data["CA_premise"] == 1) | (data["RA final output"] == 1) | (
                                    data["RA_premise"] == 1)]
                elif arg_element1 == False and arg_element2 == True and arg_element3 == True and arg_element4 == True:
                    df = data[
                        (data["CA final output"] == 1) | (data["CA_conclusion"] == 1) | (data["RA final output"] == 1) | (
                                    data["RA_conclusion"] == 1)]
                elif arg_element1 == True and arg_element2 == False and arg_element3 == True and arg_element4 == False:
                    df = data[(data["CA_premise"] == 1) | (data["RA_premise"] == 1)]
                elif arg_element1 == False and arg_element2 == True and arg_element3 == False and arg_element4 == True:
                    df = data[(data["CA_conclusion"] == 1) | (data["RA_conclusion"] == 1)]
                elif arg_element1 == False and arg_element2 == True and arg_element3 == True and arg_element4 == False:
                    df = data[
                        (data["RA final input"] == 1) | (data["RA_conclusion"] == 1) | (data["CA final input"] == 1) | (
                                    data["CA_conclusion"] == 1)]
                elif arg_element1 == True and arg_element2 == False and arg_element3 == False and arg_element4 == True:
                    df = data[
                        (data["RA final output"] == 1) | (data["RA_premise"] == 1) | (data["CA final output"] == 1) | (
                                    data["CA_premise"] == 1)]
                else:
                    df = data
        valence_type = st.selectbox(
            'choose the valence of moral foundation category',
            ('Positive',
             'Negative',))
        category_type = st.selectbox(
                'choose an moral foundation category',
                ('Care',
                 'Fairness',
                 "Loyalty",
                 "Authority",
                 "Sanctity",))

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

        if len(options) >= 2:
            st.write("##### Shared {} Word Cloud".format(word_type_))
            f3 = Compare_Shared_Words_Frequency(data=df, word_type=word_type)
            if len(f3) != 0:
                wordcloud = Word_Cloud(f3.reset_index(), word_type)
            else:
                st.info(
                    "No shared {} words".format(word_type_.lower()))

            st.write("##### Shared {} Words Frequency".format(word_type_))
            if len(f3) != 0:
                st.table(f3.astype(str))
                st.write(" Shared {} Word Frequency (Total) is {}".format(word_type_, f3.sum(axis=1).sum()))
            else:
                st.info(
                    "No shared {} words".format(word_type_.lower()))

            st.write("##### Unique {} Word Frequency".format(word_type_))
            f4, index = Compare_Unique_Words_Frequency(data=df, word_type=word_type)
            for i in index:
                if f4[i].sum() != 0:
                    st.table(f4[i].astype(str))
                    st.write(" Unique {} Word Frequency (Total) in *{}* is {}".format(word_type_, i, f4[i].sum()))
        else:
            st.write("")
    else:
        st.subheader(f"Moral Foundation Word Cloud Visualisation")
        valence_type = st.selectbox(
            'choose the valence of moral foundation category',
            ('Positive',
             'Negative',))
        category_type = st.selectbox(
            'choose an moral foundation category',
            ('Care',
             'Fairness',
             "Loyalty",
             "Authority",
             "Sanctity",))

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

        if len(options) >= 2:
            st.write("##### Shared {} Word Cloud".format(word_type_))
            f3 = Compare_Shared_Words_Frequency(data=df, word_type=word_type)
            if len(f3) != 0:
                wordcloud = Word_Cloud(f3.reset_index(), word_type)
            else:
                st.info(
                    "No shared {} words".format(word_type_.lower()))

            st.write("##### Shared {} Words Frequency".format(word_type_))
            if len(f3) != 0:
                st.table(f3.astype(str))
                st.write(" Shared {} Word Frequency (Total) is {}".format(word_type_, f3.sum(axis=1).sum()))
            else:
                st.info(
                    "No shared {} words".format(word_type_.lower()))

            st.write("##### Unique {} Word Frequency".format(word_type_))
            f4, index = Compare_Unique_Words_Frequency(data=df, word_type=word_type)
            for i in index:
                if f4[i].sum() != 0:
                    st.table(f4[i].astype(str))
                    st.write(" Unique {} Word Frequency (Total) in *{}* is {}".format(word_type_, i, f4[i].sum()))
        else:
            st.write("")


def Moral_Value_Distribution(mode):
    global df
    data = df.copy(deep=True)
    if mode == "ADU":
        st.subheader("Moral Value Distribution")
        add_spacelines(2)
        column1, column2 = st.columns([2, 2])
        with column1:
            format1 = st.radio("Choose the y-aix unit", ("number", "percentage",))
        if len(options) != 0:
            fig = Moral_Foundation_Word_In_Tweet(data, format1)
            st.pyplot(fig)
        else:
            st.write("")
    elif mode == "ARG":
        st.subheader("Moral Value Distribution")
        add_spacelines(2)
        st.write('**Elements:**')
        column1, column2, column3, column4 = st.columns([2, 2, 2, 2])
        with column1:
            arg_element1 = st.checkbox("Input", value=False)
        with column2:
            arg_element2 = st.checkbox("Output", value=True)
        with column3:
            arg_element3 = st.checkbox("Initial Input", value=False)
        with column4:
            arg_element4 = st.checkbox("Final Output", value=True)
        arg_option = st.multiselect(
            'Input/Output Relations',
            ['Support', 'Attack'],
            ['Support', 'Attack'])
        column1, column2 = st.columns([2, 2])
        with column1:
            format1 = st.radio("Choose the y-aix unit", ("number", "percentage",))
        with column2:
            format2 = st.radio("Choose the calculation method", ("union",))

        if (len(arg_option) == 1) and (arg_option[0] == "Support"):
            data = data[data["RA"] == 1]
            if arg_element1 == True and arg_element2 == False and arg_element3 == False and arg_element4 == False:
                df = data[(data["RA_premise"] == 1)]
            elif arg_element1 == False and arg_element2 == True and arg_element3 == False and arg_element4 == False:
                df = data[(data["RA_conclusion"] == 1)]
            elif arg_element1 == False and arg_element2 == False and arg_element3 == True and arg_element4 == False:
                df = data[(data["RA final input"] == 1)]
            elif arg_element1 == False and arg_element2 == False and arg_element3 == False and arg_element4 == True:
                df = data[(data["RA final output"] == 1)]
            elif arg_element1 == False and arg_element2 == False and arg_element3 == True and arg_element4 == True:
                df = data[(data["RA final input"] == 1) | (data["RA final output"] == 1)]
            elif arg_element1 == True and arg_element2 == False and arg_element3 == True and arg_element4 == True:
                df = data[(data["RA final output"] == 1) | (data["RA_premise"] == 1)]
            elif arg_element1 == False and arg_element2 == True and arg_element3 == True and arg_element4 == True:
                df = data[(data["RA final output"] == 1) | (data["RA_conclusion"] == 1)]
            elif arg_element1 == True and arg_element2 == False and arg_element3 == True and arg_element4 == False:
                df = data[(data["RA_premise"] == 1)]
            elif arg_element1 == False and arg_element2 == True and arg_element3 == False and arg_element4 == True:
                df = data[(data["RA_conclusion"] == 1)]
            elif arg_element1 == False and arg_element2 == True and arg_element3 == True and arg_element4 == False:
                df = data[(data["RA final input"] == 1) | (data["RA_conclusion"] == 1)]
            elif arg_element1 == True and arg_element2 == False and arg_element3 == False and arg_element4 == True:
                df = data[(data["RA final output"] == 1) | (data["RA_premise"] == 1)]
            elif arg_element1 == False and arg_element2 == True and arg_element3 == True and arg_element4 == False:
                df = data[(data["RA final input"] == 1) | (data["RA_conclusion"] == 1)]
            elif arg_element1 == True and arg_element2 == False and arg_element3 == False and arg_element4 == True:
                df = data[(data["RA final output"] == 1) | (data["RA_premise"] == 1)]
            else:
                df = data
        elif (len(arg_option) == 1) and (arg_option[0] == "Attack"):
            data = data[data["CA"] == 1]
            if arg_element1 == True and arg_element2 == False and arg_element3 == False and arg_element4 == False:
                df = data[(data["CA_premise"] == 1)]
            elif arg_element1 == False and arg_element2 == True and arg_element3 == False and arg_element4 == False:
                df = data[(data["CA_conclusion"] == 1)]
            elif arg_element1 == False and arg_element2 == False and arg_element3 == True and arg_element4 == False:
                df = data[(data["CA final input"] == 1)]
            elif arg_element1 == False and arg_element2 == False and arg_element3 == False and arg_element4 == True:
                df = data[(data["CA final output"] == 1)]
            elif arg_element1 == False and arg_element2 == False and arg_element3 == True and arg_element4 == True:
                df = data[(data["CA final input"] == 1) | (data["CA final output"] == 1)]
            elif arg_element1 == True and arg_element2 == False and arg_element3 == True and arg_element4 == True:
                df = data[(data["CA final output"] == 1) | (data["CA_premise"] == 1)]
            elif arg_element1 == False and arg_element2 == True and arg_element3 == True and arg_element4 == True:
                df = data[(data["CA final output"] == 1) | (data["CA_conclusion"] == 1)]
            elif arg_element1 == True and arg_element2 == False and arg_element3 == True and arg_element4 == False:
                df = data[(data["CA_premise"] == 1)]
            elif arg_element1 == False and arg_element2 == True and arg_element3 == False and arg_element4 == True:
                df = data[(data["CA_conclusion"] == 1)]
            elif arg_element1 == False and arg_element2 == True and arg_element3 == True and arg_element4 == False:
                df = data[(data["CA final input"] == 1) | (data["CA_conclusion"] == 1)]
            elif arg_element1 == True and arg_element2 == False and arg_element3 == False and arg_element4 == True:
                df = data[(data["CA final output"] == 1) | (data["CA_premise"] == 1)]
            else:
                df = data
        elif (len(arg_option) == 2):
            data = data[(data["RA"] == 1) | (data["CA"] == 1)]
            if arg_element1 == True and arg_element2 == False and arg_element3 == False and arg_element4 == False:
                df = data[(data["CA_premise"] == 1) | (data["RA_premise"] == 1)]
            elif arg_element1 == False and arg_element2 == True and arg_element3 == False and arg_element4 == False:
                df = data[(data["CA_conclusion"] == 1) | (data["RA_conclusion"] == 1)]
            elif arg_element1 == False and arg_element2 == False and arg_element3 == True and arg_element4 == False:
                df = data[(data["CA final input"] == 1) | (data["RA final input"] == 1)]
            elif arg_element1 == False and arg_element2 == False and arg_element3 == False and arg_element4 == True:
                df = data[(data["CA final output"] == 1) | (data["RA final output"] == 1)]
            elif arg_element1 == False and arg_element2 == False and arg_element3 == True and arg_element4 == True:
                df = data[
                    (data["CA final input"] == 1) | (data["CA final output"] == 1) | (data["RA final input"] == 1) | (
                                data["RA final output"] == 1)]
            elif arg_element1 == True and arg_element2 == False and arg_element3 == True and arg_element4 == True:
                df = data[
                    (data["CA final output"] == 1) | (data["CA_premise"] == 1) | (data["RA final output"] == 1) | (
                                data["RA_premise"] == 1)]
            elif arg_element1 == False and arg_element2 == True and arg_element3 == True and arg_element4 == True:
                df = data[
                    (data["CA final output"] == 1) | (data["CA_conclusion"] == 1) | (data["RA final output"] == 1) | (
                                data["RA_conclusion"] == 1)]
            elif arg_element1 == True and arg_element2 == False and arg_element3 == True and arg_element4 == False:
                df = data[(data["CA_premise"] == 1) | (data["RA_premise"] == 1)]
            elif arg_element1 == False and arg_element2 == True and arg_element3 == False and arg_element4 == True:
                df = data[(data["CA_conclusion"] == 1) | (data["RA_conclusion"] == 1)]
            elif arg_element1 == False and arg_element2 == True and arg_element3 == True and arg_element4 == False:
                df = data[
                    (data["RA final input"] == 1) | (data["RA_conclusion"] == 1) | (data["CA final input"] == 1) | (
                                data["CA_conclusion"] == 1)]
            elif arg_element1 == True and arg_element2 == False and arg_element3 == False and arg_element4 == True:
                df = data[
                    (data["RA final output"] == 1) | (data["RA_premise"] == 1) | (data["CA final output"] == 1) | (
                                data["CA_premise"] == 1)]
            else:
                df = data
        fig = Moral_Foundation_Word_In_Tweet(df, format1)
        st.pyplot(fig)
    else:
        st.write("")



def Comparative_Moral_Value_Distribution(mode):
    if mode == "ARG":
        global df
        data = df.copy(deep=True)
        st.subheader("Moral Value Distribution")
        add_spacelines(2)
        st.write('**Elements:**')
        column1, column2, column3, column4 = st.columns([2, 2, 2, 2])
        with column1:
            arg_element1 = st.checkbox("Input", value=True)
        with column2:
            arg_element2 = st.checkbox("Output", value=True)
        with column3:
            arg_element3 = st.checkbox("Initial Input", value=False)
        with column4:
            arg_element4 = st.checkbox("Final Output", value=False)
        arg_option = st.multiselect(
            'Input/Output Relations',
            ['Support', 'Attack'],
            ['Support', 'Attack'])
        if len(options) >= 2:
            if (len(arg_option) == 1) and (arg_option[0] == "Support"):
                data = data[(data["RA"] == 1)]
                if arg_element1 == True and arg_element2 == False and arg_element3 == False and arg_element4 == False:
                    df = data[(data["RA_premise"] == 1)]
                elif arg_element1 == False and arg_element2 == True and arg_element3 == False and arg_element4 == False:
                    df = data[(data["RA_conclusion"] == 1)]
                elif arg_element1 == False and arg_element2 == False and arg_element3 == True and arg_element4 == False:
                    df = data[(data["RA final input"] == 1)]
                elif arg_element1 == False and arg_element2 == False and arg_element3 == False and arg_element4 == True:
                    df = data[(data["RA final output"] == 1)]
                elif arg_element1 == False and arg_element2 == False and arg_element3 == True and arg_element4 == True:
                    df = data[(data["RA final input"] == 1) | (data["RA final output"] == 1)]
                elif arg_element1 == True and arg_element2 == False and arg_element3 == True and arg_element4 == True:
                    df = data[(data["RA final output"] == 1) | (data["RA_premise"] == 1)]
                elif arg_element1 == False and arg_element2 == True and arg_element3 == True and arg_element4 == True:
                    df = data[(data["RA final output"] == 1) | (data["RA_conclusion"] == 1)]
                elif arg_element1 == True and arg_element2 == False and arg_element3 == True and arg_element4 == False:
                    df = data[(data["RA_premise"] == 1)]
                elif arg_element1 == False and arg_element2 == True and arg_element3 == False and arg_element4 == True:
                    df = data[(data["RA_conclusion"] == 1)]
                elif arg_element1 == False and arg_element2 == True and arg_element3 == True and arg_element4 == False:
                    df = data[(data["RA final input"] == 1) | (data["RA_conclusion"] == 1)]
                elif arg_element1 == True and arg_element2 == False and arg_element3 == False and arg_element4 == True:
                    df = data[(data["RA final output"] == 1) | (data["RA_premise"] == 1)]
                elif arg_element1 == False and arg_element2 == True and arg_element3 == True and arg_element4 == False:
                    df = data[(data["RA final input"] == 1) | (data["RA_conclusion"] == 1)]
                elif arg_element1 == True and arg_element2 == False and arg_element3 == False and arg_element4 == True:
                    df = data[(data["RA final output"] == 1) | (data["RA_premise"] == 1)]
                else:
                    df = data
            elif (len(arg_option) == 1) and (arg_option[0] == "Attack"):
                data = data[data["CA"] == 1]
                if arg_element1 == True and arg_element2 == False and arg_element3 == False and arg_element4 == False:
                    df = data[(data["CA_premise"] == 1)]
                elif arg_element1 == False and arg_element2 == True and arg_element3 == False and arg_element4 == False:
                    df = data[(data["CA_conclusion"] == 1)]
                elif arg_element1 == False and arg_element2 == False and arg_element3 == True and arg_element4 == False:
                    df = data[(data["CA final input"] == 1)]
                elif arg_element1 == False and arg_element2 == False and arg_element3 == False and arg_element4 == True:
                    df = data[(data["CA final output"] == 1)]
                elif arg_element1 == False and arg_element2 == False and arg_element3 == True and arg_element4 == True:
                    df = data[(data["CA final input"] == 1) | (data["CA final output"] == 1)]
                elif arg_element1 == True and arg_element2 == False and arg_element3 == True and arg_element4 == True:
                    df = data[(data["CA final output"] == 1) | (data["CA_premise"] == 1)]
                elif arg_element1 == False and arg_element2 == True and arg_element3 == True and arg_element4 == True:
                    df = data[(data["CA final output"] == 1) | (data["CA_conclusion"] == 1)]
                elif arg_element1 == True and arg_element2 == False and arg_element3 == True and arg_element4 == False:
                    df = data[(data["CA_premise"] == 1)]
                elif arg_element1 == False and arg_element2 == True and arg_element3 == False and arg_element4 == True:
                    df = data[(data["CA_conclusion"] == 1)]
                elif arg_element1 == False and arg_element2 == True and arg_element3 == True and arg_element4 == False:
                    df = data[(data["CA final input"] == 1) | (data["CA_conclusion"] == 1)]
                elif arg_element1 == True and arg_element2 == False and arg_element3 == False and arg_element4 == True:
                    df = data[(data["CA final output"] == 1) | (data["CA_premise"] == 1)]
                else:
                    df = data
            elif (len(arg_option) == 2):
                data = data[(data["RA"] == 1) | (data["CA"] == 1)]
                if arg_element1 == True and arg_element2 == False and arg_element3 == False and arg_element4 == False:
                    df = data[(data["CA_premise"] == 1) | (data["RA_premise"] == 1)]
                elif arg_element1 == False and arg_element2 == True and arg_element3 == False and arg_element4 == False:
                    df = data[(data["CA_conclusion"] == 1) | (data["RA_conclusion"] == 1)]
                elif arg_element1 == False and arg_element2 == False and arg_element3 == True and arg_element4 == False:
                    df = data[(data["CA final input"] == 1) | (data["RA final input"] == 1)]
                elif arg_element1 == False and arg_element2 == False and arg_element3 == False and arg_element4 == True:
                    df = data[(data["CA final output"] == 1) | (data["RA final output"] == 1)]
                elif arg_element1 == False and arg_element2 == False and arg_element3 == True and arg_element4 == True:
                    df = data[
                        (data["CA final input"] == 1) | (data["CA final output"] == 1) | (data["RA final input"] == 1) | (
                                data["RA final output"] == 1)]
                elif arg_element1 == True and arg_element2 == False and arg_element3 == True and arg_element4 == True:
                    df = data[
                        (data["CA final output"] == 1) | (data["CA_premise"] == 1) | (data["RA final output"] == 1) | (
                                data["RA_premise"] == 1)]
                elif arg_element1 == False and arg_element2 == True and arg_element3 == True and arg_element4 == True:
                    df = data[
                        (data["CA final output"] == 1) | (data["CA_conclusion"] == 1) | (data["RA final output"] == 1) | (
                                data["RA_conclusion"] == 1)]
                elif arg_element1 == True and arg_element2 == False and arg_element3 == True and arg_element4 == False:
                    df = data[(data["CA_premise"] == 1) | (data["RA_premise"] == 1)]
                elif arg_element1 == False and arg_element2 == True and arg_element3 == False and arg_element4 == True:
                    df = data[(data["CA_conclusion"] == 1) | (data["RA_conclusion"] == 1)]
                elif arg_element1 == False and arg_element2 == True and arg_element3 == True and arg_element4 == False:
                    df = data[
                        (data["RA fina   l input"] == 1) | (data["RA_conclusion"] == 1) | (data["CA final input"] == 1) | (
                                data["CA_conclusion"] == 1)]
                elif arg_element1 == True and arg_element2 == False and arg_element3 == False and arg_element4 == True:
                    df = data[
                        (data["RA final output"] == 1) | (data["RA_premise"] == 1) | (data["CA final output"] == 1) | (
                                data["CA_premise"] == 1)]
                else:
                    df = data
        valence = st.selectbox("Choose moral foundation valence", ("positive", "negative", "all"))
        column1,column2 = st.columns([2,2])
        with column1:
            format1 = st.radio("Choose the y-aix unit", ("number", "percentage",))
        with column2:
            format2 = st.radio("Choose the calculation method", ("union",))

        if len(options) >= 2:
            fig = Compare_Moral_Foundation_Word_In_Tweet_Group(df, format1, valence)
            st.plotly_chart(fig)
            fig = Compare_Moral_Foundation_Word_In_Tweet_Layer(df, format1, valence)
            st.plotly_chart(fig)
        else:
            st.write("")
    else:
        st.subheader("Moral Value Distribution")
        add_spacelines(2)
        valence = st.selectbox("Choose moral foundation valence", ("positive", "negative", "all"))
        column1, column2 = st.columns([2, 2])
        with column1:
            format1 = st.radio("Choose the y-aix unit", ("number", "percentage",))

        if len(options) != 0:
            fig = Compare_Moral_Foundation_Word_In_Tweet_Group(df, format1, valence)
            st.plotly_chart(fig)
            fig = Compare_Moral_Foundation_Word_In_Tweet_Layer(df, format1, valence)
            st.plotly_chart(fig)
        else:
            st.write("")


def Compare_Moral_Concern_User_Distribution():
    st.subheader("Moral Value Score:  Interlocutor Distribution")
    add_spacelines(2)
    st.write("##### 1. Interlocutor Distribution")
    moral_foundation = st.selectbox("Choose the moral foundation",
                                    ("care", "fairness", "authority", "loyalty", "sanctity"))
    column1, column2 = st.columns([2, 2])
    with column1:
        format1 = st.radio("Choose the y-aix unit", ("number", "percentage",))
    if len(options) != 0:
        fig = Compare_User_Distribution_Stack(df, moral_foundation, format1)
        st.plotly_chart(fig)
        fig = Compare_User_Distribution_Group(df, moral_foundation, format1)
        st.plotly_chart(fig)
        fig = Compare_User_Distribution_Layer(df, moral_foundation, format1)
        st.plotly_chart(fig)
        st.write("##### 2. Average User Concern Score")
        fig = Compare_Average_User_Concern_Score(df)
        st.plotly_chart(fig)
    else:
        st.write("")


def Moral_Concern_User_Interaction_Analysis():
    st.subheader("Moral Valence Degree: Interlocutor Argumentative Interaction")
    add_spacelines(2)
    moral_foundation = st.selectbox("Choose the moral foundation",
                                    ("care", "fairness", "authority", "loyalty", "sanctity"))
    aspect = st.radio("Choose the analysis aspect",
                      ("ADU frequency",))
    if len(options) != 0:
        fig = User_Interaction_Analysis(df, aspect, moral_foundation)
        st.plotly_chart(fig)
    else:
        st.write("")


def Moral_Concern_Network():
    st.subheader("Moral Valence Degree: Interlocutor Argumentative Network")
    add_spacelines(2)
    moral_foundation = st.selectbox("Choose the moral foundation",
                                    ("care", "fairness", "authority", "loyalty", "sanctity"))
    if len(options) != 0:
        fig = Moral_Concern_In_Social_Network(df, moral_foundation, options)
        st.pyplot(fig)
    else:
        st.write("")


def Moral_Concern_User_Distribution():
    st.subheader("Moral Value Score:  Interlocutor Distribution")
    add_spacelines(2)
    format1 = st.radio("Choose y-aix unit", ("number", "percentage",))
    if len(options) != 0:
        st.write("**User Type Distribution Across Five Moral Foundations**")
        fig = User_Distribution_Stack(df, format1)
        st.plotly_chart(fig)
        fig = User_Distribution_Group(df, format1)
        st.plotly_chart(fig)
        fig = User_Distribution_Layered(df, format1)
        st.plotly_chart(fig)
    else:
        st.write("")


def Moral_Concern_User_Score():
    st.subheader("Moral Value Score:  Interlocutor Score")
    if len(options) != 0:
        fig = User_Moral_Concern_Score_Heatmap(df)
        st.plotly_chart(fig)
    else:
        st.write("")

st.set_page_config(page_title="Moral Value Analytics", layout="wide")

def style_css(file):
    with open(file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

style_css(r"style/multi_style.css")

# sidebar
with st.sidebar:
    st.title("Contents")
    contents_radio1 = st.radio("Pages",
                               ("Main Page",
                                "Single Corpus Analysis",
                                "Comparative Corpora Analysis",))
    if contents_radio1 == "Main Page":
        contents_radio4 = "Main Page"

    elif contents_radio1 == "Single Corpus Analysis":
        st.subheader("Choose Corpora")
        corpus_option = st.checkbox("Moral Maze", key="MoralMaze_Single", value=True)
        st.write('**Topic:**')
        option1 = st.checkbox("British Empire", key="BritishEmpire_Single", value=False)
        option2 = st.checkbox("DDay", key="DDay_Single", value=False)
        option3 = st.checkbox("Hypocrisy", key="Hypocrisy_Single", value=False)
        option4 = st.checkbox("Money", key="Money_Single", value=False)
        option5 = st.checkbox("Welfare State", key="Welfare_Single", value=False)

        if option1 == True and option2 == False and option3 == False and option4 == False and option5 == False:
            options = ["British Empire"]
            corpus_name1 = "British Empire"
            df = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df["topic"] = "British Empire"
            df_pair = pd.read_csv(r"data/BritishEmpire Pair.csv", index_col=0)
            df_pair["topic"] = "British Empire"

        elif option1 == False and option2 == True and option3 == False and option4 == False and option5 == False:
            options = ["DDay"]
            corpus_name1 = "DDay"
            df = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df["topic"] = "DDay"
            df_pair = pd.read_csv(r"data/DDay Pair.csv", index_col=0)
            df_pair["topic"] = "DDay"

        elif option1 == False and option2 == False and option3 == True and option4 == False and option5 == False:
            options = ["Hypocrisy"]
            corpus_name1 = "Hypocrisy"
            df = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df["topic"] = "Hypocrisy"
            df_pair = pd.read_csv(r"data/Hypocrisy Pair.csv", index_col=0)
            df_pair["topic"] = "Hypocrisy"

        elif option1 == False and option2 == False and option3 == False and option4 == True and option5 == False:
            options = ["Money"]
            corpus_name1 = "Money"
            df = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df["topic"] = "Money"
            df_pair = pd.read_csv(r"data/Money Pair.csv", index_col=0)
            df_pair["topic"] = "Money"

        elif option1 == False and option2 == False and option3 == False and option4 == False and option5 == True:
            options = ["Welfare State"]
            corpus_name1 = "Welfare State"
            df = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df["topic"] = "Welfare State"
            df_pair = pd.read_csv(r"data/Welfare Pair.csv", index_col=0)
            df_pair["topic"] = "Welfare State"


        elif option1 == True and option2 == True and option3 == False and option4 == False and option5 == False:
            options = ["British Empire", "DDay"]
            corpus_name1 = "British Empire"
            corpus_name2 = "DDay"

            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df2 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df1["topic"] = "British Empire"
            df2["topic"] = "DDay"
            df = pd.concat([df1, df2])

            df1_pair = pd.read_csv(r"data/BritishEmpire Pair.csv", index_col=0)
            df2_pair = pd.read_csv(r"data/DDay Pair.csv", index_col=0)
            df1_pair["topic"] = "British Empire"
            df2_pair["topic"] = "DDay"
            df_pair = pd.concat([df1_pair, df2_pair])

        elif option1 == True and option2 == False and option3 == True and option4 == False and option5 == False:
            options = ["British Empire", "Hypocrisy"]
            corpus_name1 = "British Empire"
            corpus_name2 = "Hypocrisy"

            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df3 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df1["topic"] = "British Empire"
            df3["topic"] = "Hypocrisy"
            df = pd.concat([df1, df3])

            df1_pair = pd.read_csv(r"data/BritishEmpire Pair.csv", index_col=0)
            df3_pair = pd.read_csv(r"data/Hypocrisy Pair.csv", index_col=0)
            df1_pair["topic"] = "British Empire"
            df3_pair["topic"] = "Hypocrisy"
            df_pair = pd.concat([df1_pair, df3_pair])

        elif option1 == True and option2 == False and option3 == False and option4 == True and option5 == False:
            options = ["British Empire", "Money"]
            corpus_name1 = "British Empire"
            corpus_name2 = "Money"

            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df2 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df1["topic"] = "British Empire"
            df2["topic"] = "Money"
            df = pd.concat([df1, df2])

            df1_pair = pd.read_csv(r"data/BritishEmpire Pair.csv", index_col=0)
            df2_pair = pd.read_csv(r"data/Money Pair.csv", index_col=0)
            df1_pair["topic"] = "British Empire"
            df2_pair["topic"] = "Money"
            df_pair = pd.concat([df1_pair, df2_pair])

        elif option1 == True and option2 == False and option3 == False and option4 == False and option5 == True:
            options = ["British Empire", "Welfare"]
            corpus_name1 = "British Empire"
            corpus_name5 = "Welfare"
            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df5 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df1["topic"] = "British Empire"
            df5["topic"] = "Welfare"
            df = pd.concat([df1, df5])

            df1_pair = pd.read_csv(r"data/BritishEmpire Pair.csv", index_col=0)
            df5_pair = pd.read_csv(r"data/Welfare Pair.csv", index_col=0)
            df1_pair["topic"] = "British Empire"
            df5_pair["topic"] = "Welfare"
            df_pair = pd.concat([df1_pair, df5_pair])


        elif option1 == False and option2 == True and option3 == True and option4 == False and option5 == False:
            options = ["DDay", "Hypocrisy"]
            corpus_name1 = "DDay"
            corpus_name2 = "Hypocrisy"
            df1 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df2 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df1["topic"] = "DDay"
            df2["topic"] = "Hypocrisy"
            df = pd.concat([df1, df2])

            df1_pair = pd.read_csv(r"data/DDay Pair.csv", index_col=0)
            df2_pair = pd.read_csv(r"data/Hypocrisy Pair.csv", index_col=0)
            df1_pair["topic"] = "DDay"
            df2_pair["topic"] = "Hypocrisy"
            df_pair = pd.concat([df1_pair, df2_pair])

        elif option1 == False and option2 == True and option3 == False and option4 == True and option5 == False:
            options = ["British Empire", "DDay", "Hypocrisy", "Money", "Welfare"]
            corpus_name1 = "DDay"
            corpus_name2 = "Money"
            df2 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df4 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df2["topic"] = "DDay"
            df4["topic"] = "Money"
            df = pd.concat([df2, df4])

            df2_pair = pd.read_csv(r"data/DDay Pair.csv", index_col=0)
            df4_pair = pd.read_csv(r"data/Money Pair.csv", index_col=0)
            df2_pair["topic"] = "DDay"
            df4_pair["topic"] = "Money"
            df_pair = pd.concat([df2_pair, df4_pair])

        elif option1 == False and option2 == True and option3 == False and option4 == False and option5 == True:
            options = ["British Empire", "DDay", "Hypocrisy", "Money", "Welfare"]
            corpus_name1 = "DDay"
            corpus_name2 = "Welfare"
            df2 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df5 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df2["topic"] = "DDay"
            df5["topic"] = "Welfare"
            df = pd.concat([df2, df5])

            df2_pair = pd.read_csv(r"data/DDay Pair.csv", index_col=0)
            df5_pair = pd.read_csv(r"data/Welfare Pair.csv", index_col=0)
            df2_pair["topic"] = "DDay"
            df5_pair["topic"] = "Welfare"
            df_pair = pd.concat([df2_pair, df5_pair])

        elif option1 == False and option2 == False and option3 == True and option4 == True and option5 == False:
            options = ["Hypocrisy", "Money"]
            corpus_name1 = "Hypocrisy"
            corpus_name2 = "Money"
            df3 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df4 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df3["topic"] = "Hypocrisy"
            df4["topic"] = "Money"
            df = pd.concat([df3, df4])

            df3_pair = pd.read_csv(r"data/Hypocrisy Pair.csv", index_col=0)
            df4_pair = pd.read_csv(r"data/Money Pair.csv", index_col=0)
            df3_pair["topic"] = "Hypocrisy"
            df4_pair["topic"] = "Money"
            df_pair = pd.concat([df3_pair, df4_pair])

        elif option1 == False and option2 == False and option3 == True and option4 == False and option5 == True:
            options = ["Hypocrisy", "Welfare"]
            corpus_name3 = "Hypocrisy"
            corpus_name5 = "Welfare"
            df3 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df5 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df3["topic"] = "Hypocrisy"
            df5["topic"] = "Welfare"
            df = pd.concat([df3, df5])

            df3_pair = pd.read_csv(r"data/Hypocrisy Pair.csv", index_col=0)
            df5_pair = pd.read_csv(r"data/Welfare Pair.csv", index_col=0)
            df3_pair["topic"] = "Hypocrisy"
            df5_pair["topic"] = "Welfare"
            df_pair = pd.concat([df3_pair, df5_pair])

        elif option1 == False and option2 == False and option3 == False and option4 == True and option5 == True:
            options = ["Money", "Welfare"]
            corpus_name1 = "Money"
            corpus_name2 = "Welfare"
            df4 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df5 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df4["topic"] = "Money"
            df5["topic"] = "Welfare"
            df = pd.concat([df4, df5])

            df4_pair = pd.read_csv(r"data/Money Pair.csv", index_col=0)
            df5_pair = pd.read_csv(r"data/Welfare Pair.csv", index_col=0)
            df4_pair["topic"] = "Money"
            df5_pair["topic"] = "Welfare"
            df_pair = pd.concat([df4_pair, df5_pair])

        elif option1 == True and option2 == True and option3 == True and option4 == False and option5 == False:
            options = ["British Empire", "DDay", "Hypocrisy", "Money", "Welfare"]
            corpus_name1 = "British Empire"
            corpus_name2 = "DDay"
            corpus_name3 = "Hypocrisy"
            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df2 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df3 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df1["topic"] = "British Empire"
            df2["topic"] = "DDay"
            df3["topic"] = "Hypocrisy"
            df = pd.concat([df1, df2])
            df = pd.concat([df, df3])

            df1_pair = pd.read_csv(r"data/BritishEmpire Pair.csv", index_col=0)
            df2_pair = pd.read_csv(r"data/DDay Pair.csv", index_col=0)
            df3_pair = pd.read_csv(r"data/Hypocrisy Pair.csv", index_col=0)
            df1_pair["topic"] = "British Empire"
            df2_pair["topic"] = "DDay"
            df3_pair["topic"] = "Hypocrisy"
            df_pair = pd.concat([df1_pair, df2_pair])
            df_pair = pd.concat([df_pair, df3_pair])

        elif option1 == True and option2 == True and option3 == False and option4 == True and option5 == False:
            options = ["British Empire", "DDay", "Money"]
            corpus_name1 = "British Empire"
            corpus_name2 = "DDay"
            corpus_name3 = "Money"
            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df2 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df4 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df1["topic"] = "British Empire"
            df2["topic"] = "DDay"
            df4["topic"] = "Money"
            df = pd.concat([df1, df2])
            df = pd.concat([df, df4])

            df1_pair = pd.read_csv(r"data/BritishEmpire Pair.csv", index_col=0)
            df2_pair = pd.read_csv(r"data/DDay Pair.csv", index_col=0)
            df4_pair = pd.read_csv(r"data/Money Pair.csv", index_col=0)
            df1_pair["topic"] = "British Empire"
            df2_pair["topic"] = "DDay"
            df4_pair["topic"] = "Money"
            df_pair = pd.concat([df1_pair, df2_pair])
            df_pair = pd.concat([df_pair, df4_pair])



        elif option1 == True and option2 == True and option3 == False and option4 == False and option5 == True:
            options = ["British Empire", "DDay", "Welfare"]
            corpus_name1 = "British Empire"
            corpus_name2 = "DDay"
            corpus_name3 = "Welfare"
            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df2 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df5 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df1["topic"] = "British Empire"
            df2["topic"] = "DDay"
            df5["topic"] = "Welfare"
            df = pd.concat([df1, df2])
            df = pd.concat([df, df5])

            df1_pair = pd.read_csv(r"data/BritishEmpire Pair.csv", index_col=0)
            df2_pair = pd.read_csv(r"data/DDay Pair.csv", index_col=0)
            df5_pair = pd.read_csv(r"data/Welfare Pair.csv", index_col=0)
            df1_pair["topic"] = "British Empire"
            df2_pair["topic"] = "DDay"
            df5_pair["topic"] = "Welfare"
            df_pair = pd.concat([df1_pair, df2_pair])
            df_pair = pd.concat([df_pair, df5_pair])

        elif option1 == True and option2 == False and option3 == True and option4 == True and option5 == False:
            options = ["British Empire", "Hypocrisy", "Money"]
            corpus_name1 = "British Empire"
            corpus_name2 = "Hypocrisy"
            corpus_name3 = "Money"
            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df3 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df4 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df1["topic"] = "British Empire"
            df3["topic"] = "Hypocrisy"
            df4["topic"] = "Money"
            df = pd.concat([df1, df3])
            df = pd.concat([df, df4])

            df1_pair = pd.read_csv(r"data/BritishEmpire Pair.csv", index_col=0)
            df3_pair = pd.read_csv(r"data/Hypocrisy Pair.csv", index_col=0)
            df4_pair = pd.read_csv(r"data/Money Pair.csv", index_col=0)
            df1_pair["topic"] = "British Empire"
            df3_pair["topic"] = "Hypocrisy"
            df4_pair["topic"] = "Money"
            df_pair = pd.concat([df1_pair, df3_pair])
            df_pair = pd.concat([df_pair, df4_pair])


        elif option1 == True and option2 == False and option3 == True and option4 == False and option5 == True:
            options = ["British Empire", "Hypocrisy", "Welfare"]
            corpus_name1 = "British Empire"
            corpus_name2 = "Hypocrisy"
            corpus_name3 = "Welfare"
            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df3 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df5 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df1["topic"] = "British Empire"
            df3["topic"] = "Hypocrisy"
            df5["topic"] = "Welfare"
            df = pd.concat([df1, df3])
            df = pd.concat([df, df5])

            df1_pair = pd.read_csv(r"data/BritishEmpire Pair.csv", index_col=0)
            df3_pair = pd.read_csv(r"data/Hypocrisy Pair.csv", index_col=0)
            df5_pair = pd.read_csv(r"data/Welfare Pair.csv", index_col=0)
            df1_pair["topic"] = "British Empire"
            df3_pair["topic"] = "Hypocrisy"
            df5_pair["topic"] = "Welfare"
            df_pair = pd.concat([df1_pair, df3_pair])
            df_pair = pd.concat([df_pair, df5_pair])

        elif option1 == True and option2 == False and option3 == False and option4 == True and option5 == True:
            options = ["British Empire", "Money", "Welfare"]
            corpus_name1 = "British Empire"
            corpus_name2 = "Money"
            corpus_name3 = "Welfare"
            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df2 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df3 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df1["topic"] = "British Empire"
            df2["topic"] = "Money"
            df3["topic"] = "Welfare"
            df = pd.concat([df1, df2])
            df = pd.concat([df, df3])

            df1_pair = pd.read_csv(r"data/BritishEmpire Pair.csv", index_col=0)
            df2_pair = pd.read_csv(r"data/Money Pair.csv", index_col=0)
            df3_pair = pd.read_csv(r"data/Welfare Pair.csv", index_col=0)
            df1_pair["topic"] = "British Empire"
            df2_pair["topic"] = "Money"
            df3_pair["topic"] = "Welfare"
            df_pair = pd.concat([df1_pair, df2_pair])
            df_pair = pd.concat([df_pair, df3_pair])

        elif option1 == False and option2 == True and option3 == True and option4 == True and option5 == False:
            options = ["DDay", "Hypocrisy", "Money"]
            corpus_name1 = "DDay"
            corpus_name2 = "Hypocrisy"
            corpus_name3 = "Money"
            df2 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df3 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df4 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df2["topic"] = "DDay"
            df3["topic"] = "Hypocrisy"
            df4["topic"] = "Money"
            df = pd.concat([df2, df3])
            df = pd.concat([df, df4])

            df2_pair = pd.read_csv(r"data/DDay Pair.csv", index_col=0)
            df3_pair = pd.read_csv(r"data/Hypocrisy Pair.csv", index_col=0)
            df4_pair = pd.read_csv(r"data/Money Pair.csv", index_col=0)
            df2_pair["topic"] = "DDay"
            df3_pair["topic"] = "Hypocrisy"
            df4_pair["topic"] = "Money"
            df_pair = pd.concat([df2_pair, df3_pair])
            df_pair = pd.concat([df_pair, df4_pair])


        elif option1 == False and option2 == True and option3 == True and option4 == False and option5 == True:
            options = ["DDay", "Hypocrisy", "Welfare"]
            corpus_name1 = "DDay"
            corpus_name2 = "Hypocrisy"
            corpus_name3 = "Welfare"
            df2 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df3 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df5 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df2["topic"] = "DDay"
            df3["topic"] = "Hypocrisy"
            df5["topic"] = "Welfare"
            df = pd.concat([df2, df3])
            df = pd.concat([df, df5])

            df2_pair = pd.read_csv(r"data/DDay Pair.csv", index_col=0)
            df3_pair = pd.read_csv(r"data/Hypocrisy Pair.csv", index_col=0)
            df5_pair = pd.read_csv(r"data/Welfare Pair.csv", index_col=0)
            df2_pair["topic"] = "DDay"
            df3_pair["topic"] = "Hypocrisy"
            df5_pair["topic"] = "Welfare"
            df_pair = pd.concat([df2_pair, df3_pair])
            df_pair = pd.concat([df_pair, df5_pair])

        elif option1 == False and option2 == True and option3 == False and option4 == True and option5 == True:
            options = ["DDay", "Money", "Welfare"]
            corpus_name1 = "DDay"
            corpus_name2 = "Money"
            corpus_name3 = "Welfare"
            df2 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df4 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df5 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df2["topic"] = "DDay"
            df4["topic"] = "Money"
            df5["topic"] = "Welfare"
            df = pd.concat([df2, df4])
            df = pd.concat([df, df5])

            df2_pair = pd.read_csv(r"data/DDay Pair.csv", index_col=0)
            df4_pair = pd.read_csv(r"data/Money Pair.csv", index_col=0)
            df5_pair = pd.read_csv(r"data/Welfare Pair.csv", index_col=0)
            df2_pair["topic"] = "DDay"
            df4_pair["topic"] = "Money"
            df5_pair["topic"] = "Welfare"
            df_pair = pd.concat([df2_pair, df4_pair])
            df_pair = pd.concat([df_pair, df5_pair])

        elif option1 == False and option2 == False and option3 == True and option4 == True and option5 == True:
            options = ["British Empire", "DDay", "Hypocrisy", "Money", "Welfare"]
            corpus_name1 = "Hypocrisy"
            corpus_name2 = "Money"
            corpus_name3 = "Welfare"
            df3 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df4 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df5 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df3["topic"] = "Hypocrisy"
            df4["topic"] = "Money"
            df5["topic"] = "Welfare"
            df = pd.concat([df3, df4])
            df = pd.concat([df, df5])

            df3_pair = pd.read_csv(r"data/Hypocrisy Pair.csv", index_col=0)
            df4_pair = pd.read_csv(r"data/Money Pair.csv", index_col=0)
            df5_pair = pd.read_csv(r"data/Welfare Pair.csv", index_col=0)
            df3_pair["topic"] = "Hypocrisy"
            df4_pair["topic"] = "Money"
            df5_pair["topic"] = "Welfare"
            df_pair = pd.concat([df3_pair, df4_pair])
            df_pair = pd.concat([df_pair, df5_pair])


        elif option1 == True and option2 == True and option3 == True and option4 == True and option5 == False:
            options = ["British Empire", "DDay", "Hypocrisy", "Money"]
            corpus_name1 = "British Empire"
            corpus_name2 = "DDay"
            corpus_name3 = "Hypocrisy"
            corpus_name4 = "Money"
            df1_pair = pd.read_csv(r"data/BritishEmpire Pair.csv", index_col=0)
            df2_pair = pd.read_csv(r"data/DDay Pair.csv", index_col=0)
            df3_pair = pd.read_csv(r"data/Hypocrisy Pair.csv", index_col=0)
            df4_pair = pd.read_csv(r"data/Money Pair.csv", index_col=0)
            df1_pair["topic"] = "British Empire"
            df2_pair["topic"] = "DDay"
            df3_pair["topic"] = "Hypocrisy"
            df4_pair["topic"] = "Money"
            df_pair = pd.concat([df1_pair, df2_pair])
            df_pair = pd.concat([df_pair, df3_pair])
            df_pair = pd.concat([df_pair, df4_pair])

            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df2 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df3 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df4 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df1["topic"] = "British Empire"
            df2["topic"] = "DDay"
            df3["topic"] = "Hypocrisy"
            df4["topic"] = "Money"
            df = pd.concat([df1, df2])
            df = pd.concat([df, df3])
            df = pd.concat([df, df4])

        elif option1 == True and option2 == True and option3 == True and option4 == False and option5 == True:
            options = ["British Empire", "DDay", "Hypocrisy", "Welfare State"]
            corpus_name1 = "British Empire"
            corpus_name2 = "DDay"
            corpus_name3 = "Hypocrisy"
            corpus_name4 = "Welfare State"
            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df2 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df3 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df4 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df1["topic"] = "British Empire"
            df2["topic"] = "DDay"
            df3["topic"] = "Hypocrisy"
            df4["topic"] = "Welfare State"
            df = pd.concat([df1, df2])
            df = pd.concat([df, df3])
            df = pd.concat([df, df4])

            df1_pair = pd.read_csv(r"data/BritishEmpire Pair.csv", index_col=0)
            df2_pair = pd.read_csv(r"data/DDay Pair.csv", index_col=0)
            df3_pair = pd.read_csv(r"data/Hypocrisy Pair.csv", index_col=0)
            df4_pair = pd.read_csv(r"data/Welfare Pair.csv", index_col=0)
            df1_pair["topic"] = "British Empire"
            df2_pair["topic"] = "DDay"
            df3_pair["topic"] = "Hypocrisy"
            df4_pair["topic"] = "Welfare State"
            df_pair = pd.concat([df1_pair, df2_pair])
            df_pair = pd.concat([df_pair, df3_pair])
            df_pair = pd.concat([df_pair, df4_pair])

        elif option1 == True and option2 == True and option3 == False and option4 == True and option5 == True:
            options = ["British Empire", "DDay", "Money", "Welfare"]
            corpus_name1 = "British Empire"
            corpus_name2 = "DDay"
            corpus_name3 = "Money"
            corpus_name4 = "Welfare"
            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df2 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df4 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df5 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df1["topic"] = "British Empire"
            df2["topic"] = "DDay"
            df4["topic"] = "Money"
            df5["topic"] = "Welfare"
            df = pd.concat([df1, df2])
            df = pd.concat([df, df4])
            df = pd.concat([df, df5])

            df1_pair = pd.read_csv(r"data/BritishEmpire Pair.csv", index_col=0)
            df2_pair = pd.read_csv(r"data/DDay Pair.csv", index_col=0)
            df4_pair = pd.read_csv(r"data/Money Pair.csv", index_col=0)
            df5_pair = pd.read_csv(r"data/Welfare Pair.csv", index_col=0)
            df1_pair["topic"] = "British Empire"
            df2_pair["topic"] = "DDay"
            df4_pair["topic"] = "Money"
            df5_pair["topic"] = "Welfare"
            df_pair = pd.concat([df1_pair, df2_pair])
            df_pair = pd.concat([df_pair, df4_pair])
            df_pair = pd.concat([df_pair, df5_pair])

        elif option1 == True and option2 == False and option3 == True and option4 == True and option5 == True:
            options = ["British Empire", "Hypocrisy", "Money", "Welfare"]
            corpus_name1 = "British Empire"
            corpus_name2 = "Hypocrisy"
            corpus_name3 = "Money"
            corpus_name4 = "Welfare"
            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df3 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df4 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df5 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df1["topic"] = "British Empire"
            df3["topic"] = "Hypocrisy"
            df4["topic"] = "Money"
            df5["topic"] = "Welfare"
            df = pd.concat([df1, df3])
            df = pd.concat([df, df4])
            df = pd.concat([df, df5])

            df1_pair = pd.read_csv(r"data/BritishEmpire Pair.csv", index_col=0)
            df3_pair = pd.read_csv(r"data/Hypocrisy Pair.csv", index_col=0)
            df4_pair = pd.read_csv(r"data/Money Pair.csv", index_col=0)
            df5_pair = pd.read_csv(r"data/Welfare Pair.csv", index_col=0)
            df1_pair["topic"] = "British Empire"
            df3_pair["topic"] = "Hypocrisy"
            df4_pair["topic"] = "Money"
            df5_pair["topic"] = "Welfare"
            df_pair = pd.concat([df1_pair, df3_pair])
            df_pair = pd.concat([df_pair, df4_pair])
            df_pair = pd.concat([df_pair, df5_pair])

        elif option1 == False and option2 == True and option3 == True and option4 == True and option5 == True:
            options = ["DDay", "Hypocrisy", "Money", "Welfare"]
            corpus_name1 = "DDay"
            corpus_name2 = "Hypocrisy"
            corpus_name3 = "Money"
            corpus_name4 = "Welfare"
            df2 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df3 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df4 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df5 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df2["topic"] = "DDay"
            df3["topic"] = "Hypocrisy"
            df4["topic"] = "Money"
            df5["topic"] = "Welfare"
            df = pd.concat([df2, df3])
            df = pd.concat([df, df4])
            df = pd.concat([df, df5])

            df2_pair = pd.read_csv(r"data/DDay Pair.csv", index_col=0)
            df3_pair = pd.read_csv(r"data/Hypocrisy Pair.csv", index_col=0)
            df4_pair = pd.read_csv(r"data/Money Pair.csv", index_col=0)
            df5_pair = pd.read_csv(r"data/Welfare Pair.csv", index_col=0)
            df2_pair["topic"] = "DDay"
            df3_pair["topic"] = "Hypocrisy"
            df4_pair["topic"] = "Money"
            df5_pair["topic"] = "Welfare"
            df_pair = pd.concat([df2_pair, df3_pair])
            df_pair = pd.concat([df_pair, df4_pair])
            df_pair = pd.concat([df_pair, df5_pair])

        elif option1 == True and option2 == True and option3 == True and option4 == True and option5 == True:
            options = ["British Empire", "DDay", "Hypocrisy", "Money", "Welfare"]
            corpus_name1 = "British Empire"
            corpus_name2 = "DDay"
            corpus_name3 = "Hypocrisy"
            corpus_name4 = "Money"
            corpus_name5 = "Welfare"
            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df2 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df3 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df4 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df5 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df1["topic"] = "British Empire"
            df2["topic"] = "DDay"
            df3["topic"] = "Hypocrisy"
            df4["topic"] = "Money"
            df5["topic"] = "Welfare"
            df = pd.concat([df1, df2])
            df = pd.concat([df, df3])
            df = pd.concat([df, df4])
            df = pd.concat([df, df5])

            df1_pair = pd.read_csv(r"data/BritishEmpire Pair.csv", index_col=0)
            df2_pair = pd.read_csv(r"data/DDay Pair.csv", index_col=0)
            df3_pair = pd.read_csv(r"data/Hypocrisy Pair.csv", index_col=0)
            df4_pair = pd.read_csv(r"data/Money Pair.csv", index_col=0)
            df5_pair = pd.read_csv(r"data/Welfare Pair.csv", index_col=0)
            df1_pair["topic"] = "British Empire"
            df2_pair["topic"] = "DDay"
            df3_pair["topic"] = "Hypocrisy"
            df4_pair["topic"] = "Money"
            df5_pair["topic"] = "Welfare"
            df_pair = pd.concat([df1_pair, df2_pair])
            df_pair = pd.concat([df_pair, df3_pair])
            df_pair = pd.concat([df_pair, df4_pair])
            df_pair = pd.concat([df_pair, df5_pair])


        else:
            st.warning("**You need to select at least one to proceed**")
            options = list()
            df = pd.DataFrame()

        if len(df) != 0:
            df["speaker"] = df["Text"].apply(lambda x: x.split(":")[0].strip() if len(re.findall("^(.*) :", x))>0 else "")

        st.write("****************************")
        st.subheader("Analysis Units")
        contents_radio2 = st.radio("", ("ADU-Based Analysis",
                                        "In/Output-Based Analysis",
                                        "Entity-Based Analysis"), index=0)
        st.write("****************************")
        st.subheader("Analytics Module")
        if contents_radio2 == "ADU-Based Analysis":
            contents_radio4 = st.radio("",
                                       ("Word Cloud",
                                        "Text Distribution",), index=0)
        elif contents_radio2 == "In/Output-Based Analysis":
            contents_radio4 = st.radio("",
                                       ("Word Cloud",
                                        "Text Distribution",), index=0)
        elif contents_radio2 == "Entity-Based Analysis":
            contents_radio4 = st.radio("",
                                       ("Interlocutor Distribution",
                                        'Interlocutor Score',
                                        "Interlocutor Argumentative Network",
                                        "Interlocutor Argumentative Interaction",), index=0)

    elif contents_radio1 == "Comparative Corpora Analysis":
        st.subheader("Choose Corpora")
        corpus_option = st.checkbox("Moral Maze", key="MoralMaze_Single", value=True)
        st.write('**Topic:**')
        option1 = st.checkbox("British Empire", key="BritishEmpire_Single", value=False)
        option2 = st.checkbox("DDay", key="DDay_Single", value=False)
        option3 = st.checkbox("Hypocrisy", key="Hypocrisy_Single", value=False)
        option4 = st.checkbox("Money", key="Money_Single", value=False)
        option5 = st.checkbox("Welfare State", key="Welfare_Single", value=False)

        if option1 == True and option2 == True and option3 == False and option4 == False and option5 == False:
            options = ["British Empire", "DDay"]
            corpus_name1 = "British Empire"
            corpus_name2 = "DDay"
            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df2 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df1["topic"] = "British Empire"
            df2["topic"] = "DDay"
            df = pd.concat([df1, df2])

        elif option1 == True and option2 == False and option3 == True and option4 == False and option5 == False:
            options = ["British Empire", "Hypocrisy"]
            corpus_name1 = "British Empire"
            corpus_name2 = "Hypocrisy"
            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df3 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)

            df1["topic"] = "British Empire"
            df3["topic"] = "Hypocrisy"
            df = pd.concat([df1, df3])

        elif option1 == True and option2 == False and option3 == False and option4 == True and option5 == False:
            options = ["British Empire", "Money"]
            corpus_name1 = "British Empire"
            corpus_name2 = "Money"
            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df2 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df1["topic"] = "British Empire"
            df2["topic"] = "Money"
            df = pd.concat([df1, df2])

        elif option1 == True and option2 == False and option3 == False and option4 == False and option5 == True:
            options = ["British Empire", "Welfare"]
            corpus_name1 = "British Empire"
            corpus_name5 = "Welfare"
            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df5 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df1["topic"] = "British Empire"
            df5["topic"] = "Welfare"
            df = pd.concat([df1, df5])

        elif option1 == False and option2 == True and option3 == True and option4 == False and option5 == False:
            options = ["DDay", "Hypocrisy"]
            corpus_name1 = "DDay"
            corpus_name2 = "Hypocrisy"
            df1 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df2 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df1["topic"] = "DDay"
            df2["topic"] = "Hypocrisy"
            df = pd.concat([df1, df2])

        elif option1 == False and option2 == True and option3 == False and option4 == True and option5 == False:
            options = ["British Empire", "DDay", "Hypocrisy", "Money", "Welfare"]
            corpus_name1 = "DDay"
            corpus_name2 = "Money"
            df2 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df4 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df2["topic"] = "DDay"
            df4["topic"] = "Money"
            df = pd.concat([df2, df4])

        elif option1 == False and option2 == True and option3 == False and option4 == False and option5 == True:
            options = ["British Empire", "DDay", "Hypocrisy", "Money", "Welfare"]
            corpus_name1 = "DDay"
            corpus_name2 = "Welfare"
            df2 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df5 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df2["topic"] = "DDay"
            df5["topic"] = "Welfare"
            df = pd.concat([df2, df5])

        elif option1 == False and option2 == False and option3 == True and option4 == True and option5 == False:
            options = ["Hypocrisy", "Money"]
            corpus_name1 = "Hypocrisy"
            corpus_name2 = "Money"
            df3 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df4 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df3["topic"] = "Hypocrisy"
            df4["topic"] = "Money"
            df = pd.concat([df3, df4])

        elif option1 == False and option2 == False and option3 == True and option4 == False and option5 == True:
            options = ["Hypocrisy", "Welfare"]
            corpus_name3 = "Hypocrisy"
            corpus_name5 = "Welfare"
            df3 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df5 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df3["topic"] = "Hypocrisy"
            df5["topic"] = "Welfare"
            df = pd.concat([df3, df5])

        elif option1 == False and option2 == False and option3 == False and option4 == True and option5 == True:
            options = ["Money", "Welfare"]
            corpus_name1 = "Money"
            corpus_name2 = "Welfare"
            df4 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df5 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df4["topic"] = "Money"
            df5["topic"] = "Welfare"
            df = pd.concat([df4, df5])

        elif option1 == True and option2 == True and option3 == True and option4 == False and option5 == False:
            options = ["British Empire", "DDay", "Hypocrisy", "Money", "Welfare"]
            corpus_name1 = "British Empire"
            corpus_name2 = "DDay"
            corpus_name3 = "Hypocrisy"
            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df2 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df3 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df1["topic"] = "British Empire"
            df2["topic"] = "DDay"
            df3["topic"] = "Hypocrisy"
            df = pd.concat([df1, df2])
            df = pd.concat([df, df3])

        elif option1 == True and option2 == True and option3 == False and option4 == True and option5 == False:
            options = ["British Empire", "DDay", "Money"]
            corpus_name1 = "British Empire"
            corpus_name2 = "DDay"
            corpus_name3 = "Money"
            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df2 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df4 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df1["topic"] = "British Empire"
            df2["topic"] = "DDay"
            df4["topic"] = "Money"
            df = pd.concat([df1, df2])
            df = pd.concat([df, df4])

        elif option1 == True and option2 == True and option3 == False and option4 == False and option5 == True:
            options = ["British Empire", "DDay", "Welfare"]
            corpus_name1 = "British Empire"
            corpus_name2 = "DDay"
            corpus_name3 = "Welfare"
            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df2 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df5 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df1["topic"] = "British Empire"
            df2["topic"] = "DDay"
            df5["topic"] = "Welfare"
            df = pd.concat([df1, df2])
            df = pd.concat([df, df5])

        elif option1 == True and option2 == False and option3 == True and option4 == True and option5 == False:
            options = ["British Empire", "Hypocrisy", "Money"]
            corpus_name1 = "British Empire"
            corpus_name2 = "Hypocrisy"
            corpus_name3 = "Money"
            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df3 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df4 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df1["topic"] = "British Empire"
            df3["topic"] = "Hypocrisy"
            df4["topic"] = "Money"
            df = pd.concat([df1, df3])
            df = pd.concat([df, df4])


        elif option1 == True and option2 == False and option3 == True and option4 == False and option5 == True:
            options = ["British Empire", "Hypocrisy", "Welfare"]
            corpus_name1 = "British Empire"
            corpus_name2 = "Hypocrisy"
            corpus_name3 = "Welfare"
            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df3 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df5 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df1["topic"] = "British Empire"
            df3["topic"] = "Hypocrisy"
            df5["topic"] = "Welfare"
            df = pd.concat([df1, df3])
            df = pd.concat([df, df5])

        elif option1 == True and option2 == False and option3 == False and option4 == True and option5 == True:
            options = ["British Empire", "Money", "Welfare"]
            corpus_name1 = "British Empire"
            corpus_name2 = "Money"
            corpus_name3 = "Welfare"
            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df2 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df3 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df1["topic"] = "British Empire"
            df2["topic"] = "Money"
            df3["topic"] = "Welfare"
            df = pd.concat([df1, df2])
            df = pd.concat([df, df3])
        elif option1 == False and option2 == True and option3 == True and option4 == True and option5 == False:
            options = ["DDay", "Hypocrisy", "Money"]
            corpus_name1 = "DDay"
            corpus_name2 = "Hypocrisy"
            corpus_name3 = "Money"
            df2 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df3 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df4 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df2["topic"] = "DDay"
            df3["topic"] = "Hypocrisy"
            df4["topic"] = "Money"
            df = pd.concat([df2, df3])
            df = pd.concat([df, df4])
        elif option1 == False and option2 == True and option3 == True and option4 == False and option5 == True:
            options = ["DDay", "Hypocrisy", "Welfare"]
            corpus_name1 = "DDay"
            corpus_name2 = "Hypocrisy"
            corpus_name3 = "Welfare"
            df2 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df3 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df5 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df2["topic"] = "DDay"
            df3["topic"] = "Hypocrisy"
            df5["topic"] = "Welfare"
            df = pd.concat([df2, df3])
            df = pd.concat([df, df5])

        elif option1 == False and option2 == True and option3 == False and option4 == True and option5 == True:
            options = ["DDay", "Money", "Welfare"]
            corpus_name1 = "DDay"
            corpus_name2 = "Money"
            corpus_name3 = "Welfare"
            df2 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df4 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df5 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df2["topic"] = "DDay"
            df4["topic"] = "Money"
            df5["topic"] = "Welfare"
            df = pd.concat([df2, df4])
            df = pd.concat([df, df5])

        elif option1 == False and option2 == False and option3 == True and option4 == True and option5 == True:
            options = ["British Empire", "DDay", "Hypocrisy", "Money", "Welfare"]
            corpus_name1 = "Hypocrisy"
            corpus_name2 = "Money"
            corpus_name3 = "Welfare"
            df3 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df4 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df5 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df3["topic"] = "Hypocrisy"
            df4["topic"] = "Money"
            df5["topic"] = "Welfare"
            df = pd.concat([df3, df4])
            df = pd.concat([df, df5])


        elif option1 == True and option2 == True and option3 == True and option4 == True and option5 == False:
            options = ["British Empire", "DDay", "Hypocrisy", "Money"]
            corpus_name1 = "British Empire"
            corpus_name2 = "DDay"
            corpus_name3 = "Hypocrisy"
            corpus_name4 = "Money"
            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df2 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df3 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df4 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df1["topic"] = "British Empire"
            df2["topic"] = "DDay"
            df3["topic"] = "Hypocrisy"
            df4["topic"] = "Money"
            df = pd.concat([df1, df2])
            df = pd.concat([df, df3])
            df = pd.concat([df, df4])

        elif option1 == True and option2 == True and option3 == True and option4 == False and option5 == True:
            options = ["British Empire", "DDay", "Hypocrisy", "Welfare State"]
            corpus_name1 = "British Empire"
            corpus_name2 = "DDay"
            corpus_name3 = "Hypocrisy"
            corpus_name4 = "Welfare State"
            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df2 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df3 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df4 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df1["topic"] = "British Empire"
            df2["topic"] = "DDay"
            df3["topic"] = "Hypocrisy"
            df4["topic"] = "Welfare State"
            df = pd.concat([df1, df2])
            df = pd.concat([df, df3])
            df = pd.concat([df, df4])

        elif option1 == True and option2 == True and option3 == False and option4 == True and option5 == True:
            options = ["British Empire", "DDay", "Money", "Welfare"]
            corpus_name1 = "British Empire"
            corpus_name2 = "DDay"
            corpus_name3 = "Money"
            corpus_name4 = "Welfare"
            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df2 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df4 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df5 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df1["topic"] = "British Empire"
            df2["topic"] = "DDay"
            df4["topic"] = "Money"
            df5["topic"] = "Welfare"
            df = pd.concat([df1, df2])
            df = pd.concat([df, df4])
            df = pd.concat([df, df5])

        elif option1 == True and option2 == False and option3 == True and option4 == True and option5 == True:
            options = ["British Empire","Hypocrisy", "Money", "Welfare"]
            corpus_name1 = "British Empire"
            corpus_name2 = "Hypocrisy"
            corpus_name3 = "Money"
            corpus_name4 = "Welfare"
            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df3 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df4 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df5 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df1["topic"] = "British Empire"
            df3["topic"] = "Hypocrisy"
            df4["topic"] = "Money"
            df5["topic"] = "Welfare"
            df = pd.concat([df1, df3])
            df = pd.concat([df, df4])
            df = pd.concat([df, df5])

        elif option1 == False and option2 == True and option3 == True and option4 == True and option5 == True:
            options = ["DDay", "Hypocrisy", "Money", "Welfare"]
            corpus_name1 = "DDay"
            corpus_name2 = "Hypocrisy"
            corpus_name3 = "Money"
            corpus_name4 = "Welfare"
            df2 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df3 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df4 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df5 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df2["topic"] = "DDay"
            df3["topic"] = "Hypocrisy"
            df4["topic"] = "Money"
            df5["topic"] = "Welfare"
            df = pd.concat([df2, df3])
            df = pd.concat([df, df4])
            df = pd.concat([df, df5])

        elif option1 == True and option2 == True and option3 == True and option4 == True and option5 == True:
            options = ["British Empire", "DDay", "Hypocrisy", "Money","Welfare"]
            corpus_name1 = "British Empire"
            corpus_name2 = "DDay"
            corpus_name3 = "Hypocrisy"
            corpus_name4 = "Money"
            corpus_name5 = "Welfare"
            df1 = pd.read_csv(r"data/BritishEmpire_Logo_Moral.csv", index_col=0)
            df2 = pd.read_csv(r"data/DDay_Logo_Moral.csv", index_col=0)
            df3 = pd.read_csv(r"data/Hypocrisy_Logo_Moral.csv", index_col=0)
            df4 = pd.read_csv(r"data/Money_Logo_Moral.csv", index_col=0)
            df5 = pd.read_csv(r"data/Welfare_Logo_Moral.csv", index_col=0)
            df1["topic"] = "British Empire"
            df2["topic"] = "DDay"
            df3["topic"] = "Hypocrisy"
            df4["topic"] = "Money"
            df5["topic"] = "Welfare"
            df = pd.concat([df1, df2])
            df = pd.concat([df, df3])
            df = pd.concat([df, df4])
            df = pd.concat([df, df5])


        else:
            st.warning("**You need to select at least two options to proceed**")
            options = list()
            df = pd.DataFrame()

        if len(df) != 0:
            df["speaker"] = df["Text"].apply(lambda x: x.split(":")[0].strip() if len(re.findall("^(.*) :", x))>0 else "")

        st.write("****************************")
        st.subheader("Analysis Units")
        contents_radio2 = st.radio("", ("ADU-Based Analysis",
                                        "In/Output-Based Analysis",
                                        "Entity-Based Analysis"), index=0)
        st.write("****************************")
        st.subheader("Analytics Module")
        if contents_radio2 == "ADU-Based Analysis":
            contents_radio4 = st.radio("",
                                       ("Word Cloud","Text Distribution",), index=0)
        elif contents_radio2 == "In/Output-Based Analysis":
            contents_radio4 = st.radio("",
                                       ("Word Cloud", "Text Distribution",), index=0)
        else:
            contents_radio4 = st.radio("",
                                       ("Interlocutor Distribution",), index=0)

if contents_radio4 == "Main Page":
    MainPage()
elif contents_radio4 == "Word Cloud" and contents_radio1 == "Single Corpus Analysis" and contents_radio2 == "ADU-Based Analysis":
    Moral_Foundation_Word_Cloud(mode="ADU")
elif contents_radio4 == "Word Cloud" and contents_radio1 == "Comparative Corpora Analysis" and contents_radio2 == "ADU-Based Analysis":
    Comparative_Moral_Foundation_Word_Cloud(mode="ADU")
elif contents_radio4 == "Word Cloud" and contents_radio1 == "Single Corpus Analysis" and contents_radio2 == "In/Output-Based Analysis":
    Moral_Foundation_Word_Cloud(mode="ARG")
elif contents_radio4 == "Word Cloud" and contents_radio1 == "Comparative Corpora Analysis" and contents_radio2 == "In/Output-Based Analysis":
    Comparative_Moral_Foundation_Word_Cloud(mode="ARG")
elif contents_radio4 == "Text Distribution" and contents_radio1 == "Single Corpus Analysis" and contents_radio2 == "ADU-Based Analysis":
    Moral_Value_Distribution(mode="ADU")
elif contents_radio4 == "Text Distribution" and contents_radio1 == "Comparative Corpora Analysis" and contents_radio2 == "ADU-Based Analysis":
    Comparative_Moral_Value_Distribution(mode="ADU")
elif contents_radio4 == "Text Distribution" and contents_radio1 == "Single Corpus Analysis" and contents_radio2 == "In/Output-Based Analysis":
    Moral_Value_Distribution(mode="ARG")
elif contents_radio4 == "Text Distribution" and contents_radio1 == "Comparative Corpora Analysis" and contents_radio2 == "In/Output-Based Analysis":
    Comparative_Moral_Value_Distribution(mode="ARG")
elif contents_radio4 == "Top 10 Moral Foundation Words" and contents_radio1 == "Single Corpus Analysis":
    Top10_Moral_Foundation_Word()
elif contents_radio4 == "Interlocutor Argumentative Network" and contents_radio1 == "Single Corpus Analysis":
    Moral_Concern_Network()
elif contents_radio4 == "Interlocutor Score" and contents_radio1 == "Single Corpus Analysis":
    Moral_Concern_User_Score()
elif contents_radio4 == "Interlocutor Distribution" and contents_radio1 == "Single Corpus Analysis":
    Moral_Concern_User_Distribution()
elif contents_radio4 == "Interlocutor Argumentative Interaction" and contents_radio1 == "Single Corpus Analysis":
    Moral_Concern_User_Interaction_Analysis()
elif contents_radio4 == "Interlocutor Distribution" and contents_radio1 == "Comparative Corpora Analysis":
    Compare_Moral_Concern_User_Distribution()
