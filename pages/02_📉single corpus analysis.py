# Contents of ~/my_app/pages/page_2.py
import streamlit as st
from streamlit_tree_select import tree_select
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import networkx as nx
import re
import time

@st.cache_data
def customised_dataloader():
    df_ADU = pd.read_excel("data/Money_ADU_Moral.xlsx",index_col=0)
    df_arg = pd.read_excel("data/Money_ADU_Moral.xlsx",index_col=0)
    return df_ADU, df_arg

def select_corpora():
    nodes = [{"label": "US2016reddit (Real-time Reactions about US Presidential TV Debate on Reddit)", "value": "US2016reddit",
                 "children": [
            {"label": "US2016r1D (Democrats)", "value": "US2016r1D"},
            {"label": "US2016r1G (General)", "value": "US2016r1G"},
            {"label": "US2016r1R (Republicans)", "value": "US2016r1R"},
            ]},

            {"label": "MoralMaze (Live debate on moral issues)", "value": "MoralMaze","children": [
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
    return_select = tree_select(nodes,show_expand_all=True,expand_on_click=True,check_model="leaf")
    # st.write(return_select)
    return return_select


def select_corpora_customisation(customised_dataset):
    dataset_list = list()
    for dataset in customised_dataset:
        dataset_list.append({"label":dataset,"value":dataset})
    nodes = [{"label": "US2016reddit (Real-time Reactions about US Presidential TV Debate on Reddit)",
              "value": "US2016reddit",
              "children": [
                  {"label": "US2016r1D (Democrats)", "value": "US2016r1D"},
                  {"label": "US2016r1G (General)", "value": "US2016r1G"},
                  {"label": "US2016r1R (Republicans)", "value": "US2016r1R"},
              ]},

             {"label": "MoralMaze (Live debate on moral issues)", "value": "MoralMaze", "children": [
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

            {"label": "Customisation", "value": "Customisation",
             "children": dataset_list},
            ]
    return_select = tree_select(nodes,show_expand_all=True,expand_on_click=True,check_model="leaf")
    return return_select

def add_spacelines(number=2):
    for i in range(number):
        st.write("\n")
@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def speak_speration(text):
    try:
        # Extract speaker name
        speaker_end_index = text.index(":")
        speaker_name = text[:speaker_end_index].strip()
    except ValueError:
        speaker_name = "None"
    return speaker_name


def ADU_dataloader(file_list):
    dfs = [pd.read_excel(file) if file.endswith("xlsx") else df_cus[df_cus.dataset.isin([file])] for file in file_list]
    data = pd.concat(dfs, ignore_index=True)
    data = data.filter(regex='^(?!Unnamed)')

    try:
        data["speaker"] = data["Text"].apply(lambda x: speak_speration(x))
        data["Arg-ADUs"] = data["support"] | data["attack"]
        data["Nonarg-ADUs"] = (data[['support', 'attack']].sum(axis=1) == 0)
        data["Support"] = data["support"]
        data["Attack"] = data["attack"]
        data["Input"] = data["attack_input"] | data["support_input"]
        data["Output"] = data["attack_output"] | data["support_output"]
    except Exception as e:
        st.warning("Error about your data format:( please make sure that your data is annotated ADUs")
        data = pd.DataFrame()
    return data
    

def Arg_dataloader(file_list):
    dfs = [pd.read_excel(file) if file.endswith("xlsx") else df_cus[df_cus.dataset.isin([file])] for file in file_list]
    data = pd.concat(dfs, ignore_index=True)
    data = data.filter(regex='^(?!Unnamed)')
    try:
        data = data[(data["connection"] == "RA") | (data["connection"] == "CA")]
        st.write("You have loaded {} arguments".format(len(data)))
        data["Support"] = (data["connection"] == "RA")
        data["Attack"] = (data["connection"] == "CA")
        data["Same speaker"] = (data["same_speakers"] == True)
        data["Different Speakers"]  = (data["different_speakers"] == True)
    except Exception as e:
        st.warning("Error about your data format:( Please make sure that your data is annotated arguments")
        data = pd.DataFrame()
    return data

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


def User_Moral_Concern_Score_Heatmap(df,personal_width,personal_height):
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
    # data = df.to_numpy().T.round(2)
    data = df.round(2)
    fig = px.imshow(data,
                    labels=dict(x="Moral Foundation", y="Speaker", color="User Score"),
                    x=["care+",
                       "care-",
                       "fairness+",
                       "fairness-",
                       "sanctity+",
                       "sanctity-",
                       "authority+",
                       "authority-",
                       "loyalty+",
                       "loyalty-"],
                    y=df.index,text_auto=True,aspect="auto")
    fig.update_layout(width=personal_width, height=personal_height)
    return fig


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
        num_type = "num authority.virtue"
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
        fig.update_layout(barmode='group', xaxis_tickangle=-45, width=500,height=500, title="Top 10 most frequent moral foundation words")
    elif format1 == "number":
        import plotly.express as px
        fig = px.scatter(df, x="words", y="frequency")
        fig.update_traces(hovertemplate='words: %{x} <br>frequency: %{y}')
        fig.update_layout(barmode='group',
                          width=500,
                          height=500,
                          title="Top 10 most frequent moral foundation words"
                          )
        fig.update_xaxes(title_text='Moral foundation words', categoryorder='total descending')
        fig.update_yaxes(title_text='Frequency')
    return fig

def Moral_Foundation_Word_Cloud(df, category_type,valence_type):
    with st.expander("Analysis Result",expanded=True):
        st.write("#### 1. Word Cloud Visualisation")
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

def Moral_value_dynamics_in_argumentation(df, moral_scale, title):
    return None

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
        fig.update_layout(barmode='group', width=500, height=450)
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
        fig.update_layout(barmode='group', width=500, height=450)
        fig.update_traces(textposition='outside', texttemplate="%{y}%")
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
        fig.update_layout(barmode='stack', width=500, height=450)
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
        fig.update_layout(barmode='stack', width=500, height=450)
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
        fig.update_layout(barmode='group', width=500, height=450, legend=dict(
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
        fig.update_layout(barmode='stack', width=500, height=450, legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        fig.update_traces(textposition='auto', texttemplate="%{x}%", width=0.5)
        return fig


def User_Interaction_Analysis(df, df_pair, aspect, moral_foundation):
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



def word_cloud_ADU_module():
    with st.chat_message("assistant"):
        st.write(
            "Awesome! Welcome to the *****Word Cloud feature*****. Please **choose the analytical angle for ADUs and Moral Values**.")
        col1, col_, col2 = st.columns([3,1,3])
        col1.write("#### ADU Property")
        col2.write("#### Moral Dimension")
        with col1:
            ADUsel = st.multiselect("Arg/Non Arg", ["Arg-ADUs", "Nonarg-ADUs"],default=["Arg-ADUs"],placeholder="Choose one or more options")
            if "Arg-ADUs" in ADUsel:
                polaritysel = st.multiselect("Support/Attack", ["Support", "Attack"],placeholder="Choose one or more options")
                structuresel = st.multiselect("Input/Output", ["Input", "Output"],placeholder="Choose one or more options")
            else:
                polaritysel = []
                structuresel = []

        with col2:
            MFsel = st.selectbox("Moral Foundation", ["Care", "Fairness", "Loyalty", "Authority", "Sanctity"])
            sentisel = st.selectbox("Moral Valence", ["Positive", "Negative"])


        if (len(ADUsel) != 0) & (len(polaritysel) != 0) & (len(structuresel) != 0):
            if "Nonarg-ADUs" in ADUsel:
                index2 = (data[polaritysel].sum(axis=1) > 0)
                index3 = (data[structuresel].sum(axis=1) > 0)
                df = data[(data["Nonarg-ADUs"]) == 1]
                df = pd.concat([df, data[index2 & index3]])
                # st.write(df)

            else:
                index2 = (data[polaritysel].sum(axis=1) > 0)
                index3 = (data[structuresel].sum(axis=1) > 0)
                df = data[index2 & index3]
        elif (ADUsel == ["Nonarg-ADUs"]) and (len(polaritysel) == 0) and (len(structuresel) == 0):
            index1 = (data[ADUsel].sum(axis=1) > 0)
            df = data[index1]
            # st.write(df)
        else:
            df = pd.DataFrame()

    if len(df) != 0 and len(MFsel)!=0 and len(sentisel)!=0:
        with st.chat_message("assistant"):
            st.write(
                "Fantastic!🤗 Following your selections, the analysis outcomes will be displayed below.")
            Moral_Foundation_Word_Cloud(df,MFsel,sentisel)
    else:
        st.write(" ")
    return None

def moral_value_distribution_ADU_module():
    with st.chat_message("assistant"):
        st.write(
           "Wonderful! Welcome to the *****Moral Value Distribution***** feature. Kindly **select the analytical perspective for ADUs** and the **evaluation scale for Moral Values**.")
        col1, col_, col2 = st.columns([3,1,3])
        col1.write("#### ADU Property")
        col2.write("#### Moral Scale")
        with col1:
            ADUsel = st.multiselect("Arg/Non Arg", ["Arg-ADUs", "Nonarg-ADUs"],default="Arg-ADUs",placeholder="Choose one or more options")
            if "Arg-ADUs" in ADUsel:
                polaritysel = st.multiselect("Support/Attack", ["Support", "Attack"],placeholder="Choose one or more options")
                structuresel = st.multiselect("Input/Output", ["Input", "Output"],placeholder="Choose one or more options")
            else:
                polaritysel = []
                structuresel = []

        with col2:
            moral_scale = st.radio("", ['Moral vs No moral',
                                        '2 Moral Valences',
                                        '5 Moral Foundations',
                                        '10 Moral Values'], key="moral_scale_mv")

        n_format = st.radio("numerical representations", ['number', 'percentage'], key="moral_scale_nformat")
        if (len(ADUsel) != 0) & (len(polaritysel) != 0) & (len(structuresel) != 0):
            if "Nonarg-ADUs" in ADUsel:
                index2 = (data[polaritysel].sum(axis=1) > 0)
                index3 = (data[structuresel].sum(axis=1) > 0)
                df = data[(data["Nonarg-ADUs"]) == 1]
                df = pd.concat([df, data[index2 & index3]])
                # st.write(df)

            else:
                index2 = (data[polaritysel].sum(axis=1) > 0)
                index3 = (data[structuresel].sum(axis=1) > 0)
                df = data[index2 & index3]
        elif (ADUsel == ["Nonarg-ADUs"]) and (len(polaritysel) == 0) and (len(structuresel) == 0):
            index1 = (data[ADUsel].sum(axis=1) > 0)
            df = data[index1]
            # st.write(df)
        else:
            df = pd.DataFrame()


    if len(df) != 0:
        with st.chat_message("assistant"):
            st.write(
                "Fantastic!🤗 Following your selections, the analysis outcomes will be displayed below.")
            tab1, tab2 = st.tabs(["Bar Chart","Statistic Table"])
            with tab1:
                fig,table = Moral_Foundation_Word_In_Tweet(df, n_format, moral_scale, "ADU",title)
                st.pyplot(fig)
            with tab2:
                fig, table = Moral_Foundation_Word_In_Tweet(df, n_format, moral_scale, "ADU", title)
                st.dataframe(table)
                csv = convert_df(table)

                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='moral_value_distribution.csv',
                    mime='text/csv',
                )
    else:
        st.write(" ")


    return None

def add_datasetname(data_dict):
    dfs = []
    for key, df in data_dict.items():
        df['dataset_name'] = key
        dfs.append(df)

    # Concatenate all DataFrames together
    result = pd.concat(dfs, ignore_index=True)
    return result

def Moral_Foundation_Word_In_Tweet(df, format1, moral_scale,unit_name,title):
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

        column_list = ["morals","no moral foundation words"]
        column_renamemap = {"morals":"Morals",
                            "no moral foundation words":"No morals"}
        palette_map = {"Morals": "#FCFF33",
                       "No morals": "#33BEFF"}
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
        column_list = ["positive moral valence", "negative moral valence"]
        column_renamemap = {"positive moral valence": "Positive",
                            "negative moral valence": "Negative"}
        palette_map = {"Positive": "#239B56",
                       "Negative": "#CB4335"}
        pattern = {"Positive": "-",
                   "Negative": "/"}
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
                       "contains_sanctity"]
        column_renamemap = {"contains_care": "care",
                            "contains_fairness": "fairness",
                            "contains_loyalty": "loyalty",
                            "contains_authority": "authority",
                            "contains_sanctity": "sanctity"}
        palette_map = {"care": "#76D7C4",
                       "fairness": "#5DADE2",
                       "loyalty": "#F7DC6F",
                       "authority": "#BB8FCE",
                       "sanctity": "#D0D3D4"}
        pattern = {"care": "-",
                       "fairness": "/",
                        "loyalty": "+",
                        "authority": "x",
                        "sanctity": "."}
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
                       "contains_loyalty_vice"]
        column_renamemap = {"contains_care_virtue": "care+",
                            "contains_care_vice": "care-",
                            "contains_fairness_virtue": "fairness+",
                            "contains_fairness_vice": "fairness-",
                            "contains_loyalty_virtue": "loyalty+",
                            "contains_loyalty_vice": "loyalty-",
                            "contains_authority_virtue": "authority+",
                            "contains_authority_vice": "authority-",
                            "contains_sanctity_virtue": "sanctity+",
                            "contains_sanctity_vice": "sanctity-"}
        palette_map = {"care+": "#76D7C4",
                       "care-": "#F1C40F",
                       "fairness+": "#45B39D",
                       "fairness-": "#F5B041",
                       "loyalty+": "#1E8449",
                       "loyalty-": "#E74C3C",
                       "authority+": "#28B463",
                       "authority-": "#D35400",
                       "sanctity+": "#27AE60",
                       "sanctity-": "#EB984E"}
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
                                          (df["contains_authority_virtue"] != 1) & \
                                          (df["contains_sanctity_virtue"] != 1) & \
                                          (df["contains_sanctity_vice"] != 1) & \
                                          (df["contains_care_virtue"] != 1) & \
                                          (df["contains_care_vice"] != 1) & \
                                          (df["contains_fairness_virtue"] != 1) & \
                                          (df["contains_fairness_vice"] != 1)
        df = df[df["no moral foundation words"] != 1]

    statistic_df = df[column_list].sum().reset_index().rename(columns={"index": "Words Type",
                                                                                         0: "Number"})
    statistic_df["Words Type"] = statistic_df["Words Type"].map(column_renamemap,
                                                                na_action='ignore')
    statistic_df["Percentage"] = 100 * round(statistic_df["Number"] / len(df), 4)
    statistic_df = statistic_df.sort_values(by="Number", ascending=False).reset_index(drop=True)

    if format1 == "number":
        import seaborn as sns
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.barplot(data=statistic_df, y="Words Type", x="Number", palette=palette_map,
                    ax=ax)
        plt.tick_params(labelsize=15)
        ax.set_xlabel("number", fontsize=15)
        ax.set_ylabel("MF types", fontsize=15)
        ax.set_title("The number of {}s in {} containing different moral values".format(unit_name,title), fontsize=15)
        for i, j in enumerate(statistic_df["Number"].values.tolist()):
            plt.text(j, i, j, fontsize=12)
    else:
        import seaborn as sns
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.barplot(data=statistic_df, y="Words Type", x="Percentage", palette=palette_map,
                    ax=ax)
        plt.tick_params(labelsize=15)
        ax.set_xlabel("percentage", fontsize=15)
        ax.set_ylabel("MF types", fontsize=15)
        ax.set_title("The percentage of {}s in {} containing different moral values".format(unit_name,title), fontsize=15)
        for i, j in enumerate(statistic_df["Percentage"].values.tolist()):
            plt.text(j, i, "{}%".format(round(j, 2)), fontsize=12)
    return fig,statistic_df

def word_cloud_Arg_module():
    with st.chat_message("assistant"):
        st.write(
            "Awesome! Welcome to the *****Word Cloud feature*****. Please **choose the analytical angle for Arguments and Moral Values**.")
        col1, col_, col2 = st.columns([3,1,3])
        col1.write("#### Argument Property")
        col2.write("#### Moral Dimension")
        with col1:
            polaritysel = st.multiselect("Support/Attack", ["Support", "Attack"],placeholder="Choose one or more options")
            speakersel = st.multiselect("Arguments constructed by the same speaker/different speakers",
                                        ["Same speaker", "Different Speakers"],
                                        placeholder="Choose one or more options")
        with col2:
            MFsel = st.selectbox("Moral Foundation", ["Care", "Fairness", "Loyalty", "Authority", "Sanctity"])
            sentisel = st.selectbox("Moral Valence", ["Positive", "Negative"])

        if (len(polaritysel) != 0) and (len(speakersel) != 0):
            index1 = (data[polaritysel].sum(axis=1) > 0)
            index2 = (data[speakersel].sum(axis=1) > 0)
            df = data[index1&index2]
        else:
            df = pd.DataFrame()

    if len(df) != 0 and len(MFsel)!=0 and len(sentisel)!=0:
        with st.chat_message("assistant"):
            st.write(
                "Fantastic!🤗 Following your selections, the analysis outcomes will be displayed below.")
            Moral_Foundation_Word_Cloud(df,MFsel,sentisel)
    else:
        st.write(" ")



    return None

def moral_value_distribution_Arg_module():
    with st.chat_message("assistant"):
        st.write(
           "Wonderful! Welcome to the *****Moral Value Distribution***** feature. Kindly **select the analytical perspective for Arguments** and the **evaluation scale for Moral Values**.")
        col1, col_, col2 = st.columns([3,1,3])
        col1.write("#### Argument Property")
        col2.write("#### Moral Scale")
        with col1:
            polaritysel = st.multiselect("Support/Attack", ["Support", "Attack"],placeholder="Choose one or more options")
            speakersel = st.multiselect("Arguments constructed by the same speaker/different speakers",
                                        ["Same speaker", "Different Speakers"],
                                        placeholder="Choose one or more options")

        with col2:
            moral_scale = st.radio("", ['Moral vs No moral',
                                        '2 Moral Valences',
                                        '5 Moral Foundations',
                                        '10 Moral Values'], key="moral_scale_mv_arg")
            n_format = st.radio("numerical representations", ['number', 'percentage'], key="moral_scale_nformat_arg")

        if (len(polaritysel) != 0) and (len(speakersel) != 0):
            index1 = (data[polaritysel].sum(axis=1) > 0)
            index2 = (data[speakersel].sum(axis=1) > 0)
            df = data[index1&index2]
        else:
            df = pd.DataFrame()

    if len(df)!=0:
        with st.chat_message("assistant"):
            st.write("Fantastic!🤗 Following your selections, the analysis outcomes will be displayed below. Remember, you can always modify the presentation format of the statistics.")
            tab1,tab2 = st.tabs(["Bar Chart","Statistic Table"])
            with tab1:
                fig,table = Moral_Foundation_Word_In_Tweet(df, n_format, moral_scale,"Argument",title)
                st.pyplot(fig)
            with tab2:
                fig, table = Moral_Foundation_Word_In_Tweet(df, n_format, moral_scale,"Argument",title)
                st.dataframe(table)
                csv = convert_df(table)

                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='moral_value_distribution.csv',
                    mime='text/csv',
                )
    else:
        st.write(" ")
    return None

def moral_value_dynamics_Arg_module():
    with st.chat_message("assistant"):
        st.write(
           "Wonderful! Welcome to the *****Moral Value Dynamics***** feature. Kindly **select the analytical perspective for Arguments** and the **evaluation scale for Moral Values**.")
        col1, col_, col2 = st.columns([3,1,3])
        col1.write("#### Argument Property")
        col2.write("#### Moral Scale")
        with col1:
            polaritysel = st.multiselect("Support/Attack", ["Support", "Attack"],placeholder="Choose one or more options")
            speakersel = st.multiselect("Arguments constructed by the same speaker/different speakers",
                                        ["Same speaker", "Different Speakers"],
                                        placeholder="Choose one or more options")

        with col2:
            moral_scale = st.radio("", ['Moral vs No moral',
                                        '2 Moral Valences',
                                        '5 Moral Foundations',
                                        '10 Moral Values'], key="moral_scale_md_arg")
        if (len(polaritysel) != 0) and (len(speakersel) != 0):
            index1 = (data[polaritysel].sum(axis=1) > 0)
            index2 = (data[speakersel].sum(axis=1) > 0)
            df = data[index1&index2]
        else:
            df = pd.DataFrame()

    if len(df)!=0:
        with st.chat_message("assistant"):
            st.write("Fantastic!🤗 Following your selections, the analysis outcomes will be displayed below. Remember, you can always modify the presentation format of the statistics.")
            tab1,tab2 = st.tabs(["Heatmap","Statistic Table"])
            with tab1:
                fig,table = Moral_value_dynamics_in_argumentation(df, moral_scale,title)
                st.pyplot(fig)
            with tab2:
                fig, table = Moral_value_dynamics_in_argumentation(df,moral_scale,title)
                st.dataframe(table)
                csv = convert_df(table)

                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='moral_value_distribution.csv',
                    mime='text/csv',
                )
    else:
        st.write(" ")
    return None

def moral_speakers_distribution_ADU_module():
    df = data
    if len(df) != 0:
        with st.chat_message("assistant"):
            st.write(
                "Awesome! Welcome to the *****Interlocutors Distribution feature*****. Following your selections, the analysis outcomes will be displayed below.")
            num_format = st.radio("numerial representation",['number','percentage'])
            tab1,tab2,tab3 = st.tabs(["Stacked bar chart","Grouped bar chart","Layered bar chart"])
            with tab1:
                fig = User_Distribution_Stack(df,num_format)
                st.plotly_chart(fig)
            with tab2:
                fig = User_Distribution_Group(df, num_format)
                st.plotly_chart(fig)
            with tab3:
                fig = User_Distribution_Layered(df, num_format)
                st.plotly_chart(fig)
    else:
        st.write(" ")

def Moral_Concern_In_Social_Network(df, df_arg,moral_scale):
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
        df["no morals"] = (df["contains_loyalty_vice"] != 1) & \
                                          (df["contains_loyalty_virtue"] != 1) & \
                                          (df["contains_authority_vice"] != 1) & \
                                          (df["contains_authority_virtue"] != 1) & \
                                          (df["contains_sanctity_virtue"] != 1) & \
                                          (df["contains_sanctity_vice"] != 1) & \
                                          (df["contains_care_virtue"] != 1) & \
                                          (df["contains_care_vice"] != 1) & \
                                          (df["contains_fairness_virtue"] != 1) & \
                                          (df["contains_fairness_vice"] != 1)
        
        columns_list = ["morals", "no morals"]
        legend_title = ["morals", "no morals"]
        color_dict = {"no morals": "yellow", "morals": "black"}
    elif moral_scale == 'Moral Valences':
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
        df["no morals"] = (df["contains_loyalty_vice"] != 1) & \
                                          (df["contains_loyalty_virtue"] != 1) & \
                                          (df["contains_authority_vice"] != 1) & \
                                          (df["contains_authority_virtue"] != 1) & \
                                          (df["contains_sanctity_virtue"] != 1) & \
                                          (df["contains_sanctity_vice"] != 1) & \
                                          (df["contains_care_virtue"] != 1) & \
                                          (df["contains_care_vice"] != 1) & \
                                          (df["contains_fairness_virtue"] != 1) & \
                                          (df["contains_fairness_vice"] != 1)
        
        columns_list = ["positive moral valence", "negative moral valence","no morals"]
        legend_title = ["positive moral valence", "negative moral valence","no morals"]
        color_dict = {"positive moral valence": "green", "negative moral valence": "red","no morals":"black"}
    else:
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
        df["no morals"] = (df["contains_loyalty_vice"] != 1) & \
                                          (df["contains_loyalty_virtue"] != 1) & \
                                          (df["contains_authority_vice"] != 1) & \
                                          (df["contains_authority_virtue"] != 1) & \
                                          (df["contains_sanctity_virtue"] != 1) & \
                                          (df["contains_sanctity_vice"] != 1) & \
                                          (df["contains_care_virtue"] != 1) & \
                                          (df["contains_care_vice"] != 1) & \
                                          (df["contains_fairness_virtue"] != 1) & \
                                          (df["contains_fairness_vice"] != 1)
        
        columns_list = ["contains_care",
                       "contains_fairness",
                       "contains_loyalty",
                       "contains_authority",
                       "contains_sanctity","no morals"]
        legend_title = ["contains_care",
                       "contains_fairness",
                       "contains_loyalty",
                       "contains_authority",
                       "contains_sanctity","no morals"]
        color_dict = {"contains_care": "#76D7C4",
                       "contains_fairness": "#5DADE2",
                       "contains_loyalty": "#F7DC6F",
                       "contains_authority": "#BB8FCE",
                       "contains_sanctity": "#D0D3D4",
                       "no morals":"black"}

    df_user = df.groupby(["speaker"])[columns_list].sum().reset_index()

    connection = [(i, j) for (i, j) in list(zip(df_arg.speaker_premise.values, df_arg.speaker_conclusion.values))
                  if i != j]

    def color_palette(node,moral_scale):
        if moral_scale == 'Moral vs No moral':
            # No moral
            if df_user[df_user["speaker"] == node]["morals"].values > 0:
                color = "yellow"
            # Moral
            else:
                color = "black"
        elif moral_scale == 'Moral Valences':
            try:
                idx_max = df_user[df_user["speaker"] == node][columns_list].idxmax(axis=1).values[0]
            except IndexError:
                idx_max = None
            # Positive
            if idx_max == "negative moral valence":
                color = "red"
            # Negative
            elif idx_max == "positive moral valence":
                color = "green"
            else:
                color = "black"
        elif moral_scale == 'Moral Foundations':
            # st.write(len(df_user[legend_title]))
            # st.write(len(df_user[legend_title].dropna()))
            try:
                idx_max = df_user[df_user["speaker"] == node][columns_list].idxmax(axis=1).values[0]
            except IndexError:
                idx_max = None
            # Care
            if idx_max == "contains_care":
                color = "#76D7C4"
            # Fairness
            elif idx_max == "contains_fairness":
                color = "#5DADE2"
            # Authority
            elif idx_max == "contains_authority":
                color = "#F7DC6F"
            # Loyalty
            elif idx_max == "contains_loyalty":
                color = "#BB8FCE"
            # Sanctity
            elif idx_max == "contains_sanctity":
                color = "#D0D3D4"
            else:
                color = "black"
        else:
            color = "black"

        return color

    f = plt.figure(figsize=(10, 10))
    G = nx.DiGraph(directed=True)
    G.add_nodes_from(df_user.dropna().speaker.values.tolist())
    G.add_edges_from(connection)
    color_map = [color_palette(node,moral_scale) for node in G]
    nx.draw_networkx(G,with_labels=True, node_size=100, node_color=color_map)
    for v in set(legend_title):
        plt.scatter([], [], c=color_dict[v], label='{}'.format(v))
    handles, labels = plt.gca().get_legend_handles_labels()
    res = {labels[i]: handles[i] for i in range(len(labels))}
    plt.legend([res[label] for label in legend_title], legend_title, loc="upper center", bbox_to_anchor=(0.5, -0.1), title="User Types",
               title_fontsize="12", fontsize="12", ncol=4)
    return f


def moral_concern_score_ADU_module():
    df = data
    if len(df) != 0:
        with st.chat_message("assistant"):
            st.write(
                "Awesome! Welcome to the *****Moral Value Scores feature*****. Following your selections, the analysis outcomes will be displayed below.")
            personal_width = st.text_input('Costumised width', 700)
            personal_height = st.text_input('Costumised height', 190)
            fig = User_Moral_Concern_Score_Heatmap(df,int(personal_width),int(personal_height))
            st.plotly_chart(fig)
    else:
        st.write(" ")

def interlocutors_network_ADU_module():
        with st.chat_message("assistant"):
            st.write("Awesome! Welcome to the *****Argumentative network feature*****. Following your selections, the analysis outcomes will be displayed below.")
            col1,col2 = st.columns([2,2])
            with col1:
                moral_scale = st.radio("Moral scale", ['Moral vs No moral',
                                                       'Moral Valences',
                                                       'Moral Foundations'], key="moral_scale_mnetwork")
            with col2:
                polaritysel = st.multiselect("Support/Attack", ["Support", "Attack"],
                                         placeholder="Choose one or more options")

            if len(polaritysel) != 0:
                index2 = (data_[polaritysel].sum(axis=1) > 0)
                df_arg = data_[index2]
                df = data
                fig = Moral_Concern_In_Social_Network(df,df_arg, moral_scale)
                st.pyplot(fig)
            else:
                st.write(" ")

def Moral_Concern_User_Interaction_Analysis():
    df = data
    df_arg = data_
    if len(df) != 0:
        with st.chat_message("assistant"):
            st.write(
                "Awesome! Welcome to the *****Argumentative network feature*****. Following your selections, the analysis outcomes will be displayed below.")
            moral_foundation = st.selectbox("Choose the moral foundation",
                                            ("care", "fairness", "authority", "loyalty", "sanctity"))
            col1,col2 = st.columns([2,2])
            with col1:
                aspect = st.radio("Choose the analysis aspect",
                                  ("ADU frequency",))
            with col2:
                speakersel = st.multiselect("Arguments constructed by the same speaker/different speakers",
                                            ["Same speaker", "Different Speakers"],
                                            placeholder="Choose one or more options")
            if len(speakersel) != 0:
                index2 = (data_[speakersel].sum(axis=1) > 0)
                df_arg = data_[index2]
                fig = User_Interaction_Analysis(df, df_arg, aspect, moral_foundation)
                st.plotly_chart(fig)
            else:
                st.write("")
    else:
        st.write("")


st.sidebar.markdown("# Single corpus analysis ❄️")
unit = st.sidebar.radio(
    "Analysis units",
    ("ADU-based", "Relation-based", "Entity-based")
)

if unit != "Entity-based":
    module = st.sidebar.radio(
        "Analytics module",
        ("WordCloud", "Moral Value Distribution")
    )
else:
    module = st.sidebar.radio(
        "Analytics module",
        ("Moral Value Scores","Interlocutors Distribution","Argumentative Network","Argumentative Interaction")
    )


st.markdown("# Single corpus analysis ❄️")
with st.chat_message("assistant"):
    st.write("Hello 👋! Welcome to the Single Corpus Analysis page! Here, you can delve into the intriguing patterns of the defined corpora based on your chosen selection. To begin, please **pick the corpora** to define the dataset you'd like to examine!")
    col1, col2,= st.columns([2, 2])
    with col1:
        with st.expander("##### Dataset Operation", expanded=True):
            df_adu, df_arg = customised_dataloader()
            adu_csv = convert_df(df_adu)
            arg_csv = convert_df(df_arg)
            if unit == "Relation-based":
                uploaded_files = st.file_uploader("Upload CSV Files (Arguments)", accept_multiple_files=True)
                st.info("💡 Please download annotated arguments and upload above to see how it works")
                st.download_button(
                    label="Download Annotated Arguments",
                    data=arg_csv,
                    file_name='MM_Money_Argument.csv'.format(),
                    mime='text/csv',
                )
            elif unit == "ADU-based":
                uploaded_files = st.file_uploader("Upload CSV Files (ADUs)", accept_multiple_files=True)
                st.info("💡 Please download annotated ADUs and upload above to see how it works")
                st.download_button(
                    label="Download Annotated ADUs",
                    data=adu_csv,
                    file_name='MM_Money_ADU.csv'.format(),
                    mime='text/csv',
                )
            else:
                uploaded_files = st.file_uploader("Upload CSV Files (Arguments and ADUs)", accept_multiple_files=True)
                st.info("💡 Please download annotated arguments and ADUs and upload above to see how it works")
                st.download_button(
                    label="Download Annotated Arguments",
                    data=arg_csv,
                    file_name='MM_Money_Argument.csv'.format(),
                    mime='text/csv',
                )
                st.download_button(
                    label="Download Annotated ADUs",
                    data=adu_csv,
                    file_name='MM_Money_ADU.csv'.format(),
                    mime='text/csv',
                )

            df_cus = pd.DataFrame()
            for uploaded_file in uploaded_files:
                df_cus_temp = pd.read_csv(uploaded_file, index_col=0)
                df_cus_temp["dataset"] = re.sub('.csv', "", uploaded_file.name)
                df_cus = pd.concat([df_cus, df_cus_temp])
    with col2:
        with st.expander("##### Data List", expanded=True):
            if len(df_cus) != 0:
                selection = select_corpora_customisation(customised_dataset=df_cus.dataset.unique())
            else:
                selection = select_corpora()
            title = st.text_input('You can define the name for the selected dataset', 'Customised Dataset')
    st.divider()


adu_filename_map = {
                    "US2016r1D": "data/US2016rD1_ADU_Moral.xlsx",
                    "US2016r1G": "data/US2016rG1_ADU_Moral.xlsx",
                    "US2016r1R": "data/US2016rR1_ADU_Moral.xlsx",
                    "US2016tvD": "data/US2016D1tv_ADU_Moral.xlsx",
                    "US2016tvG": "data/US2016G1tv_ADU_Moral.xlsx",
                    "US2016tvR": "data/US2016R1tv_ADU_Moral.xlsx",
                    "British Empire": "data/BritishEmpire_ADU_Moral.xlsx",
                    "DDay": "data/DDay_ADU_Moral.xlsx",
                    "Hypocrisy": "data/Hypocrisy_ADU_Moral.xlsx",
                    "Money": "data/Money_ADU_Moral.xlsx",
                    "Welfare": "data/Welfare_ADU_Moral.xlsx",
                    }

arg_filename_map = {
                    "US2016r1D": "data/US2016rD1_Arg_Moral.xlsx",
                    "US2016r1G": "data/US2016rG1_Arg_Moral.xlsx",
                    "US2016r1R": "data/US2016rR1_Arg_Moral.xlsx",
                    "US2016tvD": "data/US2016D1tv_Arg_Moral.xlsx",
                    "US2016tvG": "data/US2016G1tv_Arg_Moral.xlsx",
                    "US2016tvR": "data/US2016R1tv_Arg_Moral.xlsx",
                    "British Empire": "data/BritishEmpire_Arg_Moral.xlsx",
                    "DDay": "data/DDay_Arg_Moral.xlsx",
                    "Hypocrisy": "data/Hypocrisy_Arg_Moral.xlsx",
                    "Money": "data/Money_Arg_Moral.xlsx",
                    "Welfare": "data/Welfare_Arg_Moral.xlsx"}


if unit == "ADU-based":
    if module == "WordCloud":
        if len(selection["checked"]) > 0:
            ##########################
            selection_files = [adu_filename_map.get(corpora_name, corpora_name) for corpora_name in
                               selection["checked"]]
            data = ADU_dataloader(selection_files)
            if len(data)!=0:
                word_cloud_ADU_module()
            ##########################
    elif module == "Moral Value Distribution":
        if len(selection["checked"]) > 0:
            ##########################
            selection_files = [adu_filename_map.get(corpora_name, corpora_name) for corpora_name in
                               selection["checked"]]
            data = ADU_dataloader(selection_files)
            if len(data) != 0:
                moral_value_distribution_ADU_module()
            ###########################
elif unit == "Relation-based":
    if module == "WordCloud":
        if len(selection["checked"]) > 0:
            ##########################
            selection_files = [arg_filename_map.get(corpora_name, corpora_name) for corpora_name in
                               selection["checked"]]
            data = Arg_dataloader(selection_files)
            if len(data) != 0:
                word_cloud_Arg_module()
    elif module == "Moral Value Distribution":
        if len(selection["checked"]) > 0:
            ##########################
            selection_files = [arg_filename_map.get(corpora_name, corpora_name) for corpora_name in
                               selection["checked"]]
            data = Arg_dataloader(selection_files)
            if len(data) != 0:
                moral_value_distribution_Arg_module()
    elif module == "Moral Value Dynamics":
        if len(selection["checked"]) > 0:
            ##########################
            selection_files = [arg_filename_map.get(corpora_name, corpora_name) for corpora_name in
                               selection["checked"]]
            data = Arg_dataloader(selection_files)
            if len(data) != 0:
               moral_value_dynamics_Arg_module()
else:
    if module == "Moral Value Scores":
        if len(selection["checked"]) > 0:
            ##########################
            selection_files = [adu_filename_map.get(corpora_name, corpora_name) for corpora_name in
                               selection["checked"]]
            data = ADU_dataloader(selection_files)
            if len(data) != 0:
                moral_concern_score_ADU_module()
    elif module == "Interlocutors Distribution":
        if len(selection["checked"]) > 0:
            ##########################
            selection_files = [adu_filename_map.get(corpora_name, corpora_name) for corpora_name in
                               selection["checked"]]
            data = ADU_dataloader(selection_files)
            if len(data) != 0:
                moral_speakers_distribution_ADU_module()
    elif module == "Argumentative Network":
        if len(selection["checked"]) > 0:
            ##########################
            selection_files_arg = [arg_filename_map.get(corpora_name, corpora_name) for corpora_name in
                               selection["checked"]]
            selection_files = [adu_filename_map.get(corpora_name, corpora_name) for corpora_name in
                                   selection["checked"]]
            data = ADU_dataloader(selection_files)
            data_ = Arg_dataloader(selection_files_arg)
            if len(data) != 0 and len(data_)!=0:
                interlocutors_network_ADU_module()
    else:
        if len(selection["checked"]) > 0:
            ##########################
            selection_files_arg = [arg_filename_map.get(corpora_name, corpora_name) for corpora_name in
                               selection["checked"]]
            selection_files = [adu_filename_map.get(corpora_name, corpora_name) for corpora_name in
                                   selection["checked"]]
            data = ADU_dataloader(selection_files)
            data_ = Arg_dataloader(selection_files_arg)
            if len(data) != 0 and len(data_) != 0:
                Moral_Concern_User_Interaction_Analysis()











