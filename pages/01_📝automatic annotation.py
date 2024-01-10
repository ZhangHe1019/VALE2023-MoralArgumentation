# imports
import pandas as pd
pd.set_option("max_colwidth", 400)
import numpy as np
import seaborn as sns

# Contents of ~/my_app/pages/page_3.py
import copy
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import string
import re
import requests
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import spacy
nlp = spacy.load('en_core_web_sm')
nltk.download('omw-1.4')
import numpy as np

sns.set_theme(style="whitegrid")
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
import glob
import json


@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

def sentence_token_nltk(text):
    sent_tokenize_list = sent_tokenize(text)
    return sent_tokenize_list


class preprocessing_step():
    def __init__(self, text):
        self.raw_text = text
        self.lowercase = text.casefold()
        self.remove_url = re.sub(r'((https|http|ftp|rtsp|mms)?:\/\/)[^\s]+', "", self.lowercase)
        self.clean_text = re.sub(r"[!@#$%,^&*():']+", " ", self.remove_url)

    def normalize_text(self):
        clean_text = self.similar_item_merge(self.clean_text)
        nlp_sentence = nlp(clean_text)
        lemmatized_sentence = " ".join([token.lemma_ for token in nlp_sentence if
                                        token.lemma_.strip(string.punctuation) and not token.lemma_.isdigit()])
        self.lemmas = nltk.word_tokenize(lemmatized_sentence)
        return self.lemmas

    def normalize_text_alignment(self):
        tokens = nltk.word_tokenize(self.clean_text)
        clean_text = " ".join(tokens)
        nlp_sentence = nlp(clean_text)
        token_dict = {token.lemma_: token.text for token in nlp_sentence}
        return token_dict

    def similar_item_merge(self, text):
        lemmas = list()
        tokens = nltk.word_tokenize(text)
        for token in tokens:
            pos_tag = nltk.pos_tag([token])
            lemma = Lemmatizer().get_lemmas([token], pos_tag)
            lemmas.append(lemma[0])
        return " ".join(lemmas)


class Lemmatizer():
    def __init__(self):
        self.lemmas = list()
        self.wnl = WordNetLemmatizer()

    def get_lemmas(self, token_list, pos_list):
        for i in range(len(token_list)):
            self.lemmas.append(self.wnl.lemmatize(token_list[i], self.get_wordnet_pos(pos_list[i][1])))  # 词形还原
        return self.lemmas

    def get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN


# Get n-grams
def get_n_gram(text, n_gram_min, n_gram_max):
    vectorizer2 = CountVectorizer(ngram_range=(n_gram_min, n_gram_max),
                                  tokenizer=lambda x: preprocessing_step(x).normalize_text())
    vectorizer2.fit(text)
    Grams = list(vectorizer2.vocabulary_.keys())
    return Grams


def assign_POS_words(dataframe, text_column):
    '''Parameters:
    dataframe: dataframe with your data,
    text_column: column of a dataframe where text is located
    '''
    nlp = spacy.load('en_core_web_sm')

    df = dataframe.copy()
    pos_all = []
    noun_list = []

    for doc in nlp.pipe(df[text_column].apply(lambda x: x.strip())):
        pos_all.append(" ".join([tok.pos_ for tok in doc]))
        noun_list.append(" ".join([tok.lemma_ for tok in doc if tok.pos_ == "NOUN"]))

    df["POS"] = pos_all
    df["Noun_list"] = noun_list
    return df


def MFD():
    dict1 = pd.read_excel("./data/MFD.xlsx")
    care_virtue = dict1[dict1["Category"] == "care.virtue"]["Words"].values
    care_vice = dict1[dict1["Category"] == "care.vice"]["Words"].values
    fairness_virtue = dict1[dict1["Category"] == "fairness.virtue"]["Words"].values
    fairness_vice = dict1[dict1["Category"] == "fairness.vice"]["Words"].values
    loyalty_virtue = dict1[dict1["Category"] == "loyalty.virtue"]["Words"].values
    loyalty_vice = dict1[dict1["Category"] == "loyalty.vice"]["Words"].values
    authority_virtue = dict1[dict1["Category"] == "authority.virtue"]["Words"].values
    authority_vice = dict1[dict1["Category"] == "authority.vice"]["Words"].values
    sanctity_virtue = dict1[dict1["Category"] == "sanctity.virtue"]["Words"].values
    sanctity_vice = dict1[dict1["Category"] == "sanctity.vice"]["Words"].values
    return care_virtue, care_vice, fairness_virtue, fairness_vice, loyalty_virtue, loyalty_vice,authority_virtue, authority_vice, sanctity_virtue,sanctity_vice


class Keywords_matching():
    def __init__(self, copus_dict, lexicon):
        self.keywords = list()
        self.num = 0
        self.copus = copus_dict
        self.lexicon = lexicon

    def return_string(self):
        if self.copus:
            for token in self.copus:
                if token in self.lexicon:
                    self.keywords.append(token)
                    self.num = self.num + 1
            keywords = ",".join(self.keywords)
        else:
            keywords = ""
        return keywords

    def return_list(self):
        if self.copus:
            for token in self.copus:
                if token in self.lexicon:
                    self.keywords.append(token)
                    self.num = self.num + 1
            keywords = self.keywords
        else:
            keywords = []
        return keywords

    def return_num(self):
        self.return_string()
        return self.num


# Function to remove mentions from a text
def remove_mentions(text):
    return re.sub(r'@[\w_]+', '', text)


def Run_MV_Detection_ADU(Data,care_virtue, care_vice, fairness_virtue, fairness_vice, loyalty_virtue, loyalty_vice,authority_virtue, authority_vice, sanctity_virtue,sanctity_vice):
    with st.spinner('Removing the @mention'):
        Data['Text'] = Data["locution"].apply(remove_mentions)
    st.toast('Successfully Remove the @mention', icon='🎉')
    with st.spinner('Tokenization and Lemmatization'):
        Data["Words"] = Data.Text.apply(lambda x: preprocessing_step(x).normalize_text())
    st.toast('After Tokenization and Lemmatization', icon='🎉')
    with st.spinner('Calculating the Number of Words'):
        Data["Num_words"] = Data.Words.apply(lambda x: len(x))
    with st.spinner('Detecting Moral Foundations'):
        Data["care.virtue"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                     care_virtue).return_list() if x.Num_words > 0 else "",
                                         axis=1)
        Data["care.vice"] = Data.apply(
            lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1), care_vice).return_list() if x.Num_words > 0 else "",
            axis=1)

        Data["fairness.virtue"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                         fairness_virtue).return_list() if x.Num_words > 0 else "",
                                             axis=1)

        Data["fairness.vice"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                       fairness_vice).return_list() if x.Num_words > 0 else "",
                                           axis=1)

        Data["loyalty.virtue"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                        loyalty_virtue).return_list() if x.Num_words > 0 else "",
                                            axis=1)

        Data["loyalty.vice"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                      loyalty_vice).return_list() if x.Num_words > 0 else "",
                                          axis=1)

        Data["authority.virtue"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                          authority_virtue).return_list() if x.Num_words > 0 else "",
                                              axis=1)

        Data["authority.vice"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                        authority_vice).return_list() if x.Num_words > 0 else "",
                                            axis=1)

        Data["sanctity.virtue"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                         sanctity_virtue).return_list() if x.Num_words > 0 else "",
                                             axis=1)

        Data["sanctity.vice"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                       sanctity_vice).return_list() if x.Num_words > 0 else "",
                                           axis=1)

        Data["num care.virtue"] = Data.apply(
            lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1), care_virtue).return_num() if x.Num_words > 0 else 0,
            axis=1)

        Data["num care.vice"] = Data.apply(
            lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1), care_vice).return_num() if x.Num_words > 0 else 0,
            axis=1)

        Data["num fairness.virtue"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                             fairness_virtue).return_num() if x.Num_words > 0 else 0,
                                                 axis=1)

        Data["num fairness.vice"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                           fairness_vice).return_num() if x.Num_words > 0 else 0,
                                               axis=1)

        Data["num loyalty.virtue"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                            loyalty_virtue).return_num() if x.Num_words > 0 else 0,
                                                axis=1)

        Data["num loyalty.vice"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                          loyalty_vice).return_num() if x.Num_words > 0 else 0,
                                              axis=1)

        Data["num authority.virtue"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                              authority_virtue).return_num() if x.Num_words > 0 else 0,
                                                  axis=1)

        Data["num authority.vice"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                            authority_vice).return_num() if x.Num_words > 0 else 0,
                                                axis=1)

        Data["num sanctity.virtue"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                             sanctity_virtue).return_num() if x.Num_words > 0 else 0,
                                                 axis=1)

        Data["num sanctity.vice"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                           sanctity_vice).return_num() if x.Num_words > 0 else 0,
                                               axis=1)

        Data["contains_care_virtue"] = (Data["num care.virtue"] > 0).astype(int)
        Data["contains_care_vice"] = (Data["num care.vice"] > 0).astype(int)
        Data["contains_fairness_virtue"] = (Data["num fairness.virtue"] > 0).astype(int)
        Data["contains_fairness_vice"] = (Data["num fairness.vice"] > 0).astype(int)
        Data["contains_sanctity_virtue"] = (Data["num sanctity.virtue"] > 0).astype(int)
        Data["contains_sanctity_vice"] = (Data["num sanctity.vice"] > 0).astype(int)
        Data["contains_authority_virtue"] = (Data["num authority.virtue"] > 0).astype(int)
        Data["contains_authority_vice"] = (Data["num authority.vice"] > 0).astype(int)
        Data["contains_loyalty_virtue"] = (Data["num loyalty.virtue"] > 0).astype(int)
        Data["contains_loyalty_vice"] = (Data["num loyalty.vice"] > 0).astype(int)

        Data["no moral foundation words"] = (Data["loyalty.vice"].isnull().astype(int)) & \
                                          (Data["loyalty.virtue"].isnull().astype(int)) & \
                                          (Data["authority.vice"].isnull().astype(int)) & \
                                          (Data["authority.virtue"].isnull().astype(int)) & \
                                          (Data["sanctity.virtue"].isnull().astype(int)) & \
                                          (Data["sanctity.vice"].isnull().astype(int)) & \
                                          (Data["care.virtue"].isnull().astype(int)) & \
                                          (Data["care.vice"].isnull().astype(int)) & \
                                          (Data["fairness.virtue"].isnull().astype(int)) & \
                                          (Data["fairness.vice"].isnull().astype(int))
    st.toast('Successfully', icon='🎉')
    return Data

def Run_MV_Detection_Arg(Data,care_virtue, care_vice, fairness_virtue, fairness_vice, loyalty_virtue, loyalty_vice,authority_virtue, authority_vice, sanctity_virtue,sanctity_vice):
    with st.spinner('Removing the @mention'):
        Data['Text'] = Data["premise"].apply(remove_mentions)
        Data['text'] = Data["conclusion"].apply(remove_mentions)
        Data['Text'] = Data['Text'] + "," + Data["text"]

    st.toast('Successfully Remove the @mention', icon='🎉')
    with st.spinner('Tokenization and Lemmatization'):
        Data["Words"] = Data.Text.apply(lambda x: preprocessing_step(x).normalize_text())
    st.toast('After Tokenization and Lemmatization', icon='🎉')
    with st.spinner('Calculating the Number of Words'):
        Data["Num_words"] = Data.Words.apply(lambda x: len(x))
    with st.spinner('Detecting Moral Foundations'):
        Data["care.virtue"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                     care_virtue).return_list() if x.Num_words > 0 else "",
                                         axis=1)
        Data["care.vice"] = Data.apply(
            lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1), care_vice).return_list() if x.Num_words > 0 else "",
            axis=1)

        Data["fairness.virtue"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                         fairness_virtue).return_list() if x.Num_words > 0 else "",
                                             axis=1)

        Data["fairness.vice"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                       fairness_vice).return_list() if x.Num_words > 0 else "",
                                           axis=1)

        Data["loyalty.virtue"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                        loyalty_virtue).return_list() if x.Num_words > 0 else "",
                                            axis=1)

        Data["loyalty.vice"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                      loyalty_vice).return_list() if x.Num_words > 0 else "",
                                          axis=1)

        Data["authority.virtue"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                          authority_virtue).return_list() if x.Num_words > 0 else "",
                                              axis=1)

        Data["authority.vice"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                        authority_vice).return_list() if x.Num_words > 0 else "",
                                            axis=1)

        Data["sanctity.virtue"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                         sanctity_virtue).return_list() if x.Num_words > 0 else "",
                                             axis=1)

        Data["sanctity.vice"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                       sanctity_vice).return_list() if x.Num_words > 0 else "",
                                           axis=1)

        Data["num care.virtue"] = Data.apply(
            lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1), care_virtue).return_num() if x.Num_words > 0 else 0,
            axis=1)

        Data["num care.vice"] = Data.apply(
            lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1), care_vice).return_num() if x.Num_words > 0 else 0,
            axis=1)

        Data["num fairness.virtue"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                             fairness_virtue).return_num() if x.Num_words > 0 else 0,
                                                 axis=1)

        Data["num fairness.vice"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                           fairness_vice).return_num() if x.Num_words > 0 else 0,
                                               axis=1)

        Data["num loyalty.virtue"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                            loyalty_virtue).return_num() if x.Num_words > 0 else 0,
                                                axis=1)

        Data["num loyalty.vice"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                          loyalty_vice).return_num() if x.Num_words > 0 else 0,
                                              axis=1)

        Data["num authority.virtue"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                              authority_virtue).return_num() if x.Num_words > 0 else 0,
                                                  axis=1)

        Data["num authority.vice"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                            authority_vice).return_num() if x.Num_words > 0 else 0,
                                                axis=1)

        Data["num sanctity.virtue"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                             sanctity_virtue).return_num() if x.Num_words > 0 else 0,
                                                 axis=1)

        Data["num sanctity.vice"] = Data.apply(lambda x: Keywords_matching(get_n_gram([x.Text], 0, 1),
                                                                           sanctity_vice).return_num() if x.Num_words > 0 else 0,
                                               axis=1)

        Data["contains_care_virtue"] = (Data["num care.virtue"] > 0).astype(int)
        Data["contains_care_vice"] = (Data["num care.vice"] > 0).astype(int)
        Data["contains_fairness_virtue"] = (Data["num fairness.virtue"] > 0).astype(int)
        Data["contains_fairness_vice"] = (Data["num fairness.vice"] > 0).astype(int)
        Data["contains_sanctity_virtue"] = (Data["num sanctity.virtue"] > 0).astype(int)
        Data["contains_sanctity_vice"] = (Data["num sanctity.vice"] > 0).astype(int)
        Data["contains_authority_virtue"] = (Data["num authority.virtue"] > 0).astype(int)
        Data["contains_authority_vice"] = (Data["num authority.vice"] > 0).astype(int)
        Data["contains_loyalty_virtue"] = (Data["num loyalty.virtue"] > 0).astype(int)
        Data["contains_loyalty_vice"] = (Data["num loyalty.vice"] > 0).astype(int)

        Data["no moral foundation words"] = (Data["loyalty.vice"].isnull().astype(int)) & \
                                          (Data["loyalty.virtue"].isnull().astype(int)) & \
                                          (Data["authority.vice"].isnull().astype(int)) & \
                                          (Data["authority.virtue"].isnull().astype(int)) & \
                                          (Data["sanctity.virtue"].isnull().astype(int)) & \
                                          (Data["sanctity.vice"].isnull().astype(int)) & \
                                          (Data["care.virtue"].isnull().astype(int)) & \
                                          (Data["care.vice"].isnull().astype(int)) & \
                                          (Data["fairness.virtue"].isnull().astype(int)) & \
                                          (Data["fairness.vice"].isnull().astype(int))
    st.toast('Successfully', icon='🎉')
    return Data


def add_spacelines(number_sp=2):
    for xx in range(number_sp):
        st.write("\n")


import os.path
import requests
from datetime import datetime
from pathlib import Path
import re
# import networkx as nx
import plotly.express as px


# https://github.com/roryduthie/AIF_Converter
def get_graph_url(node_path):
    try:
        jsn_string = requests.get(node_path).text
        strng_ind = jsn_string.index('{')
        n_string = jsn_string[strng_ind:]
        dta = json.loads(n_string)
    except:
        st.error(f'File was not found: {node_path}')
    return dta


@st.cache_data()
def RetrieveNodes(node_list, from_dict=False, type_aif='old'):
    df_all_loc = pd.DataFrame(
        columns=['locution', 'connection', 'illocution', 'id_locution', 'id_connection', 'id_illocution', 'nodeset_id'])
    df_all = pd.DataFrame(
        columns=['premise', 'connection', 'conclusion', 'id_premise', 'id_connection', 'id_conclusion', 'nodeset_id'])

    if from_dict:
        for map in node_list.keys():
            try:
                match_nodeset = map
                map1 = node_list[map]
                if 'AIF' in map1.keys():
                    type_aif = 'new'
                else:
                    type_aif = 'old'

                if type_aif == 'new':
                    df_nodes = pd.DataFrame(map1['AIF']['nodes'])
                    df_edge = pd.DataFrame(map1['AIF']['edges'])
                else:
                    df_nodes = pd.DataFrame(map1['nodes'])
                    df_edge = pd.DataFrame(map1['edges'])

                tto1l = []
                tfrom1l = []
                tto2l = []
                tfrom2l = []
                connect_idsl = []
                loc_idsl = []
                illoc_idsl = []
                nodeset_idsl = []

                tto1i = []
                tfrom1i = []
                tto2i = []
                tfrom2i = []
                connect_idsi = []
                loc_idsi = []
                illoc_idsi = []
                nodeset_idsi = []
                rels = ['MA', 'CA', 'RA']

                for id1 in df_edge.index:
                    for id2 in df_edge.index:

                        id_from1 = df_edge.loc[id1, 'fromID']
                        id_to1 = df_edge.loc[id1, 'toID']

                        id_from2 = df_edge.loc[id2, 'fromID']
                        id_to2 = df_edge.loc[id2, 'toID']

                        if id_to1 == id_from2:
                            # locutions
                            if (df_nodes[(df_nodes.nodeID == id_from2)]['type'].iloc[0] == 'YA') and (
                                    df_nodes[(df_nodes.nodeID == id_to2)]['type'].iloc[0] == 'I') and (
                                    df_nodes[(df_nodes.nodeID == id_from1)]['type'].iloc[0] == 'L'):
                                d1 = df_nodes[(df_nodes.nodeID == id_from1)]
                                d2 = df_nodes[(df_nodes.nodeID == id_to1)]

                                d11 = df_nodes[(df_nodes.nodeID == id_from2)]
                                d22 = df_nodes[(df_nodes.nodeID == id_to2)]

                                tto1l.append(d2['text'].iloc[0])
                                tfrom1l.append(d1['text'].iloc[0])

                                tto2l.append(d22['text'].iloc[0])
                                tfrom2l.append(d11['text'].iloc[0])

                                connect_idsl.append(id_to1)
                                loc_idsl.append(d1['nodeID'].iloc[0])
                                illoc_idsl.append(d22['nodeID'].iloc[0])

                                nodeset_idsl.append(match_nodeset)

                            # args
                            if (df_nodes[(df_nodes.nodeID == id_to1)]['type'].iloc[0] in rels) and (
                                    df_nodes[(df_nodes.nodeID == id_from1)]['type'].iloc[0] != 'YA') and (
                                    df_nodes[(df_nodes.nodeID == id_to2)]['type'].iloc[0] == 'I'):
                                d1 = df_nodes[(df_nodes.nodeID == id_from1)]
                                d2 = df_nodes[(df_nodes.nodeID == id_to1)]

                                d11 = df_nodes[(df_nodes.nodeID == id_from2)]
                                d22 = df_nodes[(df_nodes.nodeID == id_to2)]

                                tto1i.append(d2['text'].iloc[0])
                                tfrom1i.append(d1['text'].iloc[0])

                                tto2i.append(d22['text'].iloc[0])
                                tfrom2i.append(d11['text'].iloc[0])

                                connect_idsi.append(id_to1)
                                loc_idsi.append(d1['nodeID'].iloc[0])
                                illoc_idsi.append(d22['nodeID'].iloc[0])
                                nodeset_idsi.append(match_nodeset)

                df1 = pd.DataFrame({
                    'locution': tfrom1l,
                    'connection': tto1l,
                    'illocution': tto2l,
                    'id_locution': loc_idsl,
                    'id_connection': connect_idsl,
                    'id_illocution': illoc_idsl,
                    'nodeset_id': nodeset_idsl,
                })
                df_all_loc = pd.concat([df_all_loc, df1], axis=0, ignore_index=True)

                df2 = pd.DataFrame({
                    'premise': tfrom1i,
                    'connection': tto1i,
                    'conclusion': tto2i,
                    'id_premise': loc_idsi,
                    'id_connection': connect_idsi,
                    'id_conclusion': illoc_idsi,
                    'nodeset_id': nodeset_idsi,
                })
                df_all = pd.concat([df_all, df2], axis=0, ignore_index=True)
            except:
                continue
    else:
        for map in node_list[:]:
            try:
                with open(map, 'r') as f:
                    map1 = json.load(f)
                match_nodeset = re.split('nodeset', str(map))
                match_nodeset = match_nodeset[-1][:4]
                if 'AIF' in map1.keys():
                    type_aif = 'new'
                else:
                    type_aif = 'old'

                if type_aif == 'new':
                    df_nodes = pd.DataFrame(map1['AIF']['nodes'])
                    df_edge = pd.DataFrame(map1['AIF']['edges'])
                else:
                    df_nodes = pd.DataFrame(map1['nodes'])
                    df_edge = pd.DataFrame(map1['edges'])

                tto1l = []
                tfrom1l = []
                tto2l = []
                tfrom2l = []
                connect_idsl = []
                loc_idsl = []
                illoc_idsl = []
                nodeset_idsl = []

                tto1i = []
                tfrom1i = []
                tto2i = []
                tfrom2i = []
                connect_idsi = []
                loc_idsi = []
                illoc_idsi = []
                nodeset_idsi = []
                rels = ['MA', 'CA', 'RA', 'PA']

                for id1 in df_edge.index:
                    for id2 in df_edge.index:

                        id_from1 = df_edge.loc[id1, 'fromID']
                        id_to1 = df_edge.loc[id1, 'toID']

                        id_from2 = df_edge.loc[id2, 'fromID']
                        id_to2 = df_edge.loc[id2, 'toID']

                        if id_to1 == id_from2:
                            # locutions
                            if (df_nodes[(df_nodes.nodeID == id_from2)]['type'].iloc[0] == 'YA') and (
                                    df_nodes[(df_nodes.nodeID == id_to2)]['type'].iloc[0] == 'I') and (
                                    df_nodes[(df_nodes.nodeID == id_from1)]['type'].iloc[0] == 'L'):
                                d1 = df_nodes[(df_nodes.nodeID == id_from1)]
                                d2 = df_nodes[(df_nodes.nodeID == id_to1)]

                                d11 = df_nodes[(df_nodes.nodeID == id_from2)]
                                d22 = df_nodes[(df_nodes.nodeID == id_to2)]

                                tto1l.append(d2['text'].iloc[0])
                                tfrom1l.append(d1['text'].iloc[0])

                                tto2l.append(d22['text'].iloc[0])
                                tfrom2l.append(d11['text'].iloc[0])

                                connect_idsl.append(id_to1)
                                loc_idsl.append(d1['nodeID'].iloc[0])
                                illoc_idsl.append(d22['nodeID'].iloc[0])

                                nodeset_idsl.append(match_nodeset)

                            # args
                            if (df_nodes[(df_nodes.nodeID == id_to1)]['type'].iloc[0] in rels) and (
                                    df_nodes[(df_nodes.nodeID == id_from1)]['type'].iloc[0] != 'YA') and (
                                    df_nodes[(df_nodes.nodeID == id_to2)]['type'].iloc[0] == 'I'):
                                d1 = df_nodes[(df_nodes.nodeID == id_from1)]
                                d2 = df_nodes[(df_nodes.nodeID == id_to1)]

                                d11 = df_nodes[(df_nodes.nodeID == id_from2)]
                                d22 = df_nodes[(df_nodes.nodeID == id_to2)]

                                tto1i.append(d2['text'].iloc[0])
                                tfrom1i.append(d1['text'].iloc[0])

                                tto2i.append(d22['text'].iloc[0])
                                tfrom2i.append(d11['text'].iloc[0])

                                connect_idsi.append(id_to1)
                                loc_idsi.append(d1['nodeID'].iloc[0])
                                illoc_idsi.append(d22['nodeID'].iloc[0])
                                nodeset_idsi.append(match_nodeset)

                df1 = pd.DataFrame({
                    'locution': tfrom1l,
                    'connection': tto1l,
                    'illocution': tto2l,
                    'id_locution': loc_idsl,
                    'id_connection': connect_idsl,
                    'id_illocution': illoc_idsl,
                    'nodeset_id': nodeset_idsl,
                })

                df_all_loc = pd.concat([df_all_loc, df1], axis=0, ignore_index=True)

                df2 = pd.DataFrame({
                    'premise': tfrom1i,
                    'connection': tto1i,
                    'conclusion': tto2i,
                    'id_premise': loc_idsi,
                    'id_connection': connect_idsi,
                    'id_conclusion': illoc_idsi,
                    'nodeset_id': nodeset_idsi,
                })
                df_all = pd.concat([df_all, df2], axis=0, ignore_index=True)
            except:
                continue

    return df_all_loc, df_all


@st.cache_data()
def RetrieveNodesOnline(map1, nodeset_id_str, type_aif='old'):
    if 'AIF' in map1.keys():
        type_aif = 'new'
    else:
        type_aif = 'old'

    try:
        if type_aif == 'new':
            df_nodes = pd.DataFrame(map1['AIF']['nodes'])
            df_edge = pd.DataFrame(map1['AIF']['edges'])
        else:
            df_nodes = pd.DataFrame(map1['nodes'])
            df_edge = pd.DataFrame(map1['edges'])

        tto1l = []
        tfrom1l = []
        tto2l = []
        tfrom2l = []
        connect_idsl = []
        loc_idsl = []
        illoc_idsl = []
        nodeset_idsl = []

        match_nodeset = nodeset_id_str

        tto1i = []
        tfrom1i = []
        tto2i = []
        tfrom2i = []
        connect_idsi = []
        loc_idsi = []
        illoc_idsi = []
        nodeset_idsi = []
        rels = ['MA', 'CA', 'RA']

        for id1 in df_edge.index:
            for id2 in df_edge.index:

                id_from1 = df_edge.loc[id1, 'fromID']
                id_to1 = df_edge.loc[id1, 'toID']

                id_from2 = df_edge.loc[id2, 'fromID']
                id_to2 = df_edge.loc[id2, 'toID']

                if id_to1 == id_from2:
                    # locutions
                    if (df_nodes[(df_nodes.nodeID == id_from2)]['type'].iloc[0] == 'YA') and (
                            df_nodes[(df_nodes.nodeID == id_to2)]['type'].iloc[0] == 'I') and (
                            df_nodes[(df_nodes.nodeID == id_from1)]['type'].iloc[0] == 'L'):
                        d1 = df_nodes[(df_nodes.nodeID == id_from1)]
                        d2 = df_nodes[(df_nodes.nodeID == id_to1)]

                        d11 = df_nodes[(df_nodes.nodeID == id_from2)]
                        d22 = df_nodes[(df_nodes.nodeID == id_to2)]

                        tto1l.append(d2['text'].iloc[0])
                        tfrom1l.append(d1['text'].iloc[0])

                        tto2l.append(d22['text'].iloc[0])
                        tfrom2l.append(d11['text'].iloc[0])

                        connect_idsl.append(id_to1)
                        loc_idsl.append(d1['nodeID'].iloc[0])
                        illoc_idsl.append(d22['nodeID'].iloc[0])

                        nodeset_idsl.append(match_nodeset)

                    # args
                    if (df_nodes[(df_nodes.nodeID == id_to1)]['type'].iloc[0] in rels) and (
                            df_nodes[(df_nodes.nodeID == id_from1)]['type'].iloc[0] != 'YA') and (
                            df_nodes[(df_nodes.nodeID == id_to2)]['type'].iloc[0] == 'I'):
                        d1 = df_nodes[(df_nodes.nodeID == id_from1)]
                        d2 = df_nodes[(df_nodes.nodeID == id_to1)]

                        d11 = df_nodes[(df_nodes.nodeID == id_from2)]
                        d22 = df_nodes[(df_nodes.nodeID == id_to2)]

                        tto1i.append(d2['text'].iloc[0])
                        tfrom1i.append(d1['text'].iloc[0])

                        tto2i.append(d22['text'].iloc[0])
                        tfrom2i.append(d11['text'].iloc[0])

                        connect_idsi.append(id_to1)
                        loc_idsi.append(d1['nodeID'].iloc[0])
                        illoc_idsi.append(d22['nodeID'].iloc[0])
                        nodeset_idsi.append(match_nodeset)

        df_all_loc = pd.DataFrame({
            'locution': tfrom1l,
            'connection': tto1l,
            'illocution': tto2l,
            'id_locution': loc_idsl,
            'id_connection': connect_idsl,
            'id_illocution': illoc_idsl,
            'nodeset_id': nodeset_idsl})

        df_all = pd.DataFrame({
            'premise': tfrom1i,
            'connection': tto1i,
            'conclusion': tto2i,
            'id_premise': loc_idsi,
            'id_connection': connect_idsi,
            'id_conclusion': illoc_idsi,
            'nodeset_id': nodeset_idsi})

    except:
        st.error('Error loading nodeset')
    return df_all_loc, df_all


#####################  page content  #####################
st.title("Moral Foundations' Detection in Dialogical Arguments")

maps = glob.glob(r"/maps/*.json")
directory = "tem_maps"
parent_dir = "/"
temp_path = os.path.join(parent_dir, directory)

with st.expander("###### Arguments Source",expanded=True):
    type_aif = 'old'
    own_files = st.radio('Chose method of uploading files', ('Uploading OVA+ Annotation', 'Connect to AIFdb'),horizontal=True)

    if own_files == 'Uploading OVA+ Annotation':
        uploaded_json = st.file_uploader('Choose files', type='json', accept_multiple_files=True)
        if len(uploaded_json) < 1:
            st.stop()
        elif len(uploaded_json) >= 1:
            maps_dict = {}
            for file in uploaded_json:
                fjson = json.load(file)
                maps_dict[str(file.name)[:-5]] = fjson
                st.write(f'{file.name} saved sucessfully')
            df_all_loc, df_all = RetrieveNodes(maps_dict, from_dict=True, type_aif=str(type_aif).lower())

    elif own_files == 'Connect to AIFdb':
        nodeset_id_input = st.text_input("Select nodeset ID from AIFdb", "10453")
        if len(nodeset_id_input) < 1:
            st.stop()
        elif len(nodeset_id_input) > 1:
            file_json_nodeset = get_graph_url(f'http://www.aifdb.org/json/{nodeset_id_input}')
            df_all_loc, df_all = RetrieveNodesOnline(file_json_nodeset, nodeset_id_str=nodeset_id_input,
                                                     type_aif=str(type_aif).lower())



def AIF_converter(df_all_loc,df_all):
    ids_linked = df_all[df_all.id_connection.duplicated()].id_connection.unique()
    if len(ids_linked) > 0:
        df_all.loc[df_all.id_connection.isin(ids_linked), 'argument_linked'] = True
        df_all.argument_linked = df_all.argument_linked.fillna(False)
    else:
        df_all['argument_linked'] = False

    num_cols_args = ['id_premise', 'id_connection', 'id_conclusion']
    num_cols_locs = ['id_locution', 'id_connection', 'id_illocution']

    df_all[num_cols_args] = df_all[num_cols_args].astype('str')
    df_all_loc[num_cols_locs] = df_all_loc[num_cols_locs].astype('str')

    df_1 = df_all.merge(df_all_loc[['locution', 'id_illocution']], left_on='id_conclusion', right_on='id_illocution',
                        how='left')
    df_1 = df_1.iloc[:, :-1]
    df_1.columns = ['premise', 'connection', 'conclusion', 'id_premise', 'id_connection',
                    'id_conclusion', 'nodeset_id', 'argument_linked', 'locution_conclusion']

    df_2 = df_1.merge(df_all_loc[['locution', 'id_illocution']], left_on='id_premise', right_on='id_illocution', how='left')
    df_2 = df_2.iloc[:, :-1]
    df_2.columns = ['premise', 'connection', 'conclusion', 'id_premise', 'id_connection',
                    'id_conclusion', 'nodeset_id', 'argument_linked', 'locution_conclusion', 'locution_premise']

    df_2 = df_2[['locution_conclusion', 'locution_premise', 'conclusion', 'premise',
                 'connection', 'nodeset_id', 'id_conclusion', 'id_premise', 'id_connection', 'argument_linked']]

    df_2['speaker_conclusion'] = df_2.locution_conclusion.apply(lambda x: str(str(x).split(':')[0]).strip())
    df_2['speaker_premise'] = df_2.locution_premise.apply(lambda x: str(str(x).split(':')[0]).strip())
    df_2['speaker'] = df_2.apply(lambda x: x['speaker_conclusion'] == x['speaker_premise'], axis=1)
    df_2['speaker'] = np.where(df_2['speaker'] == True, df_2['speaker_conclusion'], '')
    df_2['speaker'] = np.where((df_2['speaker'] == '') & (df_2.id_premise > df_2.id_conclusion), df_2['speaker_premise'],
                               df_2['speaker'])
    df_2['speaker'] = np.where((df_2['speaker'] == '') & (df_2.id_premise < df_2.id_conclusion), df_2['speaker_conclusion'],
                               df_2['speaker'])

    arg_stats = pd.DataFrame(df_2.connection.value_counts().sort_values(ascending=False)).reset_index()
    arg_stats.columns = ['Type', 'Number']
    arg_stats_prc = pd.DataFrame(
        df_2.connection.value_counts(normalize=True).round(3).sort_values(ascending=False) * 100).reset_index()
    arg_stats_prc.columns = ['Type', 'Percentage']
    arg_stats = pd.concat([arg_stats, arg_stats_prc.iloc[:, -1:]], axis=1)


    arg_stats_spk = pd.DataFrame(df_2.speaker.value_counts().sort_values(ascending=False)).reset_index()
    arg_stats_spk.columns = ['Speaker', 'Number']
    arg_stats_prc_spk = pd.DataFrame(
        df_2.speaker.value_counts(normalize=True).round(3).sort_values(ascending=False) * 100).reset_index()
    arg_stats_prc_spk.columns = ['Speaker', 'Percentage']
    arg_stats_spk = pd.concat([arg_stats_spk, arg_stats_prc_spk.iloc[:, -1:]], axis=1)

    df_2 = df_2.reset_index()
    df_2["time"] = df_2['index'].astype('int')
    df_2["velocity"] = 1
    df_2 = df_2.drop(columns=['index', 'time', 'velocity'], axis=1)
    return df_2


col1,col2 = st.columns(2)
with col1:
    with st.expander("###### Dialogical Arguments",expanded=True):
        tab1, tab2 = st.tabs(["ADUs", "Arguments",])
        with tab1:
            df_2 = AIF_converter(df_all_loc, df_all)
            st.dataframe(df_all_loc, width=850)
        with tab2:
            df_2.argument_linked = df_2.argument_linked.astype('int')
            # st.dataframe(df_2.iloc[:, 2:].set_index('speaker').head(), width=850)
            st.dataframe(df_2.set_index('speaker'), width=850)

with col2:
    with st.expander("##### Data Operation",expanded=True):
        unit = st.radio("Please select the annotation unit",("ADUs","Arguments",))
        submission = st.button('Start Preprocessing!!!')
    if submission:
        care_virtue, care_vice, fairness_virtue, fairness_vice, loyalty_virtue, loyalty_vice, authority_virtue, authority_vice, sanctity_virtue, sanctity_vice = MFD()
        if unit == "ADUs":
            df_locu = Run_MV_Detection_ADU(df_all_loc,care_virtue, care_vice, fairness_virtue, fairness_vice, loyalty_virtue, loyalty_vice, authority_virtue, authority_vice, sanctity_virtue, sanctity_vice)
            df_arg = pd.DataFrame()
        else:
            df_locu = pd.DataFrame()
            df_arg = Run_MV_Detection_Arg(df_2,care_virtue, care_vice, fairness_virtue, fairness_vice, loyalty_virtue, loyalty_vice, authority_virtue, authority_vice, sanctity_virtue, sanctity_vice)
    else:
        df_locu = pd.DataFrame()
        df_arg = pd.DataFrame()

    if len(df_locu)!=0 and len(df_arg)==0:
        with st.expander("###### Download Moral Foundation Annotation",expanded=True):
            download_type = st.radio('Choose file format', ('CSV',))
            file_download = convert_df(df_locu)
            st.download_button(
                label="Click to download annotated ADUs",
                data=file_download,
                file_name="annotated_ADUs.csv",
                mime='text/csv', )

    if len(df_arg)!=0 and len(df_locu)==0:
        with st.expander("###### Download Moral Foundation Annotation",expanded=True):
            download_type = st.radio('Choose file format', ('CSV',))
            file_download = convert_df(df_arg)
            st.download_button(
                label="Click to download annotated arguments",
                data=file_download,
                file_name="annotated_arguments.csv",
                mime='text/csv', )
with st.expander("###### Annotated Dialogical Arguments",expanded=True):
    if len(df_locu)!=0:
        st.dataframe(df_locu)
    if len(df_arg)!=0:
        st.dataframe(df_arg)

