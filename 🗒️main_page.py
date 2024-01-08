# Contents of ~/my_app/main_page.py
import streamlit as st
from streamlit_tree_select import tree_select

st.sidebar.markdown("# Introduction 🎈")

def add_spacelines(number=2):
    for i in range(number):
        st.write("\n")

st.title("MArgSyn: Moral Argument Synthesis")
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

st.write("#### Moral Value Analysis Metrics")
st.write("###### Moral Value Occurrence (Argumentative Discourse Units or Arguments)")
with st.expander("Definition"):
    add_spacelines(1)
    st.write(""" Drawing from the moral foundation dictionary, 
    we assess the presence of specific moral values within text sequences. 
    Additionally, we've introduced the **No Morals** category to **designate sequences that lack moral values**. 
    Taking into account the five moral foundations, each associated with two valence categories, we distinguish between **10 moral value categories**, 
    plus the category **No Morals** designed for sequences devoid of moral values.""")
st.write("###### Moral Value Score（Interlocutors)")
with st.expander("Definition"):
    add_spacelines(1)
    st.write("""We compute **the proportion of sequences that encompass specific moral values** across the 10 predefined moral value categories to obtain interlocutors' moral value score.""")
st.write("###### Moral Valence Degree（Interlocutors)")
with st.expander("Definition"):
    add_spacelines(1)
    st.write("""For each moral foundation attributed to an interlocutor within the corpora, we categorise it into one of four moral valence categories: 'only virtue', 'only vice', 'mixed', and 'no specific moral values'.""")
    st.write("Taking the 'care' foundation as an example:")
    st.write("***only care+***: This signifies that when referencing the care foundation, speakers exclusively utilise care virtues in their discourse.")
    st.write("***only care-***: This implies that when mentioning the care foundation, speakers solely incorporate care vices.")
    st.write("***mixed care***: Under this classification, when the care foundation is mentioned, speakers employ both care virtues and vices.")
    st.write("***no care***: Here, speakers abstain from integrating care foundation into their speech.")

# st.write("***************************************************************************************")
# info1 = st.container()
# with info1:
#     st.write(
#         "For technical support 👨‍💻: [zhanghe1019@hotmail.com](zhanghe1019@hotmail.com)")
#     st.write(
#         "Know more about our lab 💻: [The New Ethos](https://newethos.org/laboratory/)")

with st.container():
    hide_footer_style = """
        <style>
        footer {visibility: visible;
                color : white;
                background-color: #d2cdcd;}

        footer:after{
        visibility: visible;
        content : 'Project developed by: XX';
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




