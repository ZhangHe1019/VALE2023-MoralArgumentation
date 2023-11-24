import pandas as pd
data1 = pd.read_excel(r"BritishEmpire Pair.xlsx",index_col=0)
data2 = pd.read_excel(r"DDay Pair.xlsx",index_col=0)
data3 = pd.read_excel(r"Money Pair.xlsx",index_col=0)
data4 = pd.read_excel(r"Welfare Pair.xlsx",index_col=0)
data5 = pd.read_excel(r"Hypocrisy Pair.xlsx",index_col=0)


data_1 = pd.read_excel(r"BritishEmpire_Logo_Moral.xlsx",index_col=0)
data_2 = pd.read_excel(r"DDay_Logo_Moral.xlsx",index_col=0)
data_3 = pd.read_excel(r"Money_Logo_Moral.xlsx",index_col=0)
data_4 = pd.read_excel(r"Welfare_Logo_Moral.xlsx",index_col=0)
data_5 = pd.read_excel(r"Hypocrisy_Logo_Moral.xlsx",index_col=0)


data1.to_csv(r"BritishEmpire Pair.csv")
data2.to_csv(r"DDay Pair.csv")
data3.to_csv(r"Money Pair.csv")
data4.to_csv(r"Welfare Pair.csv")
data5.to_csv(r"Hypocrisy Pair.csv")


data_1.to_csv(r"BritishEmpire_Logo_Moral.csv")
data_2.to_csv(r"DDay_Logo_Moral.csv")
data_3.to_csv(r"Money_Logo_Moral.csv")
data_4.to_csv(r"Welfare_Logo_Moral.csv")
data_5.to_csv(r"Hypocrisy_Logo_Moral.csv")