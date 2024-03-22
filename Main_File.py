import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve   #plot_roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)



df_1 = pd.read_csv("datasets/1- female-to-male-ratio-of-time-devoted-to-unpaid-care-work.csv")
df_1.head()

df_2 = pd.read_csv("datasets/2- share-of-women-in-top-income-groups.csv")
df_3 = pd.read_csv("datasets/3- ratio-of-female-to-male-labor-force-participation-rates-ilo-wdi.csv")
df_4 = pd.read_csv("datasets/4- female-to-male-ratio-of-time-devoted-to-unpaid-care-work.csv")
df_5 = pd.read_csv("datasets/5- maternal-mortality.csv")
df_6 = pd.read_csv("datasets/6- gender-gap-in-average-wages-ilo.csv")
df_7 = pd.read_csv("datasets/Labor Force-Women Entrpreneurship.csv", sep=";")
df_8 = pd.read_csv("datasets/Labour Force Participation - Male.csv")
df_9 = pd.read_csv("datasets/Labour Force Participation Female.csv")
df_10 = pd.read_csv("datasets/Placement.csv")
df_11 = pd.read_csv("datasets/Women Ent_Data3.csv", sep=";")

# tek tek okutmamak için, for döngüsü içinde okutma

name_of_files = ["1- female-to-male-ratio-of-time-devoted-to-unpaid-care-work",
                 "2- share-of-women-in-top-income-groups",
                 "3- ratio-of-female-to-male-labor-force-participation-rates-ilo-wdi",
                 "4- female-to-male-ratio-of-time-devoted-to-unpaid-care-work",
                 "5- maternal-mortality",
                 "6- gender-gap-in-average-wages-ilo",
                 "Labor Force-Women Entrpreneurship",
                 "Labour Force Participation - Male",
                 "Labour Force Participation Female",
                 "Placement",
                 "Women Ent_Data3"]
dfs = []
for i in name_of_files:
    df = pd.read_csv(f"datasets/{i}.csv")
    dfs.append(df)


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NaN #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.95, 0.99, 1]))
    print("##################### Category Numbers #####################")
    print(dataframe.nunique())

for dataframe in dfs:
    print(check_df(dataframe))

df_1.head()
df_1.columns
df_1.shape
#['Entity', 'Code', 'Year', 'Female to male ratio of time devoted to unpaid care work (OECD (2014))']
# Entity: ülke ismi (ör: Albania). Kategorik
# Code: ülke kodu (ör: ALB). Kategorik
# Year: 2014 yılı
# F/M: Ne kadar çok kadın, erkeğe göre ev işi yükleniyor? Min: 1.18, Max: 17.29. Numerik.
'''(Rapordan) Note: Gender inequality in unpaid care work refers to the female to
male ratio of time spent in unpaid care work. The fitted value of the
female share in the active population is estimated by controlling for
the country’s GDP per capita, fertility rate, urbanisation rate, maternity
leave policies and gender inequality in unemployment and education.'''

df_2.head()
df_2.columns
#['Entity', 'Code', 'Year', 'Share of women in top 0.1%', 'Share of women in top 0.25%', 'Share of women in top 0.5%', 'Share of women in top 1%','Share of women in top 10%', 'Share of women in top 5%']
# Share of women: Maaş olarak en üst %x'te yer alan kadınların oranı. Numerik.

df_3.head()
df_3.columns
#['Entity', 'Code', 'Year', 'Ratio of female to male labor force participation rate (%) (modeled ILO estimate)']

# df_1 ve df_4 aynı dosyalar
df_4.head()
df_4.columns
#['Entity', 'Code', 'Year', 'Female to male ratio of time devoted to unpaid care work (OECD (2014))']
df_4.shape


# Maternal Mortality Ratio (MMR), bir ülkede veya bölgede her 100.000 canlı doğum başına annenin ölümünü ölçen bir sağlık göstergesidir.
df_5.head()
df_5.columns
# ['Entity', 'Code', 'Year', 'Maternal Mortality Ratio (Gapminder (2010) and World Bank (2015))']
df_5.shape
# (5800, 4)

df_6.head()
df_6.columns
# ['Entity', 'Code', 'Year', 'Gender wage gap (%)']
df_6.shape
# (413, 4)

df_7.head()
df_7.columns
#['No', 'Country', 'Level of development', 'European Union Membership', 'Currency',
# 'Women Entrepreneurship Index', 'Entrepreneurship Index', 'Inflation rate','Female Labor Force Participation Rate']

df_8.head()
# 195 ülke için, 1990-2021 arasındaki yıllar için, 15 yaş ve üzeri erkeklerin iş gücüne katılma oranları
df_8.columns
#['ISO3', 'Country', 'Continent', 'Hemisphere', 'HDI Rank (2021)',
# 'Labour force participation rate, male (% ages 15 and older) (1990-2021)
df_8.shape

df_9.head()
df_9.columns
df_9.shape
# 195 ülke için, 1990-2021 arasındaki yıllar için, 15 yaş ve üzeri kadınların iş gücüne katılma oranları
# TODO 8 ve 9 birleştirilebilir

df_10.head()
df_10.columns
#['gender', 'ssc_percentage', 'ssc_board', 'hsc_percentage', 'hsc_board', 'hsc_subject',
# 'degree_percentage', 'undergrad_degree', 'work_experience','emp_test_percentage', 'specialisation', 'mba_percent', 'status']
df_10.shape
#(215, 13)
df_10.hsc_subject.value_counts()
df_10.info()
# todo

df_11.head()
df_11.columns
#['No', 'Country', 'Level of development', 'European Union Membership', 'Currency',
# 'Women Entrepreneurship Index', 'Entrepreneurship Index', 'Inflation rate','Female Labor Force Participation Rate']
# No
# Country
# Level of development: Kategorik
# EU Membership: Kategorik
# Currency: Kategorik
# TODO Women Entrepreneurship Index önceki bir df'te vardı
df_11.info()
df_11.sort_values(by="Country")
# TODO No değişkeni drop edilebiliri.