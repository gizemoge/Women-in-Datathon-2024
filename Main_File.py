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

# 23 mart - gizem
name_of_files = ["1- female-to-male-ratio-of-time-devoted-to-unpaid-care-work",
                 "2- share-of-women-in-top-income-groups",
                 "3- ratio-of-female-to-male-labor-force-participation-rates-ilo-wdi",
                 "5- maternal-mortality",
                 "Labor Force-Women Entrpreneurship",
                 "Labour Force Participation - Male",
                 "Labour Force Participation Female",
                 "Placement",
                 "Women Ent_Data3"]
new_names = [
    "f_to_m_unpaid_care_work",
    "w_in_top_income_groups",
    "f_to_m_labor_force_part",
    "maternal_mortality",
    "gender_wage_gap",
    "w_entrepreneurship",
    "male_labor_force",
    "female_labor_force",
    "placement"
]

dfs = [pd.read_csv(f"datasets/{name}.csv", sep=",") if sep == "," else pd.read_csv(f"datasets/{name}.csv", sep=";") for name in name_of_files]

dfs_dict = {new_name: df for new_name, df in zip(new_names, dfs)}

for isim, df in dfs_dict.items():
    globals()[isim] = df


# 23 mart - gizem's alternative (normal okunuş)
f_to_m_unpaid_care_work = pd.read_csv("datasets/1- female-to-male-ratio-of-time-devoted-to-unpaid-care-work.csv")
w_in_top_income_groups = pd.read_csv("datasets/2- share-of-women-in-top-income-groups.csv")
f_to_m_labor_force_part = pd.read_csv("datasets/3- ratio-of-female-to-male-labor-force-participation-rates-ilo-wdi.csv")
maternal_mortality = pd.read_csv("datasets/5- maternal-mortality.csv")
gender_wage_gap = pd.read_csv("datasets/6- gender-gap-in-average-wages-ilo.csv")
w_entrepreneurship = pd.read_csv("datasets/Labor Force-Women Entrpreneurship.csv", sep=";")
male_labor_force = pd.read_csv("datasets/Labour Force Participation - Male.csv")
female_labor_force = pd.read_csv("datasets/Labour Force Participation Female.csv")
placement = pd.read_csv("datasets/Placement.csv")
# 1 ve 4, 7 ve 11 excel'ler aynı


f_to_m_unpaid_care_work.head()
f_to_m_unpaid_care_work.columns
f_to_m_unpaid_care_work.shape
#['Entity', 'Code', 'Year', 'Female to male ratio of time devoted to unpaid care work (OECD (2014))']
# Entity: ülke ismi (ör: Albania). Kategorik
# Code: ülke kodu (ör: ALB). Kategorik
# Year: 2014 yılı
# F/M: Ne kadar çok kadın, erkeğe göre ev işi yükleniyor? Min: 1.18, Max: 17.29. Numerik.
f_to_m_unpaid_care_work = f_to_m_unpaid_care_work.rename(columns={'Entity' : 'Country'})

'''(Rapordan) Note: Gender inequality in unpaid care work refers to the female to
male ratio of time spent in unpaid care work. The fitted value of the
female share in the active population is estimated by controlling for
the country’s GDP per capita, fertility rate, urbanisation rate, maternity
leave policies and gender inequality in unemployment and education.'''

w_in_top_income_groups.head()
w_in_top_income_groups.columns
#['Entity', 'Code', 'Year', 'Share of women in top 0.1%', 'Share of women in top 0.25%', 'Share of women in top 0.5%', 'Share of women in top 1%','Share of women in top 10%', 'Share of women in top 5%']
# Share of women: Maaş olarak en üst %x'te yer alan kadınların oranı. Numerik.
w_in_top_income_groups = w_in_top_income_groups.rename(columns={'Entity' : 'Country'})


f_to_m_labor_force_part.head()
f_to_m_labor_force_part.columns
#['Entity', 'Code', 'Year', 'Ratio of female to male labor force participation rate (%) (modeled ILO estimate)']
f_to_m_labor_force_part = f_to_m_labor_force_part.rename(columns={'Entity' : 'Country'})


# Maternal Mortality Ratio (MMR), bir ülkede veya bölgede her 100.000 canlı doğum başına annenin ölümünü ölçen bir sağlık göstergesidir.
maternal_mortality.head()
maternal_mortality.columns
# ['Entity', 'Code', 'Year', 'Maternal Mortality Ratio (Gapminder (2010) and World Bank (2015))']
maternal_mortality.shape
maternal_mortality = maternal_mortality.rename(columns={'Entity' : 'Country'})

# (5800, 4)

gender_wage_gap.head()
gender_wage_gap.columns
# ['Entity', 'Code', 'Year', 'Gender wage gap (%)']
gender_wage_gap.shape
gender_wage_gap = gender_wage_gap.rename(columns={'Entity' : 'Country'})

# (413, 4)

w_entrepreneurship.head()
w_entrepreneurship.shape # (51,9)
w_entrepreneurship.columns
#['No', 'Country', 'Level of development', 'European Union Membership', 'Currency',
# 'Women Entrepreneurship Index', 'Entrepreneurship Index', 'Inflation rate','Female Labor Force Participation Rate']
# Level of development: Kategorik
# EU Membership: Kategorik
# Currency: Kategorik
# TODO No değişkeni drop edilebiliriz.

male_labor_force.head()
# 195 ülke için, 1990-2021 arasındaki yıllar için, 15 yaş ve üzeri erkeklerin iş gücüne katılma oranları
male_labor_force.columns
#['ISO3', 'Country', 'Continent', 'Hemisphere', 'HDI Rank (2021)',
# 'Labour force participation rate, male (% ages 15 and older) (1990-2021)
male_labor_force.shape

female_labor_force.head()
female_labor_force.columns
female_labor_force.shape
# 195 ülke için, 1990-2021 arasındaki yıllar için, 15 yaş ve üzeri kadınların iş gücüne katılma oranları
# TODO 8 ve 9 birleştirilebilir
countries=[]
[countries.append if female_labor_force.Country not in male_labor_force.Country]


placement.head()
placement.columns
#['gender', 'ssc_percentage', 'ssc_board', 'hsc_percentage', 'hsc_board', 'hsc_subject',
# 'degree_percentage', 'undergrad_degree', 'work_experience','emp_test_percentage', 'specialisation', 'mba_percent', 'status']
placement.shape
#(215, 13)
placement.hsc_subject.value_counts()
placement.info()
# todo.


# iş gücüne katılım
merged_labor_force = pd.merge(female_labor_force, male_labor_force, on=["ISO3", 'Country', "HDI Rank (2021)", "Continent", "Hemisphere"])
merged_labor_force.head()
merged_labor_force.info()
merged_labor_force.isnull().sum()


male_labor_force.isnull().sum()
female_labor_force.isnull().sum()

#
new_column_names = {}
cols = [col for col in merged_labor_force.columns if col not in ["ISO3", 'Country', "HDI Rank (2021)", "Continent", "Hemisphere"]]
for column_name in cols:
    # Eski sütun ismi içerisindeki yıl bilgisini kaldırarak yeni isim oluştur
    new_name = column_name.split(" ")[4] + "_" + column_name.split(" ")[-1]
    new_column_names[column_name] = new_name

# Yeni isimlerle sütunları yeniden adlandıralım
merged_labor_force.rename(columns=new_column_names, inplace=True)

merged_labor_force.info()


merged_all = pd.merge(female_labor_force, male_labor_force, on=["Year", 'Country'])


# this is myyy branch guysszzzz!!!!

