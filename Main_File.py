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
parliament = pd.read_excel("datasets/Parliament.xlsx")
adolescent_fertility_rate = pd.read_excel("datasets/Adolescent_Fertility_Rate.xlsx")
human_dev_indices = pd.read_excel("datasets/Human Development Composite Indices.xlsx")

# Veriseti isimlerini çıktı alabilmek için sözlük yaratalım:
df_names = {"f_to_m_unpaid_care_work": f_to_m_unpaid_care_work,
            "w_in_top_income_groups": w_in_top_income_groups,
            "f_to_m_labor_force_part": f_to_m_labor_force_part,
            "maternal_mortality": maternal_mortality,
            "gender_wage_gap": gender_wage_gap,
            "w_entrepreneurship": w_entrepreneurship,
            "male_labor_force": male_labor_force,
            "female_labor_force": female_labor_force,
            "placement": placement,
            "parliament": parliament,
            "adolescent_fertility_rate": adolescent_fertility_rate,
            "human_dev_indices": human_dev_indices}

#######################
# Verisetlerini İnceleme
#######################

f_to_m_unpaid_care_work.head()
w_in_top_income_groups.head()
f_to_m_labor_force_part.head()
maternal_mortality.head()
gender_wage_gap.head()
w_entrepreneurship.head()
male_labor_force.head()
female_labor_force.head()
placement.head()
parliament.head()
adolescent_fertility_rate.head()
human_dev_indices.head()

#######################
# Feature Engineering
#######################

# TODO NEDEN ENTITY DEDİĞİNİ ANLADIM! ÇÜNKÜ SADECE ÜLKELER DEĞİL, ÜLKE DIŞI YAPILAR, MESELA MENA GİBİ BÖLGELER DE VAR!
# Başka isimlerle verilen ülke değişkeni ismini Country olarak standardize edelim:
entity_dfs = [f_to_m_unpaid_care_work, w_in_top_income_groups, gender_wage_gap]

for df in entity_dfs:
    df.replace({"Entity":"Country"})

parliament = parliament.rename(columns={'Country Name' : 'Country'})


# Ülke ismi kısaltmalarını içeren değişkenleri silelim:
for name, df in df_names.items():
    vars_to_drop = ["Code", "Country Code", "ISO3", "No"]
    matched_vars = [var for var in vars_to_drop if var in df.columns]
    if matched_vars:
        print(f"Matched variables in {name}: {', '.join(matched_vars)}")
        for var in matched_vars:
            df.drop(columns=var, inplace=True)
        print(f"--> Dropped columns from {name}: {', '.join(matched_vars)}\n")
    else:
        print(f"No matched variables in {name}\n")


# Gereksiz indikatör isim değişkenlerini silelim:
for name, df in df_names.items():
    inds_to_drop = ["Indicator Name", "Indicator Code"]
    matched_inds = [ind for ind in inds_to_drop if ind in df.columns]
    if matched_inds:
        print(f"Indicator variables in {name}: {', '.join(matched_inds)}")
        print("###################")
        print(df[matched_inds].nunique())
        print("###################")
        for ind in matched_inds:
            df.drop(columns=ind, inplace=True)
        print(f"--> Dropped columns from {name}: {', '.join(matched_inds)}\n")
    else:
        print(f"No indicators in {name}\n")



# Değişken isimlerini kısaltalım:
maternal_mortality = maternal_mortality.rename(columns={'Maternal Mortality Ratio (Gapminder (2010) and World Bank (2015))' : 'Maternal Mortality Ratio'})
maternal_mortality.info()


# Cinsiyetlerin iş gücüne katılımını gösteren iki ayrı verisetini birleştirelim:
merged_labor_force = pd.merge(female_labor_force, male_labor_force, on=["ISO3", 'Country', "HDI Rank (2021)", "Continent", "Hemisphere"])
df_names.update({"merged_labor_force": merged_labor_force})
"merged_labor_force" in df_names # True
merged_labor_force.head()
merged_labor_force.info()
merged_labor_force.isnull().sum()

male_labor_force.isnull().sum()
female_labor_force.isnull().sum()

male_labor_force.head()

    # TODO Yılları sütun adı yapalım yazacağım ama emin olamadım - D
new_column_names = {}
cols = [col for col in merged_labor_force.columns if col not in ["ISO3", 'Country', "HDI Rank (2021)", "Continent", "Hemisphere"]]
for column_name in cols:
    # Eski sütun ismi içerisindeki yıl bilgisini kaldırarak yeni isim oluştur
    new_name = column_name.split(" ")[4] + "_" + column_name.split(" ")[-1]
    new_column_names[column_name] = new_name

    # Yeni isimlerle sütunları yeniden adlandıralım
merged_labor_force.rename(columns=new_column_names, inplace=True)

merged_labor_force.info()
merged_labor_force.head()




# this is myyy branch guysszzzz!!!!

#25 mart parlemanto verisi
parliament.info()
parliament.head()
parliament.columns

parliament["Indicator Name"]

# Farklı yılları temsil eden sütunları, satır yapalım:
parliament = pd.melt(parliament, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], var_name='Year', value_name='Women Seat Ratio')
# Kontrol edelim:
parliament.tail()
parliament.info()

# TODO bu ne yapıyor? - D
parliament["Year"] = parliament[parliament["Year"]]

# Farklı yılları temsil eden sütunları, satır yapalım:
male_labor_force = pd.melt(male_labor_force, id_vars=['Country', 'Continent', 'Hemisphere', 'HDI Rank (2021)'], var_name='Labour Force Participation Rate', value_name='Male Labour Force Participation Rate')
male_labor_force.head()

# Eski sütun ismi içerisindeki yıl bilgisini kaldırarak yeni isim oluşturalım:
m_val = []
for val in male_labor_force["Labour Force Participation Rate"].values:
    new_name = val.split(" ")[-1].replace("(", "").replace(")", "")
    m_val.append(new_name)

male_labor_force['Year'] = male_labor_force['Labour Force Participation Rate'].replace(male_labor_force['Labour Force Participation Rate'].tolist(), m_val)

# Verisetini şu anki gibi yıla göre değil, ülkelere göre alfabetik sıralayalım:
male_labor_force.sort_values(by='Country', inplace=True)
male_labor_force.head()

# Fazla değişkenleri silelim:
male_labor_force.drop(["HDI Rank (2021)", "Labour Force Participation Rate", "Hemisphere"], axis=1, inplace=True)


# Farklı yılları temsil eden sütunları, satır yapalım:
female_labor_force = pd.melt(female_labor_force, id_vars=['ISO3', 'Country', 'Continent', 'Hemisphere', 'HDI Rank (2021)'], var_name='Labour Force Participation Rate', value_name='Female Labour Force Participation Rate')
female_labor_force.head()
female_labor_force.tail()

# Eski sütun ismi içerisindeki yıl bilgisini kaldırarak yeni isim oluşturalım:
f_val = []
for val in female_labor_force["Labour Force Participation Rate"].values:
    new_name = val.split(" ")[-1].replace("(", "").replace(")", "")
    f_val.append(new_name)

female_labor_force['Year'] = female_labor_force['Labour Force Participation Rate'].replace(female_labor_force['Labour Force Participation Rate'].tolist(), f_val)

# Verisetini şu anki gibi yıla göre değil, ülkelere göre alfabetik sıralayalım:
female_labor_force.sort_values(by='Country', inplace=True)
female_labor_force.head()

# Fazla değişkenleri silelim:
female_labor_force.drop(["HDI Rank (2021)", "Labour Force Participation Rate", "Hemisphere"], axis=1, inplace=True)

# TODO male ve female labor_force verisetlerini aynı işlemlerden geçiriyoruz. Daha şık olması için bu ikisini liste + fonksiyon + for_döngüsü ile birleştirelim.


#####

# TODO buradaki country ve indicator ile ilgili yukarıda derleme yapıldı. Ama belki de tek tek gitmeyi tercih ederiz.
adolescent_fertility_rate.head()
adolescent_fertility_rate["Country Name"].value_counts()
adolescent_fertility_rate = pd.melt(adolescent_fertility_rate, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], var_name='Year', value_name='Adolescent fertility rate')
adolescent_fertility_rate.head()
adolescent_fertility_rate.drop(columns=["Indicator Name", "Indicator Code"], axis=1, inplace=True)

gender_wage_gap.columns # TODO target

parliament.columns
# ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', 'Year', 'Women Seat Ratio']
parliament = parliament.rename(columns={'Country Name' : 'Country'})
parliament.drop(["Indicator Name", "Indicator Code"], axis=1, inplace=True)
parliament.head()

maternal_mortality.columns
# ['Country', 'Code', 'Year', 'Maternal Mortality Ratio (Gapminder (2010) and World Bank (2015))']
maternal_mortality.drop("Code", axis=1, inplace=True)
maternal_mortality = maternal_mortality.rename(columns={'Maternal Mortality Ratio (Gapminder (2010) and World Bank (2015))' : 'Maternal Mortality Ratio'})
maternal_mortality.head()

male_labor_force.columns # labor
# ['ISO3', 'Country', 'Continent', 'Hemisphere', 'Male Labour Force Participation Rate', 'Year']
male_labor_force.drop(["ISO3", "Continent", "Hemisphere"], axis=1, inplace=True)
male_labor_force.head()

female_labor_force.columns # labor
# ['ISO3', 'Country', 'Continent', 'Hemisphere', 'Female Labour Force Participation Rate', 'Year']
female_labor_force.drop(["ISO3", "Continent", "Hemisphere"], axis=1, inplace=True)
female_labor_force.head()

f_to_m_labor_force_part.columns # labor
# ['Entity', 'Code', 'Year', 'Ratio of female to male labor force participation rate (%) (modeled ILO estimate)']
f_to_m_labor_force_part = f_to_m_labor_force_part.rename(columns={'Entity' : 'Country', 'Ratio of female to male labor force participation rate (%) (modeled ILO estimate)' : 'Ratio of female to male labor force participation rate' })
f_to_m_labor_force_part.drop("Code", axis=1, inplace=True)
f_to_m_labor_force_part.head()

adolescent_fertility_rate.columns
# ['Country Name', 'Country Code', 'Indicator Code', 'Year', 'Adolescent fertility rate']
adolescent_fertility_rate = adolescent_fertility_rate.rename(columns={'Country Name': 'Country'})
adolescent_fertility_rate.drop(["Country Code", "Indicator Code"], axis=1, inplace=True)

# TODO Biz burada bir sıkıntı var sandık (merge, aldığı iki verisetinde de ortak olmayanları sildiği için, merge sıralamasının veri kaybetmemek içni önemli olduğunu düşündük) ama her halükarda en küçükle en büyük veriseti çarpışacağı için aslında önemli değil.
merged_df = pd.merge(gender_wage_gap, parliament, on=['Country', "Year"])
merged_df = pd.merge(merged_df, maternal_mortality, on=['Country', "Year"])
merged_df = pd.merge(merged_df, male_labor_force, on=['Country', "Year"])
merged_df = pd.merge(merged_df, female_labor_force, on=['Country', "Year"])
merged_df = pd.merge(merged_df, f_to_m_labor_force_part, on=['Country', "Year"])
merged_df["Year"].describe().T
merged_df.shape # 301, 8

# Merge'e alternatif concat çalıştık.
dfs_to_concat = [gender_wage_gap, parliament, maternal_mortality, male_labor_force, female_labor_force, f_to_m_labor_force_part]
merged_df_2 = pd.concat(dfs_to_concat, ignore_index=True)
merged_df_2.shape # 41883, 8
merged_df_2.head()

merged_df_2.nunique()

for df in dfs_to_concat:
    print(df.Year.describe().T)
    print("\n")

# Bu verisetlerindeki Year değişkenini biz metinden split ile yarattığımız dtype'ı farklı olmuş.
# TODO split olmasa da aynı melt işlemini başka verisetlerine de uyguladık, kontrol et.
female_labor_force["Year"] = female_labor_force["Year"].astype("int")
female_labor_force.info()

male_labor_force["Year"] = male_labor_force["Year"].astype("int")
male_labor_force.info()

# Burada merge'deki sıkıntıyı gidermek için, target'taki ülkeler arasında olmayan ülkeleri yakalayıp silmek istiyoruz (Ama işte yukarıdaki nedenden dolayı gereksiz gibi).
gender_set = set(gender_wage_gap['Country'])


other_union = set().union(*[set(df['Country']) for df in dfs_to_concat if not df.equals(gender_wage_gap)])

# A_set'te olmayıp diğerlerinin birleşiminde olan ülkelerin farkını hesaplayalım
difference = other_union.difference(gender_set)

sorted_gender_set = sorted(gender_set)
sorted_difference = sorted(difference)

print("gender DataFrame'deki ülkeler (alfabetik sıralı):")
print(sorted_gender_set)
print("\ngender'da olmayıp diğerlerinin birleşiminde olan ülkeler (alfabetik sıralı):")
print(sorted_difference)