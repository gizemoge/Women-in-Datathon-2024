##############################################
# DATATHON MART 2024
##############################################

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
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


##############################################
# VERİ SETİ İNCELEME VE AÇIKLAMASI
##############################################
# Size sağlanacak olan veri seti, belirli bir konuya ait gerçek verilerden oluşmaktadır.
# Bu veri seti üzerinde analiz yaparken, veri setlerini detaylıca incelemeniz, anlamanız ve anlamlı bir şekilde yorumlayabilmelisiniz.
# Hatırlatma: Dış kaynaklardan ekstra veriler bularak veri analizi çalışmanızı zenginleştirebilirsiniz.


# 1- female-to-male-ratio-of-time-devoted-to-unpaid-care-work
'''(Rapordan) Note: Gender inequality in unpaid care work refers to the female to
male ratio of time spent in unpaid care work. The fitted value of the
female share in the active population is estimated by controlling for
the country’s GDP per capita, fertility rate, urbanisation rate, maternity
leave policies and gender inequality in unemployment and education.'''
f_to_m_unpaid_care_work.head()
f_to_m_unpaid_care_work.shape # (69, 4)
f_to_m_unpaid_care_work.info() # veritiplerinde sıkıntı yok.
"""
 #   Column                                                                  Non-Null Count  Dtype    Notes
---  ------                                                                  --------------  -----    -----
 0   Entity                                                                  69 non-null     object   69 nunique
 1   Code                                                                    68 non-null     object 
 2   Year                                                                    69 non-null     int64    2014 SADECE
 3   Female to male ratio of time devoted to unpaid care work (OECD (2014))  69 non-null     float64  Ne kadar çok kadın, erkeğe göre ev işi yükleniyor? 
        mean     3.249
        std      2.511
        min      1.180
        25%      1.810
        50%      2.530
        75%      3.380
        max     17.290
"""


# 2- share-of-women-in-top-income-groups
w_in_top_income_groups.head()
w_in_top_income_groups.shape # (168, 9)
w_in_top_income_groups.info() # veritiplerinde sıkıntı yok.
""" 
 #   Column                       Non-Null Count  Dtype    Notes
---  ------                       --------------  -----    -----
 0   Entity                       168 non-null    object   8 nunique: New Zealand, Denmark, Canada, UK, Italy, Australia, Spain, Norway
 1   Code                         148 non-null    object 
 2   Year                         168 non-null    int64    1980-2015
 3   Share of women in top 0.1%   131 non-null    float64  Maaş olarak en üst %0.1'de yer alan kadınların oranı
 4   Share of women in top 0.25%  37 non-null     float64
 5   Share of women in top 0.5%   82 non-null     float64
 6   Share of women in top 1%     167 non-null    float64
 7   Share of women in top 10%    168 non-null    float64
 8   Share of women in top 5%     168 non-null    float64
"""


# 3- ratio-of-female-to-male-labor-force-participation-rates-ilo-wdi
f_to_m_labor_force_part.head()
f_to_m_labor_force_part.shape # (6432, 4)
f_to_m_labor_force_part.info() # veritiplerinde sıkıntı yok.
"""
 #   Column                                                                             Non-Null Count  Dtype    Notes
---  ------                                                                             --------------  -----    -----
 0   Entity                                                                             6432 non-null   object   201 nunique
 1   Code                                                                               5984 non-null   object 
 2   Year                                                                               6432 non-null   int64    1990-2021
 3   Ratio of female to male labor force participation rate (%) (modeled ILO estimate)  6432 non-null   float64   
        mean      68.872
        std       19.953
        min        8.863
        25%       57.791
        50%       73.685
        75%       83.292
        max      108.372
"""


#4 (1 ile aynı olduğu için silindi)


# 5- maternal-mortality
# The maternal mortality ratio is the number of women who die from pregnancy-related causes while pregnant or within 42 days of pregnancy termination per 100,000 live births.
maternal_mortality.head()
maternal_mortality.shape # (5800, 4)
maternal_mortality.info() # veritiplerinde sıkıntı yok.
"""
 #   Column                                                             Non-Null Count  Dtype    Notes
---  ------                                                             --------------  -----    -----
 0   Entity                                                             5800 non-null   object   200 nunique
 1   Code                                                               5548 non-null   object 
 2   Year                                                               5800 non-null   int64    1751-2020
 3   Maternal Mortality Ratio (Gapminder (2010) and World Bank (2015))  5800 non-null   float64
        mean     216.929
        std      297.109
        min        0.000
        25%       13.000
        50%       61.185
        75%      356.000
        max     2480.000
"""


# 6- gender-gap-in-average-wages-ilo
gender_wage_gap.head()
gender_wage_gap.shape # (413, 4)
gender_wage_gap.info() # veritiplerinde sıkıntı yok.
"""
 #   Column               Non-Null Count  Dtype    Notes
---  ------               --------------  -----    -----
 0   Entity               413 non-null    object   64 nunique
 1   Code                 413 non-null    object   
 2   Year                 413 non-null    int64    1981-2016
 3   Gender wage gap (%)  413 non-null    float64
        mean     10.887
        std      10.242
        min     -28.270
        25%       3.920
        50%      10.680
        75%      17.630
        max      35.750
"""


# adolescent_fertility_rate
adolescent_fertility_rate.head()
adolescent_fertility_rate.shape # (266, 65)
adolescent_fertility_rate.info() # veritiplerinde sıkıntı yok.
"""
 #   Column        Non-Null Count  Dtype    Notes
---  ------        --------------  -----    -----
 0   Country Name  266 non-null    object   266 nunique
 1   Country Code  266 non-null    object 
 2   1960          265 non-null    float64
 ...
 63  2021          265 non-null    float64
 64  2022          0 non-null      float64   silinebilir
"""


# human_dev_indices
human_dev_indices.head()
human_dev_indices.shape # (206, 1076)
human_dev_indices.info() # TODO sorun nedir?
"""
"""


# Labor Force-Women Entrpreneurship
w_entrepreneurship.head()
w_entrepreneurship.shape # (51,9)
w_entrepreneurship.info() # veritiplerinde sıkıntı yok.
"""
 #   Column                                 Non-Null Count  Dtype    Notes  
---  ------                                 --------------  -----    -----
 0   No                                     51 non-null     int64    
 1   Country                                51 non-null     object   51 nunique
 2   Level of development                   51 non-null     object   Developed 27, Developing 24
 3   European Union Membership              51 non-null     object   Not member 31, Member 20
 4   Currency                               51 non-null     object   National Currency 36, Euro 15
 5   Women Entrepreneurship Index           51 non-null     float64
 6   Entrepreneurship Index                 51 non-null     float64
 7   Inflation rate                         51 non-null     float64
 8   Female Labor Force Participation Rate  51 non-null     float64
        mean    58.482
        std     13.865
        min     13.000
        25%     55.800
        50%     61.000
        75%     67.400
        max     82.300
"""


# Labour Force Participation - Male
# 195 ülke için, 1990-2021 arasındaki yıllar için, 15 yaş ve üzeri erkeklerin iş gücüne katılma oranları
male_labor_force.head()
male_labor_force.shape # (195, 37)
male_labor_force.info() # veritiplerinde sıkıntı yok.
"""
 #   Column                                                              Non-Null Count  Dtype    Notes   
---  ------                                                              --------------  -----    -----
 0   ISO3                                                                195 non-null    object   Country code
 1   Country                                                             195 non-null    object   195 nunique
 2   Continent                                                           195 non-null    object 
 3   Hemisphere                                                          195 non-null    object 
 4   HDI Rank (2021)                                                     191 non-null    float64
 5   Labour force participation rate, male (% ages 15 and older) (1990)  180 non-null    float64
 ...
 36  Labour force participation rate, male (% ages 15 and older) (2021)  180 non-null    float64
"""


# Labour Force Participation - Female
# 195 ülke için, 1990-2021 arasındaki yıllar için, 15 yaş ve üzeri kadınların iş gücüne katılma oranları
female_labor_force.head()
female_labor_force.shape # (195, 37)
female_labor_force.info() # veritiplerinde sıkıntı yok.
"""
 #   Column                                                                Non-Null Count  Dtype    Notes   
---  ------                                                                --------------  -----    -----
 0   ISO3                                                                  195 non-null    object   Country code
 1   Country                                                               195 non-null    object   195 nunique
 2   Continent                                                             195 non-null    object 
 3   Hemisphere                                                            195 non-null    object 
 4   HDI Rank (2021)                                                       191 non-null    float64
 5   Labour force participation rate, female (% ages 15 and older) (1990)  180 non-null    float64
 ...
 36  Labour force participation rate, female (% ages 15 and older) (2021)  180 non-null    float64
"""


# parliament
parliament.head()
parliament.shape # (266, 65)
parliament.info() # veritiplerinde sıkıntı yok.
"""
 #   Column        Non-Null Count  Dtype    Notes  
---  ------        --------------  -----    -----
 0   Country Name  266 non-null    object   266 nunique
 1   Country Code  266 non-null    object 
 2   1960          0 non-null      float64  1960-1996 arası null sayısı 0.
 ...
 38  1996          0 non-null      float64
 39  1997          199 non-null    float64  1997-2022 arası null sayısı değişiyor.
 ...
 64  2022          235 non-null    float64
"""


# Placement
placement.head()
placement.shape # (215, 13)
placement.info() # veritiplerinde sıkıntı yok.
"""
 #   Column               Non-Null Count  Dtype    Notes  
---  ------               --------------  -----    -----
 0   gender               215 non-null    object   M/F
 1   ssc_percentage       215 non-null    float64
 2   ssc_board            215 non-null    object   Others/Central
 3   hsc_percentage       215 non-null    float64
 4   hsc_board            215 non-null    object 
 5   hsc_subject          215 non-null    object   Commerce 113, Science 91, Arts 11
 6   degree_percentage    215 non-null    float64
 7   undergrad_degree     215 non-null    object   Comm&Mgmt 145, Sci&Tech 59, Others 11
 8   work_experience      215 non-null    object   No 141, Yes 74
 9   emp_test_percentage  215 non-null    float64
 10  specialisation       215 non-null    object   Mkt&Fin 120, Mkt&HR 95
 11  mba_percent          215 non-null    float64
 12  status               215 non-null    object   Placed/Not placed
"""

#Women Ent_Data3 (7 ile aynı olduğu için silindi)



##############################################
# DEEP DATA
##############################################
# Verinin özünü anlama ve derinlemesine analiz etme yeteneği büyük önem taşımaktadır.
# Veri setinin karmaşıklığını anlayarak, içindeki değerli bilgileri keşfetmeye odaklanmanız önemlidir.

# TODO NEDEN ENTITY DEDİĞİNİ ANLADIM! ÇÜNKÜ SADECE ÜLKELER DEĞİL, ÜLKE DIŞI YAPILAR, MESELA MENA GİBİ BÖLGELER DE VAR!
# Başka isimlerle verilen ülke değişkeni ismini Country olarak standardize edelim:

cols_to_change = ["Entity", "Country Name"]
for name, df in df_names.items():
    matched_cols = [col for col in cols_to_change if col in df.columns]
    if matched_cols:
        print(f"Matched columns in {name}: {', '.join(matched_cols)}")
        rename_dict = {col: 'Country' for col in matched_cols}  # Dictionary to hold old and new column names
        df.rename(columns=rename_dict, inplace=True)  # Use rename_dict in rename method
        print(f"--> Columns renamed as 'Country' in {name}: {', '.join(matched_cols)}\n")
    else:
        print(f"No matched columns in {name}\n")

parliament.columns
maternal_mortality.columns


# Ülke ismi kısaltmalarını içeren değişkenleri silelim:
for name, df in df_names.items():
    cols_to_drop = ["Code", "Country Code", "ISO3", "No"]
    matched_cols = [col for col in cols_to_drop if col in df.columns]
    if matched_cols:
        print(f"Matched variables in {name}: {', '.join(matched_cols)}")
        for col in matched_cols:
            df.drop(columns=col, inplace=True)
        print(f"--> Dropped columns from {name}: {', '.join(matched_cols)}\n")
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


# Tablo okuma kolaylığı için uzun değişken isimlerini kısaltalım:
f_to_m_unpaid_care_work = f_to_m_unpaid_care_work.rename(columns={"Female to male ratio of time devoted to unpaid care work (OECD (2014))" : "F/M Unpaid Care Work Time"})
f_to_m_unpaid_care_work.info()

f_to_m_labor_force_part = f_to_m_labor_force_part.rename(columns={"Ratio of female to male labor force participation rate (%) (modeled ILO estimate)" : "F/M Labor Force Part"})
f_to_m_labor_force_part.info()

maternal_mortality = maternal_mortality.rename(columns={'Maternal Mortality Ratio (Gapminder (2010) and World Bank (2015))' : 'Maternal Mortality Ratio'})
maternal_mortality.info()


##############################################
# ANALİZ KISITLARI
##############################################
# Katılımcılardan analizlerini belirli kısıtlar çerçevesinde yapmaları beklenmektedir.
# Bu kısıtlar, belirli bir konuyla ilgili olabilir veya veri setinin belirli özellikleri üzerine odaklanmayı içerebilir.



##############################################
# VERİ ANALİZİ ÇALIŞMASI
##############################################
# Analizlerinizde çeşitli teknikler kullanarak veri setini keşfetmeniz ve bu çalışmayı yorumlamanız beklenmektedir.
# Bu teknikler arasında görselleştirme, istatistiksel analiz, kümeleme ve tahminleme gibi yöntemler bulunmaktadır.


# Farklı yılları temsil eden sütunları, satır yapalım: # TODO üstte "country name"="country" oldu, diğer üçü de uçtu. Onayınız olmadan buraya dokunmak istemedim, ne yapalım?
parliament = pd.melt(parliament, id_vars=['Country'], var_name='Year', value_name='Women Seat Ratio')
# Kontrol edelim:
parliament.tail()
parliament.info()

# TODO bunu çalıştıramadım
parliament["Year"] = parliament["Year"].astype("int")



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
male_labor_force.drop(["HDI Rank (2021)", "Labour Force Participation Rate", "Continent", "Hemisphere"], axis=1, inplace=True)


# Farklı yılları temsil eden sütunları, satır yapalım:
female_labor_force = pd.melt(female_labor_force, id_vars=['ISO3', 'Country', 'Continent', 'Hemisphere', 'HDI Rank (2021)'], var_name='Labour Force Participation Rate', value_name='Female Labour Force Participation Rate')
female_labor_force.head()


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
female_labor_force.drop(["HDI Rank (2021)", "Labour Force Participation Rate", "Continent", "Hemisphere"], axis=1, inplace=True)

# TODO male ve female labor_force verisetlerini aynı işlemlerden geçiriyoruz. Daha şık olması için bu ikisini liste + fonksiyon + for_döngüsü ile birleştirelim.

# Şimdi birleştirelim:

# Cinsiyetlerin iş gücüne katılımını gösteren iki ayrı verisetini birleştirerek yeni veriseti oluşturalım:
merged_labor_force = pd.merge(female_labor_force, male_labor_force, on=["Country"])
df_names.update({"merged_labor_force": merged_labor_force})
"merged_labor_force" in df_names # True


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
# TODO Merge kullanacaksak for döngüsü ile yaparsak daha şık olur.
# TODO Tüm bunların ülke kolonlarını tek liste yap, alfabetik sırala, temizle.
merged_df = pd.merge(gender_wage_gap, parliament, on=['Country', "Year"])
merged_df = pd.merge(merged_df, maternal_mortality, on=['Country', "Year"])
merged_df = pd.merge(merged_df, male_labor_force, on=['Country', "Year"])
merged_df = pd.merge(merged_df, female_labor_force, on=['Country', "Year"])
merged_df = pd.merge(merged_df, f_to_m_labor_force_part, on=['Country', "Year"])
merged_df["Year"].describe().T
merged_df.shape # 301, 8

parliament.Country.values()

######## COMPLETE COUNTRY LIST

countries = []
for name, df in df_names.items():
    if "Country" in df.columns:
        country_of_df = df["Country"].tolist()
        countries.extend(country_of_df)

countries_unique = sorted(list(set(countries)))

regions = ['Africa Eastern and Southern',
           'Africa Western and Central',
           'Arab World',
           'Caribbean small states',
           'Central Europe and the Baltics',
           'Early-demographic dividend',
           'East Asia & Pacific',
           'East Asia & Pacific (IDA & IBRD countries)',
           'East Asia & Pacific (WB)',
           'East Asia & Pacific (excluding high income)',
           'East Asia and Pacific (WB)',
           'Euro area',
           'Europe & Central Asia',
           'Europe & Central Asia (IDA & IBRD countries)',
           'Europe & Central Asia (WB)',
           'Europe & Central Asia (excluding high income)',
           'Europe and Central Asia (WB)',
           'European Union',
           'European Union (27)',
           'Fragile and conflict affected situations',
           'Heavily indebted poor countries (HIPC)',
           'High income',
           'High income (WB)',
           'High-income countries',
           'IBRD only',
           'IDA & IBRD total',
           'IDA blend',
           'IDA only',
           'IDA total',
           'Late-demographic dividend',
           'Latin America & Caribbean',
           'Latin America & Caribbean (WB)',
           'Latin America & Caribbean (excluding high income)',
           'Latin America & the Caribbean (IDA & IBRD countries)',
           'Latin America and Caribbean (WB)',
           'Least developed countries: UN classification',
           'Low & middle income',
           'Low & middle income (WB)',
           'Low income',
           'Low income (WB)',
           'Low-income countries',
           'Lower middle income',
           'Lower middle income (WB)',
           'Lower-middle-income countries',
           'Middle East & North Africa',
           'Middle East & North Africa (IDA & IBRD countries)',
           'Middle East & North Africa (WB)',
           'Middle East & North Africa (excluding high income)',
           'Middle East and North Africa (WB)',
           'Middle income',
           'Middle income (WB)',
           'Middle-income countries',
           'North America',
           'North America (WB)',
           'Not classified',
           'OECD members',
           'Other small states',
           'Pacific island small states',
           'Post-demographic dividend',
           'Pre-demographic dividend',
           'Small states',
           'South Africa',
           'South Asia',
           'South Asia (IDA & IBRD)',
           'South Asia (WB)',
           'Sub-Saharan Africa',
           'Sub-Saharan Africa (IDA & IBRD countries)',
           'Sub-Saharan Africa (WB)',
           'Sub-Saharan Africa (excluding high income)',
           'Upper middle income',
           'Upper middle income (WB)',
           'Upper-middle-income countries',
           'World']

# TODO South Sudan, Kosovo vb. yeni devletleri dahil edip etmeme konusunu düşünmemiz lazım.
repetitions = [['American Samoa', 'Samoa'],
               ['Bahamas', 'Bahamas, The'],
               ['Brunei', 'Brunei Darussalam'],
               ['Congo', 'Congo, Dem. Rep.', 'Congo, Rep.', 'Democratic Republic of Congo', 'The Democratic Republic of the Congo'], # TODO Congo derken?
               ["Cote d'Ivoire", 'Ivory Coast'],
               ['Egypt', 'Egypt, Arab Rep.'],
               ['East Timor', 'Timor-Leste'],
               ['Gambia', 'Gambia, The'],
               ['Hong Kong', 'Hong Kong SAR, China'],
               ['Iran', 'Iran, Islamic Rep.'],
               ['Korea', "Korea, Dem. People's Rep.", 'Korea, Rep.', 'North Korea', 'South Korea',],
               ['Kyrgyz Republic', 'Kyrgyzstan'],
               ['Lao', 'Lao PDR', 'Laos'],
               ['Macao', 'Macao SAR, China'],
               ['Macedonia', 'North Macedonia'],
               ['Micronesia', 'Micronesia (country)', 'Micronesia, Fed. Sts.'], # TODO Micronesia bölge de olabilir, verisetine bir bakmam lazım.
               ['Palestine','Palestine, State of', 'West Bank and Gaza'],
               ['Puerto Rico'], # TODO ABD'ye bağlı, hem ABD hem PR sorun olur mu bilmiyorum, belki çıkarırız.
               ['Russia', 'Russian Federation'],
               ['Saint Kitts and Nevis', 'St. Kitts and Nevis'],
               ['Saint Lucia', 'St. Lucia'],
               ['Saint Vincent and the Grenadines', 'St. Vincent and the Grenadines'],
               ['Slovak Republic', 'Slovakia'],
               ['Sudan', 'South Sudan'],
               ['Syria', 'Syrian Arab Republic'],
               ['Turkey', 'Turkiye'],
               ['UK', 'United Kingdom'],
               ['Venezuela', 'Venezuela, RB'],
               ['Viet Nam', 'Vietnam'],
               ['United States Virgin Islands', 'Virgin Islands (U.S.)'],
               ['Yemen', 'Yemen, Rep.']]

########

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

