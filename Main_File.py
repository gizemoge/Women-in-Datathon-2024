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
            "adolescent_fertility_rate": adolescent_fertility_rate}


##############################################
# VERİ SETİ İNCELEME VE AÇIKLAMASI
##############################################
# Size sağlanacak olan veri seti, belirli bir konuya ait gerçek verilerden oluşmaktadır.
# Bu veri seti üzerinde analiz yaparken, veri setlerini detaylıca incelemeniz, anlamanız ve anlamlı bir şekilde yorumlayabilmelisiniz.
# Hatırlatma: Dış kaynaklardan ekstra veriler bularak veri analizi çalışmanızı zenginleştirebilirsiniz.

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col, "Entity"].describe(quantiles).T)

# 1- female-to-male-ratio-of-time-devoted-to-unpaid-care-work
'''(Rapordan) Note: Gender inequality in unpaid care work refers to the female to
male ratio of time spent in unpaid care work. The fitted value of the
female share in the active population is estimated by controlling for
the country’s GDP per capita, fertility rate, urbanisation rate, maternity
leave policies and gender inequality in unemployment and education.'''
f_to_m_unpaid_care_work.head()
f_to_m_unpaid_care_work.shape # (69, 4)
f_to_m_unpaid_care_work.info() # veritiplerinde sıkıntı yok.
f_to_m_unpaid_care_work["Code"].nunique()
f_to_m_unpaid_care_work.loc[f_to_m_unpaid_care_work["Female to male ratio of time devoted to unpaid care work (OECD (2014))"]==1.180]
f_to_m_unpaid_care_work.loc[f_to_m_unpaid_care_work["Female to male ratio of time devoted to unpaid care work (OECD (2014))"]==17.290]
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
w_in_top_income_groups.Year.nunique()
w_in_top_income_groups["Share of women in top 0.1%"].describe()
w_in_top_income_groups.loc[w_in_top_income_groups["Share of women in top 0.1%"]==4.600] # min: Denmark
w_in_top_income_groups.loc[w_in_top_income_groups["Share of women in top 0.1%"]==20.000] # max: Spain
w_in_top_income_groups["Share of women in top 10%"].describe()
w_in_top_income_groups.loc[w_in_top_income_groups["Share of women in top 10%"]==9.400] # min: Denmark
w_in_top_income_groups.loc[w_in_top_income_groups["Share of women in top 10%"]==34.800] # max: Spain
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
w_in_top_income_groups.loc[w_in_top_income_groups["Share of women in top 0.1%"]==20]
w_in_top_income_groups["Share of women in top 0.1%"].describe()

# 3- ratio-of-female-to-male-labor-force-participation-rates-ilo-wdi
f_to_m_labor_force_part.head()
f_to_m_labor_force_part.shape # (6432, 4)
f_to_m_labor_force_part.info() # veritiplerinde sıkıntı yok.
f_to_m_labor_force_part.Year.nunique()
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
f_to_m_labor_force_part["Ratio of female to male labor force participation rate (%) (modeled ILO estimate)"].describe()
f_to_m_labor_force_part.loc[f_to_m_labor_force_part["Ratio of female to male labor force participation rate (%) (modeled ILO estimate)"] == 83.292]

#4 (1 ile aynı olduğu için silindi)


# 5- maternal-mortality
# The maternal mortality ratio is the number of women who die from pregnancy-related causes while pregnant or within 42 days of pregnancy termination per 100,000 live births.
maternal_mortality.head()
maternal_mortality.shape # (5800, 4)
maternal_mortality.info() # veritiplerinde sıkıntı yok.
maternal_mortality.Year.nunique()

quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
maternal_mortality["Maternal Mortality Ratio (Gapminder (2010) and World Bank (2015))"].describe(quantiles).T
maternal_mortality.loc[maternal_mortality["Maternal Mortality Ratio (Gapminder (2010) and World Bank (2015))"] == 1180]
maternal_mortality.loc[maternal_mortality["Entity"] == "Sierra Leone"]

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
gender_wage_gap.head(30)
gender_wage_gap.shape # (413, 4)
gender_wage_gap.info() # veritiplerinde sıkıntı yok.
gender_wage_gap.Year.nunique()
gender_wage_gap.Year.value_counts()
gender_wage_gap.loc[gender_wage_gap["Gender wage gap (%)"] == -28.270]
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
w_entrepreneurship["No"].nunique()
w_entrepreneurship["Women Entrepreneurship Index"].describe()
w_entrepreneurship.loc[w_entrepreneurship["Women Entrepreneurship Index"]==25.300]
w_entrepreneurship.loc[w_entrepreneurship["Female Labor Force Participation Rate"]==13.000]
w_entrepreneurship.loc[w_entrepreneurship["Country"] == "Belgium"]
female_labor_force.loc[female_labor_force["Country"] == "Belgium"]


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
male_labor_force["Continent"].nunique()
male_labor_force["Continent"].value_counts()
male_labor_force["Hemisphere"].nunique()
male_labor_force["Hemisphere"].value_counts()

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
placement["hsc_board"].info()
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

gender_wage_gap.columns # TODO target

# Farklı yılları temsil eden sütunları, satır yapalım:
parliament = pd.melt(parliament, id_vars=['Country'], var_name='Year', value_name='Women Seat Ratio')
# Kontrol edelim:
parliament.tail()
parliament.info()
# Year değişkenini biz string'den yarattığımız için tipini int yapmalıyız:
parliament["Year"] = parliament["Year"].astype("int")

# Farklı yılları temsil eden sütunları, satır yapalım:
adolescent_fertility_rate = pd.melt(adolescent_fertility_rate, id_vars=['Country'], var_name='Year', value_name='Adolescent fertility rate')
adolescent_fertility_rate.head()
adolescent_fertility_rate["Year"] = adolescent_fertility_rate["Year"].astype("int")


# Farklı yılları temsil eden sütunları, satır yapalım:
male_labor_force = pd.melt(male_labor_force, id_vars=['Country', 'Continent', 'Hemisphere', 'HDI Rank (2021)'], var_name='Labour Force Participation Rate', value_name='Male Labour Force Participation Rate')
male_labor_force.head()

# Eski sütun ismi içerisindeki yıl bilgisini kaldırarak yeni isim oluşturalım:

m_val = []
for val in male_labor_force["Labour Force Participation Rate"].values:
    new_name = val.split(" ")[-1].replace("(", "").replace(")", "")
    m_val.append(new_name)

male_labor_force['Year'] = male_labor_force['Labour Force Participation Rate'].replace(male_labor_force['Labour Force Participation Rate'].tolist(), m_val)
male_labor_force["Year"] = male_labor_force["Year"].astype("int")

# Verisetini şu anki gibi yıla göre değil, ülkelere göre alfabetik sıralayalım:
male_labor_force.sort_values(by='Country', inplace=True)
male_labor_force.head()

# Fazla değişkenleri silelim:
male_labor_force.drop(["HDI Rank (2021)", "Labour Force Participation Rate", "Continent", "Hemisphere"], axis=1, inplace=True)

# Farklı yılları temsil eden sütunları, satır yapalım:
female_labor_force = pd.melt(female_labor_force, id_vars=['Country', 'Continent', 'Hemisphere', 'HDI Rank (2021)'], var_name='Labour Force Participation Rate', value_name='Female Labour Force Participation Rate')
female_labor_force.head()
female_labor_force.tail()

# Eski sütun ismi içerisindeki yıl bilgisini kaldırarak yeni isim oluşturalım:
f_val = []
for val in female_labor_force["Labour Force Participation Rate"].values:
    new_name = val.split(" ")[-1].replace("(", "").replace(")", "")
    f_val.append(new_name)

female_labor_force['Year'] = female_labor_force['Labour Force Participation Rate'].replace(female_labor_force['Labour Force Participation Rate'].tolist(), f_val)
female_labor_force["Year"] = female_labor_force["Year"].astype("int")

# Verisetini şu anki gibi yıla göre değil, ülkelere göre alfabetik sıralayalım:
female_labor_force.sort_values(by='Country', inplace=True)
female_labor_force.head()

# Fazla değişkenleri silelim:
female_labor_force.drop(["HDI Rank (2021)", "Labour Force Participation Rate", "Continent", "Hemisphere"], axis=1, inplace=True)



########################
# TODO male ve female labor_force verisetlerini aynı işlemlerden geçiriyoruz. Daha şık olması için bu ikisini liste + fonksiyon + for_döngüsü ile birleştirelim.
# Şimdi birleştirelim:
# Cinsiyetlerin iş gücüne katılımını gösteren iki ayrı verisetini birleştirerek yeni veriseti oluşturalım:
#merged_labor_force = pd.merge(female_labor_force, male_labor_force, on=["Country"])
#df_names.update({"merged_labor_force": merged_labor_force})
#"merged_labor_force" in df_names # True


# COUNTRY HESAPLARI

# TODO Biz burada bir sıkıntı var sandık (merge, aldığı iki verisetinde de ortak olmayanları sildiği için, merge sıralamasının veri kaybetmemek içni önemli olduğunu düşündük) ama her halükarda en küçükle en büyük veriseti çarpışacağı için aslında önemli değil.
countries = []
for name, df in df_names.items():
    if "Country" in df.columns:
        country_vals_list = df["Country"].tolist()
        countries.extend(country_vals_list)

# Her bir ülkeye ait kaç gözlemimiz olduğuna bakalım:
for country in set(countries):
    tekrar_sayisi = countries.count(country)
    print(f"{country}: {tekrar_sayisi}")

# Unique ülkeleri alıp alfabetik sıralayalım:
countries_unique = sorted(list(set(countries)))

# Bu setin içerisinde aslında ülke olmayan bölge isimlerini ayıralım (NaN doldurmak için kullanabiliriz):
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
# Şimdi de yazım farklılıklarını ayıralım:
diffs = {'Samoa': 'American Samoa',
         'Bahamas, The': 'Bahamas',
         'Brunei Darussalam': 'Brunei',
         "Cote d'Ivoire": 'Ivory Coast',
         'Egypt, Arab Rep.': 'Egypt',
         'East Timor': 'Timor-Leste',
         'Gambia, The': 'Gambia',
         'Hong Kong SAR, China': 'Hong Kong',
         'Iran, Islamic Rep.': 'Iran',
         'Kyrgyz Republic': 'Kyrgyzstan',
         'Macao SAR, China': 'Macao',
         'Macedonia': 'North Macedonia',
         # ['Puerto Rico'], # TODO ABD'ye bağlı, hem ABD hem PR sorun olur mu bilmiyorum, belki çıkarırız.
         'Russian Federation': 'Russia',
         'Saint Kitts and Nevis': 'St. Kitts and Nevis',
         'Saint Lucia': 'St. Lucia',
         'Saint Vincent and the Grenadines': 'St. Vincent and the Grenadines',
         'Slovak Republic': 'Slovakia',
         # ['Sudan', 'South Sudan'],
         'Syrian Arab Republic': 'Syria',
         'Turkey': 'Turkiye',
         'UK': 'United Kingdom',
         'Venezuela, RB': 'Venezuela',
         'Viet Nam': 'Vietnam',
         'United States Virgin Islands': 'Virgin Islands (U.S.)',
         'Yemen, Rep.': 'Yemen',
         'Palestine, State of': 'Palestine',
         'West Bank and Gaza': 'Palestine'}

# Farklı yazımlara sahip benzer isimleri olan ülkeleri ayıralım sadeleştirme için kaynak verisetlerinden kontrol edebilelim:
confusions = ['Congo', 'Congo, Dem. Rep.', 'Congo, Rep.', 'Democratic Republic of Congo','The Democratic Republic of the Congo',  # TODO Congo derken?
              'Korea', "Korea, Dem. People's Rep.", 'Korea, Rep.', 'North Korea', 'South Korea',
              'Lao', 'Lao PDR', 'Laos',
              'Micronesia', 'Micronesia (country)', 'Micronesia, Fed. Sts.']
len(confusions) # 19

"""
# Bu karışık yazımların hangi verisetlerinden geldiğini bulalım:
for confused_name in confusions:
    print(f"'{confused_name}':")
    df_with_confusion = []
    for name, df in df_names.items():
        if "Country" in df.columns:
            if confused_name in df["Country"].values:
                df_with_confusion.append(name)
            #else:
                #print(f"- not in: '{name}'")
        #else:
            # print(f"No 'Country' column applicable in: '{name}'")
    print(df_with_confusion)
    print("\n")
"""
# Hangi verisetinde hangi karışık yazımlar var bakalım ve her veriseti için kaydedelim:
confusion_df_dict = {}
for name, df in df_names.items():
    print(f"'{name}':")
    confusions_in_df = []
    if "Country" in df.columns:
        for confused_name in confusions:
            if confused_name in df["Country"].values:
                confusions_in_df.append(confused_name)
        print(confusions_in_df)
        confusion_df_dict.update({ name : confusions_in_df})
        #globals()[f"confusions_in_{name}"] = confusions_in_df
    else:
        print("[]")
    print("\n")

confusion_df_dict.keys()
df_names

# Temizleyelim.
confusion_sorted = {'Congo': 'Congo, Rep.',
                   'Democratic Republic of Congo': 'Congo, Dem. Rep.',
                   'The Democratic Republic of the Congo': 'Congo, Dem. Rep.',
                   'Korea': 'South Korea',
                   'Korea, Rep.': 'South Korea',
                   "Korea, Dem. People's Rep.": 'North Korea',
                   'Lao': 'Laos',
                   'Lao PDR': 'Laos',
                   'Micronesia (country)': 'Micronesia',
                   'Micronesia, Fed. Sts.': 'Micronesia'}

for df in list(df_names.values()):
     if "Country" in df.columns:
         df["Country"] = df["Country"].replace(confusion_sorted)
         df["Country"] = df["Country"].replace(diffs)

f_to_m_unpaid_care_work[f_to_m_unpaid_care_work["Country"]=="Korea"]

# FOR ÇALIŞMAYINCA PES EDİP MANUEL YAZDIK
f_to_m_unpaid_care_work.head()
f_to_m_unpaid_care_work.info()
f_to_m_unpaid_care_work["Country"] = f_to_m_unpaid_care_work["Country"].replace(confusion_sorted)
f_to_m_unpaid_care_work["Country"] = f_to_m_unpaid_care_work["Country"].replace(diffs)

w_in_top_income_groups["Country"] = w_in_top_income_groups["Country"].replace(confusion_sorted)
w_in_top_income_groups["Country"] = w_in_top_income_groups["Country"].replace(diffs)

f_to_m_labor_force_part["Country"] = f_to_m_labor_force_part["Country"].replace(confusion_sorted)
f_to_m_labor_force_part["Country"] = f_to_m_labor_force_part["Country"].replace(diffs)

maternal_mortality["Country"] = maternal_mortality["Country"].replace(confusion_sorted)
maternal_mortality["Country"] = maternal_mortality["Country"].replace(diffs)

gender_wage_gap["Country"] = gender_wage_gap["Country"].replace(confusion_sorted)
gender_wage_gap["Country"] = gender_wage_gap["Country"].replace(diffs)

w_entrepreneurship["Country"] = w_entrepreneurship["Country"].replace(confusion_sorted)
w_entrepreneurship["Country"] = w_entrepreneurship["Country"].replace(diffs)

male_labor_force["Country"] = male_labor_force["Country"].replace(confusion_sorted)
male_labor_force["Country"] = male_labor_force["Country"].replace(diffs)

female_labor_force["Country"] = female_labor_force["Country"].replace(confusion_sorted)
female_labor_force["Country"] = female_labor_force["Country"].replace(diffs)

parliament["Country"] = parliament["Country"].replace(confusion_sorted)
parliament["Country"] = parliament["Country"].replace(diffs)

adolescent_fertility_rate["Country"] = adolescent_fertility_rate["Country"].replace(confusion_sorted)
adolescent_fertility_rate["Country"] = adolescent_fertility_rate["Country"].replace(diffs)


# Kontrol edelim:
sorted(parliament["Country"].unique())


# MERGE VAKTİ
merged_df = pd.merge(gender_wage_gap, parliament, on=['Country', "Year"])
merged_df = pd.merge(merged_df, maternal_mortality, on=['Country', "Year"])
merged_df = pd.merge(merged_df, male_labor_force, on=['Country', "Year"])
merged_df = pd.merge(merged_df, female_labor_force, on=['Country', "Year"])
merged_df = pd.merge(merged_df, f_to_m_labor_force_part, on=['Country', "Year"])
merged_df = pd.merge(merged_df, adolescent_fertility_rate, on=['Country', "Year"])
"""
# gdi için gerder_wage_gap olmadan birleştirme
merged_df_gdi = pd.merge(parliament, maternal_mortality, on=['Country', "Year"])
merged_df_gdi = pd.merge(merged_df_gdi, male_labor_force, on=['Country', "Year"])
merged_df_gdi = pd.merge(merged_df_gdi, female_labor_force, on=['Country', "Year"])
merged_df_gdi = pd.merge(merged_df_gdi, f_to_m_labor_force_part, on=['Country', "Year"])
merged_df_gdi = pd.merge(merged_df_gdi, adolescent_fertility_rate, on=['Country', "Year"])
# bunu excel'e alıyorum
merged_df_gdi.to_excel("output.xlsx", index=False)

unique_countries = merged_df_gdi["Country"].unique()
unique_country_count = len(unique_countries)
# 178 adet ülke var unique
"""

# C. Merge'den önce yılları gruplayalım:

merged_df_year_group = pd.DataFrame()
for name, df in df_names.items():
    if "Year" in df.columns:
        print(f"{name}")
        df["Year_group"] = pd.cut(df["Year"], [1995, 2002, 2009, 2016])
        print(f"{name}: {df['Year_group'].value_counts()}")
        df = df.groupby(["Country", "Year_group"]).mean().reset_index()
        cleaned_df = df.dropna(subset=df.columns.difference(["Country", "Year_group"]), how="all").drop("Year", axis=1, inplace=True)
        countries_with_three_year_groups = cleaned_df["Country"].value_counts()[cleaned_df["Country"].value_counts() == 3].index.tolist()
        print(len(countries_with_three_year_groups))
        if merged_df_year_group.empty:
            merged_df_year_group = df
        else:
            merged_df_year_group = pd.merge(merged_df_year_group, df, on=['Country', "Year_group"])

merged_df_year_group.head()
merged_df_year_group.shape

gender_wage_gap.head()
f_to_m_labor_force_part.head()
f_to_m_unpaid_care_work.head()
maternal_mortality.head()
w_in_top_income_groups.head()

maternal_mortality["Year_group"] = pd.cut(maternal_mortality["Year"], [1995, 2002, 2009, 2016])
maternal_mortality_grouped = maternal_mortality.groupby(["Country", "Year_group"]).mean().reset_index()
cleaned_df = grouped_merged_df.dropna(subset=grouped_merged_df.columns.difference(["Country", "Year_group"]), how="all")
countries_with_three_year_groups = cleaned_df["Country"].value_counts()[cleaned_df["Country"].value_counts() == 3].index.tolist()
len(countries_with_three_year_groups) # 24

gender_wage_gap["Year_group"] = pd.cut(gender_wage_gap["Year"], [1995, 2002, 2009, 2016])
gender_wage_gap_grouped = gender_wage_gap.groupby(["Country", "Year_group"]).mean().reset_index()
cleaned_df = grouped_merged_df.dropna(subset=grouped_merged_df.columns.difference(["Country", "Year_group"]), how="all")
countries_with_three_year_groups = cleaned_df["Country"].value_counts()[cleaned_df["Country"].value_counts() == 3].index.tolist()
len(countries_with_three_year_groups) # 24

f_to_m_labor_force_part["Year_group"] = pd.cut(f_to_m_labor_force_part["Year"], [1995, 2002, 2009, 2016])
f_to_m_labor_force_part_grouped = f_to_m_labor_force_part.groupby(["Country", "Year_group"]).mean().reset_index()
cleaned_df = grouped_merged_df.dropna(subset=grouped_merged_df.columns.difference(["Country", "Year_group"]), how="all")
countries_with_three_year_groups = cleaned_df["Country"].value_counts()[cleaned_df["Country"].value_counts() == 3].index.tolist()
len(countries_with_three_year_groups) # 24

f_to_m_unpaid_care_work["Year_group"] = pd.cut(f_to_m_unpaid_care_work["Year"], [1995, 2002, 2009, 2016])
f_to_m_unpaid_care_work_grouped = f_to_m_unpaid_care_work.groupby(["Country", "Year_group"]).mean().reset_index()
cleaned_df = grouped_merged_df.dropna(subset=grouped_merged_df.columns.difference(["Country", "Year_group"]), how="all")
countries_with_three_year_groups = cleaned_df["Country"].value_counts()[cleaned_df["Country"].value_counts() == 3].index.tolist()
len(countries_with_three_year_groups) # 24

w_in_top_income_groups["Year_group"] = pd.cut(w_in_top_income_groups["Year"], [1995, 2002, 2009, 2016])
w_in_top_income_groups_grouped = w_in_top_income_groups.groupby(["Country", "Year_group"]).mean().reset_index()
cleaned_df = grouped_merged_df.dropna(subset=grouped_merged_df.columns.difference(["Country", "Year_group"]), how="all")
countries_with_three_year_groups = cleaned_df["Country"].value_counts()[cleaned_df["Country"].value_counts() == 3].index.tolist()
len(countries_with_three_year_groups) # 24

merged_df = pd.merge(gender_wage_gap, parliament, on=['Country', "Year"])
merged_df = pd.merge(merged_df, maternal_mortality, on=['Country', "Year"])
merged_df = pd.merge(merged_df, male_labor_force, on=['Country', "Year"])
merged_df = pd.merge(merged_df, female_labor_force, on=['Country', "Year"])
merged_df = pd.merge(merged_df, f_to_m_labor_force_part, on=['Country', "Year"])
merged_df = pd.merge(merged_df, adolescent_fertility_rate, on=['Country', "Year"])





# B.1. Merge'de yılları gruplayalım:
merged_df["Year_group"] = pd.cut(merged_df["Year"], [1995, 2002, 2009, 2016])
merged_df["Year_group"].value_counts()
# (2010, 2015]    117
# (2005, 2010]     99
# (2000, 2005]     78
# 24 ülke oluyor

# (2011, 2016]    111
# (2001, 2006]     89
# (2006, 2011]     89
# (1996, 2001]     29
# 12 ülke oluyor

# (2009, 2016]    155
# (2002, 2009]    109
# (1995, 2002]     56
# 25 ülke oluyor
merged_df.head(20)

grouped_merged_df = merged_df.groupby(["Country", "Year_group"]).mean().reset_index()
merged_df[merged_df["Country"]==("Poland")]

cleaned_df = grouped_merged_df.dropna(subset=grouped_merged_df.columns.difference(["Country", "Year_group"]), how="all")
countries_with_three_year_groups = cleaned_df["Country"].value_counts()[cleaned_df["Country"].value_counts() == 3].index.tolist()
len(countries_with_three_year_groups) # 24


####
# A.1. Concat
dfs_to_concat = [gender_wage_gap, parliament, maternal_mortality, male_labor_force, female_labor_force, f_to_m_labor_force_part, adolescent_fertility_rate]
concat_df = pd.concat(dfs_to_concat, ignore_index=True)
concat_df.shape # (58641, 9)
concat_df.head()

# A.2. Yılları grupla.
concat_df["Year_group"] = pd.cut(concat_df["Year"], [1995, 2002, 2009, 2016])
concat_df["Year_group"].value_counts()

grouped_concat_df = concat_df.groupby(["Country", "Year_group"]).mean().reset_index() # (867, 10)
grouped_concat_df.head(20)
grouped_concat_df.shape
grouped_concat_df = grouped_concat_df[~grouped_concat_df["Country"].isin(regions)] # (648, 10)

# A.3. Dropna
# A.3.1. Ülke ve yıl grubu dışındakilerin tamamı boşsa satırı sil: (böyle bir satır yokmuş)
clean_grouped_concat_df = grouped_concat_df.dropna(subset=grouped_concat_df.columns.difference(["Country", "Year_group"]), how="all")
clean_grouped_concat_df.shape # (648, 10)
clean_grouped_concat_df.head(20)

concat_countries_with_three_year_groups = clean_grouped_concat_df["Country"].value_counts()[clean_grouped_concat_df["Country"].value_counts() == 3].index.tolist()
len(concat_countries_with_three_year_groups) # 216

# A.3.2. Üstte bir etki yaratamadığımız için başka bir yol deneyelim. Gender wage gap'i NaN olanları droplayalım.
clean_grouped_concat_df = grouped_concat_df.dropna(subset=["Gender wage gap (%)"])
clean_grouped_concat_df.shape # (119, 10)

# A.3.2.2. 3 yıl grubuna da sahip hangi ülkeler kaldı?
concat_countries_with_three_year_groups = clean_grouped_concat_df["Country"].value_counts()[clean_grouped_concat_df["Country"].value_counts() == 3].index.tolist()
len(concat_countries_with_three_year_groups) # 24
clean_grouped_concat_df["Country"].value_counts()


# B.
# Yılları grupla, sonra mergele
##########




#
merged_df["Year"].describe().T # 1990-2016
merged_df.shape # 324, 9

merged_df.head()

merged_df.isnull().sum() # Women Seat Ratio'da 14 adet null değer var
merged_df.loc[merged_df["Women Seat Ratio"].isnull()]
merged_df.loc[merged_df["Country"] == "Chile"]
merged_df.loc[merged_df["Country"] == "Mexico"]
merged_df.loc[merged_df["Country"] == "Argentina"]

merged_df[merged_df["Country"] == "Chile"].fillna(merged_df["Country"] == ["Chile"].min())

merged_df['Women Seat Ratio'] = merged_df['Women Seat Ratio'].fillna(merged_df.groupby('Country')['Women Seat Ratio'].transform('min'))


parliament[parliament["Country"] == "Chile"]

[col for col in df.columns if merged_df.loc[merged_df["Women Seat Ratio"].isnull()]]

# Outer merge edip (ki bu da concat demekmiş) bakmak istiyoruz:
merged_df_2 = pd.merge(gender_wage_gap, parliament, on=['Country', "Year"], how='outer')
merged_df_2 = pd.merge(merged_df_2, maternal_mortality, on=['Country', "Year"], how='outer')
merged_df_2 = pd.merge(merged_df_2, male_labor_force, on=['Country', "Year"], how='outer')
merged_df_2 = pd.merge(merged_df_2, female_labor_force, on=['Country', "Year"],  how='outer')
merged_df_2 = pd.merge(merged_df_2, f_to_m_labor_force_part, on=['Country', "Year"],  how='outer')
merged_df_2 = pd.merge(merged_df_2, adolescent_fertility_rate, on=['Country', "Year"],  how='outer')

merged_df_2["Year"].describe().T # 1751-2022
merged_df_2.shape # (41883, 8)
merged_df_2.head()

# Target'ımız senesine uysun diye left merge etmek istiyorum:
merged_df_3 = pd.merge(gender_wage_gap, parliament, on=['Country', "Year"], how='left')
merged_df_3 = pd.merge(merged_df_3, maternal_mortality, on=['Country', "Year"], how='left')
merged_df_3 = pd.merge(merged_df_3, male_labor_force, on=['Country', "Year"], how='left')
merged_df_3 = pd.merge(merged_df_3, female_labor_force, on=['Country', "Year"],  how='left')
merged_df_3 = pd.merge(merged_df_3, f_to_m_labor_force_part, on=['Country', "Year"],  how='left')
merged_df_3 = pd.merge(merged_df_3, adolescent_fertility_rate, on=['Country', "Year"],  how='left')

merged_df_3["Year"].describe().T # 1981-2016
merged_df_3.shape # (413, 9)
merged_df_3.head()
male_labor_force["Year"].describe().T # 1990-2021
parliament["Year"].describe().T
parliament.loc[parliament["Year"] < 1997]
parliament.head()


# Parliament'ı kırparak tekrar inner merge atalım
parliament = parliament.loc[parliament["Year"] > 1996]
parliament.head()
merged_df = pd.merge(gender_wage_gap, parliament, on=['Country', "Year"])
merged_df = pd.merge(merged_df, maternal_mortality, on=['Country', "Year"])
merged_df = pd.merge(merged_df, male_labor_force, on=['Country', "Year"])
merged_df = pd.merge(merged_df, female_labor_force, on=['Country', "Year"])
merged_df = pd.merge(merged_df, f_to_m_labor_force_part, on=['Country', "Year"])
merged_df = pd.merge(merged_df, adolescent_fertility_rate, on=['Country', "Year"])

merged_df["Year"].describe().T # 1998-2016
merged_df.shape # (295, 9)

sorted(male_labor_force["Year"].unique())
male_labor_force["Year"].nunique() # 32
adolescent_fertility_rate["Year"].value_counts()
merged_df.loc('Country', "Year")

grouped = merged_df.groupby(["Country","Year"]).apply(lambda x: x.reset_index(drop=True))

for name, df in df_names.items():
    df.groupby(["Country", "Year"]).apply(lambda x: x.reset_index(drop=True))



merged_df.columns
merged_df_3.loc[merged_df_3["Country"] == "United Kingdom"]
gender_wage_gap["Entity"].value_counts()
gender_wage_gap["Year"].value_counts()
maternal_mortality["Year"].sort_values().value_counts()

merged_df["Year"].value_counts()

"""
years_array = np.arange(1970, 2024)
for year in years_array:
    for name, df in df_names.items():
        if "Year" in df.columns:
            if year not in df["Year"].values:
                # Create a new row with NaN values for other columns
                new_row = pd.DataFrame([[np.nan] * len(df.columns)], columns=df.columns)
                new_row['Year'] = year  # Set the year value
                # Append the new row to the DataFrame
                df = pd.concat([df, new_row], ignore_index=True)
                # Update the original DataFrame in df_names dictionary
                df_names[name] = df
"""
gender_wage_gap["Entity"].value_counts()


#Eda-------------------------------------------------------------------------
merged_df_copy = merged_df.copy()

merged_df_copy.head()

merged_df_2014 = merged_df_copy[merged_df_copy["Year"] == 2014]
merged_df_2014.reset_index()
merged_df_2014.isnull().sum()




# multiple linear regression
X = merged_df_2014.drop(['Gender wage gap (%)', "Country"], axis=1)
y = merged_df_2014[["Gender wage gap (%)"]]

##########################
# Model
##########################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

y_test.shape
y_train.shape

reg_model = LinearRegression().fit(X_train, y_train)

# sabit (b - bias)
reg_model.intercept_
# 111.41544466

# coefficients (w - weights)
reg_model.coef_
# ([[ 0.        , -0.10085855,  0.0755021 , -1.916704  ,  2.37494941,
#         -1.17009989, -0.05977133]])






merged_df_2014.columns


##########################
# Tahmin Başarısını Değerlendirme
##########################

# 1- hold out yöntemi:
# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
#5.757216090869661


# Train RKARE
reg_model.score(X_train, y_train)
# 0.5299800183820808

# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# 1.41
# test hatası normalde train hatasından daha yüksek çıkar

# Test RKARE
reg_model.score(X_test, y_test)
# 0.89


# 2- Cross validation yöntemi
# 10 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))
# 1.69


# 5 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))
# 1.71


plt.figure(figsize=(8, 6))
sns.heatmap(merged_df_2014[['Gender wage gap (%)', 'Women Seat Ratio', 'Maternal Mortality Ratio', 'Male Labour Force Participation Rate',
       'Female Labour Force Participation Rate', 'F/M Labor Force Part', 'Adolescent fertility rate']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Korelasyon Matrisi')
plt.show()

merged_df_2014.head()


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    print(f"{col_name} sütunu için alt ve üst limit değerleri: ({low_limit}, {up_limit})")
    return low_limit, up_limit

num_cols = [col for col in merged_df_2014.columns if col not in ["Country", "Year"]]
for col in num_cols:
    outlier_thresholds(merged_df_2014, col)

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

for col in num_cols:
    grab_outliers(merged_df_2014, col)
# outlier tespit edilmedi


# standartlaştırma
rs = RobustScaler()
merged_df_2014[num_cols] = rs.fit_transform(merged_df_2014[num_cols])

merged_df_2014.head()

#tekrar model
X = merged_df_2014.drop(['Gender wage gap (%)', "Country"], axis=1)
y = merged_df_2014[["Gender wage gap (%)"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
reg_model = LinearRegression().fit(X_train, y_train)

# sabit (b - bias)
reg_model.intercept_
# 111.41544466

# coefficients (w - weights)
reg_model.coef_
# ([[ 0.        , -0.10085855,  0.0755021 , -1.916704  ,  2.37494941,
#         -1.17009989, -0.05977133]])
##########################
# Tahmin Başarısını Değerlendirme
##########################

# 1- hold out yöntemi:
# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
#5.757216090869661    # 0.5865732135374082


# Train RKARE
reg_model.score(X_train, y_train)
# 0.5299800183820808   #0.5299800183820806

# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# 1.41                                  # 0.7217801148772511
# test hatası normalde train hatasından daha yüksek çıkar

# Test RKARE
reg_model.score(X_test, y_test)
# 0.89                      #-0.36022491213410235


#random forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
mse
# 0.4428928913486209

model.score(X_test, y_test)
# -0.13

# accuracy'lere de bakmak gerek

# gizem -----------------------------------
merged_df_2014 = merged_df_2014.drop(49)

merged_df_2014.isnull().sum()

merged_df_2014.shape # (50, 9)

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

X = merged_df_2014.drop(['Gender wage gap (%)', "Country", "Year"], axis=1)
y = merged_df_2014[["Gender wage gap (%)"]]

# Veri setini eğitim ve test kümelerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVR modelini oluştur ve eğit
svr_model = SVR(kernel='linear')  # Lineer çekirdek kullanarak bir SVR modeli oluştur
svr_model.fit(X_train, y_train)   # Modeli eğit

# Test veri kümesi üzerinde tahmin yap
y_pred = svr_model.predict(X_test)

# Model performansını değerlendir
mse = mean_squared_error(y_test, y_pred)
print("SVR ortalama karesel hata (MSE):", mse)
#38.80809525628783

svr_model.score(X_test, y_test)
# 0.4321485835554504

r2 = r2_score(y_test, y_pred)

# accuracy

# Grafik çizimi
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Hata', s=100, alpha=0.5)
plt.axhline(0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Gerçek Değerler', fontsize=14)
plt.ylabel('Hatalar (Gerçek - Tahmin)', fontsize=14)
plt.title('MSE: {:.2f}'.format(mse), fontsize=16)
plt.legend()
plt.grid(True)
plt.show()



# Grafik çizimi
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', label='Gerçek vs Tahmin')
plt.plot(y_test, y_pred, color='red', linestyle='--', label='Doğru Tahmin')
plt.title(f'SVR Modeli: Gerçek vs Tahmin (MSE={mse:.2f})')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Değerleri')
plt.legend()
plt.grid(True)
plt.show()




mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

merged_df


merged_df_2014.shape
# 50, 9

merged_df_2014 = pd.merge(merged_df_2014, f_to_m_unpaid_care_work, on=['Country', "Year"])

merged_df_2014["Country"]

country_array = merged_df_2014["Country"].to_numpy()
# veya
# country_array = merged_df_2014["Country"].values

print(country_array)


# ['Argentina' 'Austria' 'Belgium' 'Bolivia' 'Brazil' 'Bulgaria' 'Colombia'
#  'Cyprus' 'Czechia' 'Denmark' 'Dominican Republic' 'Ecuador' 'El Salvador'
#  'Estonia' 'Finland' 'France' 'Germany' 'Guatemala' 'Honduras' 'Hungary'
#  'Iceland' 'Ireland' 'Israel' 'Italy' 'Latvia' 'Lithuania' 'Luxembourg'
#  'Malaysia' 'Malta' 'Mexico' 'Montenegro' 'Netherlands' 'Nicaragua'
#  'North Macedonia' 'Norway' 'Panama' 'Paraguay' 'Peru' 'Poland' 'Romania'
#  'Serbia' 'Slovakia' 'Slovenia' 'South Korea' 'Spain' 'Sweden'
#  'Switzerland' 'Turkiye' 'United Kingdom' 'Uruguay']

merged_df_2014.columns
merged_df_2014["Gender wage gap (%)"]

# gender wage gap encoding
merged_df_2014['Gender wage gap (%)'] = merged_df_2014['Gender wage gap (%)'].apply(lambda x: 1 if x > 0 else 0)

#model
y = merged_df_2014["Gender wage gap (%)"]
X = merged_df_2014.drop(["Gender wage gap (%)", "Country", "Year"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=42)
log_model = LogisticRegression().fit(X_train, y_train)

# multicolinearity
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data

vif_results = calculate_vif(X)
print(vif_results)
#                                   Feature     VIF
# 0                        Women Seat Ratio   7.797
# 1                Maternal Mortality Ratio   5.815
# 2    Male Labour Force Participation Rate  95.932
# 3  Female Labour Force Participation Rate 269.944
# 4                    F/M Labor Force Part 195.432
# 5               Adolescent fertility rate   9.584
# iki değişkeni silmek gerek



y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
#               precision    recall  f1-score   support
#            0       0.00      0.00      0.00         0
#            1       1.00      0.80      0.89        10
#     accuracy                           0.80        10
#    macro avg       0.50      0.40      0.44        10
# weighted avg       1.00      0.80      0.89        10


from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, RocCurveDisplay   #plot_roc_curve

RocCurveDisplay.from_estimator(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()
# çizemedim

# AUC
roc_auc_score(y_test, y_prob)
#0.93
# error verdi

# Model Validation: 5-Fold Cross Validation
log_model = LogisticRegression().fit(X, y)
cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# Accuracy: 0.86

cv_results['test_precision'].mean()
# Precision: 0.9355555555555555

cv_results['test_recall'].mean()
# Recall: 0.9111111111111111

cv_results['test_f1'].mean()
# F1-score: 0.9200292397660819

cv_results['test_roc_auc'].mean()
# AUC: 0.7777777777777777

# deneme 2 loj
logreg = LogisticRegression().fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))
# Training set score: 0.900
# Test set score: 0.800

# ridge
from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))
# Training set score: 0.40
# Test set score: 0.00

# linear
lr = LinearRegression().fit(X_train, y_train)
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))
# Training set score: 0.40
# Test set score: 0.00

# x train farklı seçsem------------------------
merged_df_2014.columns

merged_df_2014.reset_index(inplace=True)

y = merged_df_2014["Gender wage gap (%)"]
X = merged_df_2014.drop(["Gender wage gap (%)", "Country", 'Male Labour Force Participation Rate',
       'Female Labour Force Participation Rate', "Year"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=42)

merged_df_2014['Gender wage gap (%)'] = merged_df_2014['Gender wage gap (%)'].apply(lambda x: 1 if x > 0 else 0)
log_model = LogisticRegression().fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
#               precision    recall  f1-score   support
#            0       0.00      0.00      0.00         0
#            1       1.00      0.90      0.95        10
#     accuracy                           0.90        10
#    macro avg       0.50      0.45      0.47        10
# weighted avg       1.00      0.90      0.95        10

# Model Validation: 5-Fold Cross Validation
log_model = LogisticRegression().fit(X, y)
cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# Accuracy: 0.86

cv_results['test_precision'].mean()
# Precision: 0.9355555555555555

cv_results['test_recall'].mean()
# Recall: 0.9111111111111111

cv_results['test_f1'].mean()
# F1-score: 0.9200292397660819

cv_results['test_roc_auc'].mean()
# AUC: 0.8666666666666666



merged_df_copy = merged_df.copy()

merged_df_copy.head()

merged_df_2014 = merged_df_copy[merged_df_copy["Year"] == 2014]
merged_df_2014.reset_index()
merged_df_2014.isnull().sum()

merged_df_2014 = merged_df_2014.drop(49)


y = merged_df_2014["Gender wage gap (%)"]
X = merged_df_2014.drop(["Gender wage gap (%)", "Country", 'Male Labour Force Participation Rate',
       'Female Labour Force Participation Rate', "Year"], axis=1)


# linear
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

y_test.shape
y_train.shape

lin_reg_model = LinearRegression().fit(X_train, y_train)

# 1- hold out yöntemi:
# Train RMSE
y_pred = lin_reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# 6.712896909893688
#

# Train RKARE
lin_reg_model.score(X_train, y_train)
# 0.4562168903260765

# Test RMSE
y_pred = lin_reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# 7.156711795989951
# test hatası normalde train hatasından daha yüksek çıkar

# Test RKARE
lin_reg_model.score(X_test, y_test)
# -0.2436628302988626


# 2- Cross validation yöntemi
# 10 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(lin_reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))
# 0.26010342713230333


# 5 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(lin_reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))
# 0.28077612080862735



# MSE
# y_pred = y_hat
y_pred = lin_reg_model.predict(X)
mean_squared_error(y, y_pred)
# 0.07531075650202572
y.mean()
y.std()
# satışların ortalaması 14 iken, bir tahmin yaptığımda ortalama 10 birim ile hata yapıyorsam bu büyüktür

# RMSE
np.sqrt(mean_squared_error(y, y_pred))
# 3.24

# MAE
mean_absolute_error(y, y_pred)
# 2.54

# R-KARE
reg_model.score(X, y)
# 0.61
# bağımsız değişkenlerin bağımlı değişkeni açıklama yüzdesidir.
# tv, sales'in yüzde 61ini açıklıyor
# makine öğrenmesinde optimal başarı kovalanır, robustness check'ler yapılmaz yani anlamlılık testleri



# ridge
from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train, y_train)


# multicollinearty denemesi
from statsmodels.stats.outliers_influence import variance_inflation_factor
# VIF
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data

vif_results = calculate_vif(X)
print(vif_results)
#                      Feature   VIF
# 0           Women Seat Ratio 7.763
# 1   Maternal Mortality Ratio 4.911
# 2       F/M Labor Force Part 7.912
# 3  Adolescent fertility rate 5.953

# 10 dan büyük değer yok o yüzden multicolinearity ok



##########################
# Tahmin Başarısını Değerlendirme
##########################

# 1- hold out yöntemi:
# Train RMSE
y_pred = ridge.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# 0.23586845656847694


# Train RKARE
ridge.score(X_train, y_train)
# 0.38184523551116123


# Test RMSE
y_pred = ridge.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# 0.3924278474154638
# test hatası normalde train hatasından daha yüksek çıkar

# Test RKARE
ridge.score(X_test, y_test)
# -0.7111068380792727

# test ve train'e bölmemize gerek yok sanki


# train set'siz ridge
ridge_2 = Ridge().fit(X_train, y_train)

##########################
# Tahmin Başarısını Değerlendirme
##########################

# 1- hold out yöntemi:
# Train RMSE
y_pred = ridge.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# 0.23586845656847694


# Train RKARE
ridge.score(X_train, y_train)
# 0.38184523551116123


# Test RMSE
y_pred = ridge.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# 0.3924278474154638
# test hatası normalde train hatasından daha yüksek çıkar

# Test RKARE
ridge.score(X_test, y_test)
# -0.7111068380792727



# test ve train'e bölmemize gerek yok sanki
# bir de encode'suz haliyle deneyeceğim
from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train, y_train)

##########################
# Tahmin Başarısını Değerlendirme
##########################

# 1- hold out yöntemi:
# Train RMSE
y_pred = ridge.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# 6.460922195996195



# Train RKARE
ridge.score(X_train, y_train)
# 0.4576928301390928


# Test RMSE
y_pred = ridge.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# 6.907153987133104
# test hatası normalde train hatasından daha yüksek çıkar

# Test RKARE
ridge.score(X_test, y_test)
# 0.30191121300305235

# test ve train'e bölmemize gerek yok sanki


# train set'siz ridge
ridge_2 = Ridge().fit(X_train, y_train)

##########################
# Tahmin Başarısını Değerlendirme
##########################

# Train RKARE
ridge.score(X_train, y_train)
# 0.4576928301390928


# Test RMSE
np.sqrt(mean_squared_error(y_test, y_pred))
# 6.907153987133104
# test hatası normalde train hatasından daha yüksek çıkar

# Test RKARE
ridge.score(X_test, y_test)
# 0.30191121300305235

# model son gali - 29 mart
# bu merged_df'i 680. satırdan getirdim
merged_df = pd.merge(gender_wage_gap, parliament, on=['Country', "Year"])
merged_df = pd.merge(merged_df, maternal_mortality, on=['Country', "Year"])
merged_df = pd.merge(merged_df, male_labor_force, on=['Country', "Year"])
merged_df = pd.merge(merged_df, female_labor_force, on=['Country', "Year"])
merged_df = pd.merge(merged_df, f_to_m_labor_force_part, on=['Country', "Year"])
merged_df = pd.merge(merged_df, adolescent_fertility_rate, on=['Country', "Year"])

# 2014'ü seçtim
merged_df_copy = merged_df.copy()

merged_df_copy.head()

merged_df_2014 = merged_df_copy[merged_df_copy["Year"] == 2014]
merged_df_2014.reset_index()
merged_df_2014.isnull().sum()

# boş değeri sildim
merged_df_2014 = merged_df_2014.drop(49)

merged_df_2014.isnull().sum()

# hedef değişkeni binary yaptım
merged_df_2014['Gender wage gap (%)'] = merged_df_2014['Gender wage gap (%)'].apply(lambda x: 1 if x > 0 else 0)


# dataframe'i böldüm
y = merged_df_2014["Gender wage gap (%)"]
X = merged_df_2014.drop(["Gender wage gap (%)", "Country", "Year"], axis=1)


# multicollinearty testini yaptım
from statsmodels.stats.outliers_influence import variance_inflation_factor
# VIF
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data

vif_results = calculate_vif(X)
print(vif_results)
#                                   Feature         VIF
# 0                        Women Seat Ratio    7.796947
# 1                Maternal Mortality Ratio    5.815239
# 2    Male Labour Force Participation Rate   95.931919
# 3  Female Labour Force Participation Rate  269.943975
# 4                    F/M Labor Force Part  195.431690
# 5               Adolescent fertility rate    9.584256

# burada female ve male labor force'u dataframe'den silmem gerektiğini gördüm
# bağımlı ve bağımsız değişkenleri yeniden seçiyorum o yüzden
y = merged_df_2014["Gender wage gap (%)"]
X = merged_df_2014.drop(["Gender wage gap (%)", "Country", 'Male Labour Force Participation Rate',
       'Female Labour Force Participation Rate', "Year"], axis=1)

# model
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=42)
log_model = LogisticRegression().fit(X_train, y_train)

cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# Accuracy: 0.86

cv_results['test_precision'].mean()
# Precision: 0.9355555555555555

cv_results['test_recall'].mean()
# Recall: 0.9111111111111111

cv_results['test_f1'].mean()
# F1-score: 0.9200292397660819

cv_results['test_roc_auc'].mean()
# AUC: 0.7777777777777777
