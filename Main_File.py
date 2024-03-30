##############################################
# DATATHON MART 2024
##############################################

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from functools import reduce
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

# Load datasets
f_to_m_unpaid_care_work = pd.read_csv("datasets/1- female-to-male-ratio-of-time-devoted-to-unpaid-care-work.csv")
f_to_m_unpaid_care_work.name = 'f_to_m_unpaid_care_work'

w_in_top_income_groups = pd.read_csv("datasets/2- share-of-women-in-top-income-groups.csv")
w_in_top_income_groups.name = 'w_in_top_income_groups'

f_to_m_labor_force_part = pd.read_csv("datasets/3- ratio-of-female-to-male-labor-force-participation-rates-ilo-wdi.csv")
f_to_m_labor_force_part.name = 'f_to_m_labor_force_part'

maternal_mortality = pd.read_csv("datasets/5- maternal-mortality.csv")
maternal_mortality.name = 'maternal_mortality'

gender_wage_gap = pd.read_csv("datasets/6- gender-gap-in-average-wages-ilo.csv")
gender_wage_gap.name = 'gender_wage_gap'

w_entrepreneurship = pd.read_csv("datasets/Labor Force-Women Entrpreneurship.csv", sep=";")
w_entrepreneurship.name = 'w_entrepreneurship'

male_labor_force = pd.read_csv("datasets/Labour Force Participation - Male.csv")
male_labor_force.name = 'male_labor_force'

female_labor_force = pd.read_csv("datasets/Labour Force Participation Female.csv")
female_labor_force.name = 'female_labor_force'

placement = pd.read_csv("datasets/Placement.csv")
placement.name = 'placement'

# 1 ve 4, 7 ve 11 excel'ler aynı
parliament = pd.read_excel("datasets/Parliament.xlsx")
parliament.name = 'parliament'

adolescent_fertility_rate = pd.read_excel("datasets/Adolescent_Fertility_Rate.xlsx")
adolescent_fertility_rate.name = 'adolescent_fertility_rate'

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

# TODO bunlara bak

df_list3 = list(df_names.values())

df_list = [f_to_m_unpaid_care_work,
           w_in_top_income_groups,
           f_to_m_labor_force_part,
           maternal_mortality,
           gender_wage_gap,
           w_entrepreneurship,
           male_labor_force,
           female_labor_force,
           placement,
           parliament,
           adolescent_fertility_rate]

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

# 2- share-of-women-in-top-income-groups
w_in_top_income_groups.head()
w_in_top_income_groups.shape # (168, 9)
w_in_top_income_groups.info() # veritiplerinde sıkıntı yok.
w_in_top_income_groups["Share of women in top 0.1%"].describe()
w_in_top_income_groups.loc[w_in_top_income_groups["Share of women in top 0.1%"]==4.600] # min: Denmark
w_in_top_income_groups.loc[w_in_top_income_groups["Share of women in top 0.1%"]==20.000] # max: Spain
w_in_top_income_groups["Share of women in top 10%"].describe()
w_in_top_income_groups.loc[w_in_top_income_groups["Share of women in top 10%"]==9.400] # min: Denmark
w_in_top_income_groups.loc[w_in_top_income_groups["Share of women in top 10%"]==34.800] # max: Spain

# 3- ratio-of-female-to-male-labor-force-participation-rates-ilo-wdi
f_to_m_labor_force_part.head()
f_to_m_labor_force_part.shape # (6432, 4)
f_to_m_labor_force_part.info() # veritiplerinde sıkıntı yok.

#4 (1 ile aynı olduğu için silindi)

# 5- maternal-mortality
# The maternal mortality ratio is the number of women who die from pregnancy-related causes while pregnant or within 42 days of pregnancy termination per 100,000 live births.
maternal_mortality.head()
maternal_mortality.shape # (5800, 4)
maternal_mortality.info() # veritiplerinde sıkıntı yok.

# 6- gender-gap-in-average-wages-ilo
gender_wage_gap.head(30)
gender_wage_gap.shape # (413, 4)
gender_wage_gap.info() # veritiplerinde sıkıntı yok.

# adolescent_fertility_rate
adolescent_fertility_rate.head()
adolescent_fertility_rate.shape # (266, 65)
adolescent_fertility_rate.info() # veritiplerinde sıkıntı yok.

# Labor Force-Women Entrpreneurship
w_entrepreneurship.head()
w_entrepreneurship.shape # (51,9)
w_entrepreneurship.info() # veritiplerinde sıkıntı yok.

# Labour Force Participation - Male
# 195 ülke için, 1990-2021 arasındaki yıllar için, 15 yaş ve üzeri erkeklerin iş gücüne katılma oranları
male_labor_force.head()
male_labor_force.shape # (195, 37)
male_labor_force.info() # veritiplerinde sıkıntı yok.

# Labour Force Participation - Female
# 195 ülke için, 1990-2021 arasındaki yıllar için, 15 yaş ve üzeri kadınların iş gücüne katılma oranları
female_labor_force.head()
female_labor_force.shape # (195, 37)
female_labor_force.info() # veritiplerinde sıkıntı yok.

# parliament
parliament.head()
parliament.shape # (266, 65)
parliament.info() # veritiplerinde sıkıntı yok.

# Placement
placement.head()
placement.shape # (215, 13)
placement.info() # veritiplerinde sıkıntı yok.
placement["hsc_board"].info()

#Women Ent_Data3 (7 ile aynı olduğu için silindi)


##############################################
# DEEP DATA
##############################################
# Verinin özünü anlama ve derinlemesine analiz etme yeteneği büyük önem taşımaktadır.
# Veri setinin karmaşıklığını anlayarak, içindeki değerli bilgileri keşfetmeye odaklanmanız önemlidir.

# Başka isimlerle verilen ülke değişkeni ismini Country olarak standardize edelim:
cols_to_change = ["Entity", "Country Name"]
for df in df_list:
    matched_cols = [col for col in cols_to_change if col in df.columns]
    if matched_cols:
        print(f"Matched columns in {df.name}: {', '.join(matched_cols)}")
        rename_dict = {col: 'Country' for col in matched_cols}  # Dictionary to hold old and new column names
        df.rename(columns=rename_dict, inplace=True)  # Use rename_dict in rename method
        print(f"--> Columns renamed as 'Country' in {df.name}: {', '.join(matched_cols)}\n")
    else:
        print(f"No matched columns in {df.name}\n")


# Ülke ismi kısaltmalarını içeren değişkenleri silelim:
for df in df_list:
    cols_to_drop = ["Code", "Country Code", "ISO3", "No"]
    matched_cols = [col for col in cols_to_drop if col in df.columns]
    if matched_cols:
        print(f"Matched variables in {df.name}: {', '.join(matched_cols)}")
        for col in matched_cols:
            df.drop(columns=col, inplace=True)
        print(f"--> Dropped columns from {df.name}: {', '.join(matched_cols)}\n")
    else:
        print(f"No matched variables in {df.name}\n")

# Gereksiz indikatör isim değişkenlerini silelim:
for df in df_list:
    inds_to_drop = ["Indicator Name", "Indicator Code"]
    matched_inds = [ind for ind in inds_to_drop if ind in df.columns]
    if matched_inds:
        print(f"Indicator variables in {df.name}: {', '.join(matched_inds)}")
        print("###################")
        print(df[matched_inds].nunique())
        print("###################")
        for ind in matched_inds:
            df.drop(columns=ind, inplace=True)
        print(f"--> Dropped columns from {df.name}: {', '.join(matched_inds)}\n")
    else:
        print(f"No indicators in {df.name}\n")

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

# Farklı yılları temsil eden sütunları, satır yapalım ve Year değişkenini string'den yarattığımız için tipini int yapalım:
parliament = pd.melt(parliament, id_vars=['Country'], var_name='Year', value_name='Women Seat Ratio')
parliament["Year"] = parliament["Year"].astype("int")

adolescent_fertility_rate = pd.melt(adolescent_fertility_rate, id_vars=['Country'], var_name='Year', value_name='Adolescent fertility rate')
adolescent_fertility_rate["Year"] = adolescent_fertility_rate["Year"].astype("int")

# Male ve female labor force df'lerini process ediyoruz:
labor_force_dfs = [male_labor_force, female_labor_force]
new_labor_dfs = []
var_name = "Labour Force Participation Rate"

for df in labor_force_dfs:
    df_copy = df.copy()  # Create a copy of the DataFrame
    df_copy = pd.melt(df_copy, id_vars=['Country', 'Continent', 'Hemisphere', 'HDI Rank (2021)'],
                      var_name=var_name)
    if any(df_copy[var_name].str.contains('female', case=False, na=False)):
        df_copy.rename(columns={'value': 'Female Labour Force Participation Rate'}, inplace=True)
    else:
        df_copy.rename(columns={'value': 'Male Labour Force Participation Rate'}, inplace=True)
    year_val = []
    for val in df_copy[var_name].values:
        year = val.split(" ")[-1].replace("(", "").replace(")", "")
        year_val.append(year)
    df_copy['Year'] = df_copy[var_name].replace(df_copy[var_name].tolist(), year_val)
    df_copy['Year'] = df_copy['Year'].astype('int')
    # Verisetini şu anki gibi yıla göre değil, ülkelere göre alfabetik sıralayalım:
    df_copy.sort_values(by='Country', inplace=True)
    # Fazla değişkenleri silelim:
    df_copy.drop([var_name, "HDI Rank (2021)", "Continent", "Hemisphere"], axis=1, inplace=True)
    new_labor_dfs.append(df_copy)

male_labor_force, female_labor_force = new_labor_dfs

male_labor_force.head()
female_labor_force.head()

# Country değerleri standardizasyonu
countries = []
for df in df_list:
    if "Country" in df.columns:
        country_vals_list = df["Country"].tolist()
        countries.extend(country_vals_list)
"""
# Her bir ülkeye ait kaç gözlemimiz olduğuna bakalım:
for country in set(countries):
    tekrar_sayisi = countries.count(country)
    print(f"{country}: {tekrar_sayisi}")
"""

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
         'Russian Federation': 'Russia',
         'Saint Kitts and Nevis': 'St. Kitts and Nevis',
         'Saint Lucia': 'St. Lucia',
         'Saint Vincent and the Grenadines': 'St. Vincent and the Grenadines',
         'Slovak Republic': 'Slovakia',
         'Syrian Arab Republic': 'Syria',
         'Turkey': 'Turkiye',
         'UK': 'United Kingdom',
         'Venezuela, RB': 'Venezuela',
         'Viet Nam': 'Vietnam',
         'United States Virgin Islands': 'Virgin Islands (U.S.)',
         'Yemen, Rep.': 'Yemen',
         'Palestine, State of': 'Palestine',
         'West Bank and Gaza': 'Palestine',
         'Lao': 'Laos',
         'Lao PDR': 'Laos'}

# Farklı yazımlara sahip benzer isimleri olan ülkeleri ayıralım sadeleştirme için kaynak verisetlerinden kontrol edebilelim:
confusions = ['Congo', 'Congo, Dem. Rep.', 'Congo, Rep.', 'Democratic Republic of Congo','The Democratic Republic of the Congo',  # TODO Congo derken?
              'Korea', "Korea, Dem. People's Rep.", 'Korea, Rep.', 'North Korea', 'South Korea',
              'Micronesia', 'Micronesia (country)', 'Micronesia, Fed. Sts.']
len(confusions)

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
        confusion_df_dict.update({name: confusions_in_df})
        #globals()[f"confusions_in_{name}"] = confusions_in_df
    else:
        print("[]")
    print("\n")

confusion_df_dict.keys()

# Temizleyelim.
confusion_sorted = {'Congo': 'Congo, Rep.',
                   'Democratic Republic of Congo': 'Congo, Dem. Rep.',
                   'The Democratic Republic of the Congo': 'Congo, Dem. Rep.',
                   'Korea': 'South Korea',
                   'Korea, Rep.': 'South Korea',
                   "Korea, Dem. People's Rep.": 'North Korea',
                   'Micronesia (country)': 'Micronesia',
                   'Micronesia, Fed. Sts.': 'Micronesia'}

# Update df_list with updated versions
df_list = [f_to_m_unpaid_care_work, w_in_top_income_groups, f_to_m_labor_force_part, maternal_mortality, gender_wage_gap, w_entrepreneurship, male_labor_force, female_labor_force, placement, parliament, adolescent_fertility_rate]

# Replace Country names (can also add code to remove region names).
dfs_after_confusion_sorted = []
for df in df_list:
    df_copy = df.copy()
    if "Country" in df_copy.columns:
        df_copy["Country"] = df_copy["Country"].replace(confusion_sorted).replace(diffs)
    dfs_after_confusion_sorted.append(df_copy)
f_to_m_unpaid_care_work, w_in_top_income_groups, f_to_m_labor_force_part, maternal_mortality, gender_wage_gap, w_entrepreneurship, male_labor_force, female_labor_force, placement, parliament, adolescent_fertility_rate = dfs_after_confusion_sorted

# Kontrol edelim:
print(f_to_m_unpaid_care_work[f_to_m_unpaid_care_work["Country"] == "Korea"])
sorted(parliament["Country"].unique())

# Update df_list with updated versions
df_list = [f_to_m_unpaid_care_work, w_in_top_income_groups, f_to_m_labor_force_part, maternal_mortality, gender_wage_gap, w_entrepreneurship, male_labor_force, female_labor_force, placement, parliament, adolescent_fertility_rate]

# Şimdi yılları gruplayalım:
for df in df_list:
    if "Year" in df.columns:
        df["Year_group"] = pd.cut(df["Year"], [1995, 2002, 2009, 2016])
        df.groupby(["Country", "Year_group"]).mean().reset_index()
        df.dropna(subset=df.columns.difference(["Country", "Year_group"]), how="all", inplace=True)
        df.drop("Year", axis=1, inplace=True)
        print(df.columns)

parliament.columns

# Şimdi mergeleyelim:
dfs_to_merge = [gender_wage_gap, parliament, maternal_mortality, male_labor_force, female_labor_force, f_to_m_labor_force_part, adolescent_fertility_rate]
merge_on_columns = ['Country', 'Year_group']
merged_df = reduce(lambda left, right: pd.merge(left, right, on=merge_on_columns), dfs_to_merge) # TODO nedense bu şimdi çalışmadı
merged_df.head()
merged_df.shape # TODO (324, 9) BİR ARALAR BÖYLEYDİ
merged_df.columns

# ['Country', 'Year', 'Gender wage gap (%)', 'Women Seat Ratio', 'Maternal Mortality Ratio', 'Male Labour Force Participation Rate',
#        'Female Labour Force Participation Rate', 'F/M Labor Force Part', 'Adolescent fertility rate']

# TODO Bu rakamları kontrol et
# Df'lere Year_group atamalarına göre merge sonrasında kalan gözlem sayısı:
merged_df["Year_group"].value_counts() # [1995, 2002, 2009, 2016])
# (2010, 2015]    117
# (2005, 2010]     99
# (2000, 2005]     78
# 3 grupta da gözlemi olan 24 ülke.

# (2011, 2016]    111
# (2001, 2006]     89
# (2006, 2011]     89
# (1996, 2001]     29
# 3 grupta da gözlemi olan 12 ülke.

# (2009, 2016]    155
# (2002, 2009]    109
# (1995, 2002]     56
# 3 grupta da gözlemi olan 25 ülke.
####

"""
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
"""

# İlk merged_df'imizin 2014 cross-section verilerini üzerine modeller. Eski model olduğu için comment yaparak sakladım.
# Eda-------------------------------------------------------------------------
# ----ve sonra Gizem de var
"""
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
# bu merged_df'i 680. satırdan getirdim (merge fonksiyonu güncellendi ve belge düzenlendi. eşdeğer merge artık 489. satırda. - D)
merged_df = pd.merge(gender_wage_gap, parliament, on=['Country', "Year"])
merged_df = pd.merge(merged_df, maternal_mortality, on=['Country', "Year"])
merged_df = pd.merge(merged_df, male_labor_force, on=['Country', "Year"])
merged_df = pd.merge(merged_df, female_labor_force, on=['Country', "Year"])
merged_df = pd.merge(merged_df, f_to_m_labor_force_part, on=['Country', "Year"])
merged_df = pd.merge(merged_df, adolescent_fertility_rate, on=['Country', "Year"])

merged_df.head()

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

# İlk merged_df'imiz üzerinde hangi yılın daha dolu olduğunu keşfetme çalışmaları:
merged_df.head()

merged_df["Year"].value_counts()
# 2014    51
# 2010    28
# 2006    26
# 2002    25
# 2013    19
# 2012    18
# 2009    17
# 2011    16
# 2008    15
# 2001    15
# 2005    14

# 2014 ve 2010
df_2014_2010 = merged_df[(merged_df["Year"] == 2014) | (merged_df["Year"] == 2010)]
df_2014_2010["Country"].value_counts()

desired_countries = df_2014_2010["Country"].value_counts()[df_2014_2010["Country"].value_counts() == 2].index.tolist()
print("Desired Countries with Counts 2:", desired_countries)
# Desired Countries with Counts 2: ['Argentina', 'Netherlands', 'Ireland', 'Italy', 'Austria', 'Luxembourg', 'Malta', 'Mexico',
# 'Panama', 'Germany', 'Paraguay', 'Peru', 'Slovakia', 'Slovenia', 'South Korea', 'Spain', 'Honduras', 'Uruguay', 'Ecuador',
# 'El Salvador', 'Cyprus', 'Dominican Republic', 'Belgium', 'Colombia', 'Finland', 'France']
len(desired_countries)
# 26

# 2014, 2010, 2006
df_2014_2010_2006 = merged_df[(merged_df["Year"] == 2014) | (merged_df["Year"] == 2010) | (merged_df["Year"] == 2006)]
df_2014_2010_2006["Country"].value_counts()

desired_countries = df_2014_2010_2006["Country"].value_counts()[df_2014_2010_2006["Country"].value_counts() == 3].index.tolist()
print("Desired Countries with Counts 3:", desired_countries)
# Desired Countries with Counts 3: ['Argentina', 'Ecuador', 'Peru', 'Netherlands', 'Luxembourg', 'Austria', 'Italy', 'Ireland',
# 'Honduras', 'Panama', 'France', 'Finland', 'El Salvador', 'Germany', 'Dominican Republic', 'Spain', 'Belgium', 'Uruguay', 'Paraguay']
len(desired_countries)
# 19


# 2014, 2010, 2006, 2002
df_2014_2010_2006_2002 = merged_df[(merged_df["Year"] == 2014) | (merged_df["Year"] == 2010) | (merged_df["Year"] == 2006) | (merged_df["Year"] == 2002)]
df_2014_2010_2006_2002["Country"].value_counts()

desired_countries = df_2014_2010_2006_2002["Country"].value_counts()[df_2014_2010_2006_2002["Country"].value_counts() == 4].index.tolist()
print("Desired Countries with Counts 4:", desired_countries)
# Desired Countries with Counts 4: ['Argentina', 'Panama', 'Ireland', 'Luxembourg', 'Honduras', 'Germany', 'France', 'Finland', 'Netherlands',
# 'El Salvador', 'Italy', 'Dominican Republic', 'Paraguay', 'Peru', 'Spain', 'Uruguay', 'Belgium', 'Austria']
len(desired_countries)
# 18
"""

#################### SON MODEL #################################################3
# Veri çerçevelerini birleştirme
merged_df = pd.merge(gender_wage_gap, parliament, on=['Country', "Year"])
merged_df = pd.merge(merged_df, maternal_mortality, on=['Country', "Year"])
merged_df = pd.merge(merged_df, male_labor_force, on=['Country', "Year"])
merged_df = pd.merge(merged_df, female_labor_force, on=['Country', "Year"])
merged_df = pd.merge(merged_df, f_to_m_labor_force_part, on=['Country', "Year"])
merged_df = pd.merge(merged_df, adolescent_fertility_rate, on=['Country', "Year"])

min_year = merged_df['Year'].min() # 1990
max_year = merged_df['Year'].max() # 2016

merged_df.head()

merged_df.shape # 324, 9

merged_df.isnull().sum()
# Country                                    0
# Year                                       0
# Gender wage gap (%)                        0
# Women Seat Ratio                          14
# Maternal Mortality Ratio                   0
# Male Labour Force Participation Rate       0
# Female Labour Force Participation Rate     0
# F/M Labor Force Part                       0
# Adolescent fertility rate                  0

# doldurma denemesi
merged_df["Women Seat Ratio"] = merged_df["Women Seat Ratio"].fillna(merged_df.groupby("Country")["Women Seat Ratio"].transform("median"))
merged_df.isnull().sum()

null_row = merged_df[merged_df.isnull().any(axis=1)]
# (323, 9)

# bir satır var, bunu sileceğim, bu ülkeden de tek bir satır var zaten
merged_df.dropna(subset=["Women Seat Ratio"], inplace=True)

min_year = merged_df['Year'].min() # 1990
max_year = merged_df['Year'].max() # 2016

merged_df["Year"].value_counts()

merged_df["Year"].value_counts().sort_index()
# Year
# 1990     1
# 1992     2
# 1994     1
# 1996     2
# 1998     2
# 2000    12
# 2001    15
# 2002    25
# 2003    12
# 2004    12
# 2005    14
# 2006    26
# 2007    13
# 2008    15
# 2009    17
# 2010    28
# 2011    16
# 2012    18
# 2013    19
# 2014    50
# 2015    13
# 2016    10

merged_df["Country"]

country_year_count = merged_df.groupby('Country')['Year'].nunique()

def assign_year_category(year):
    if 1990 <= year < 2000:
        return 0
    elif 2000 <= year < 2005:
        return 1
    elif 2005 <= year < 2010:
        return 2
    elif 2010 <= year < 20017:
        return 3
    else:
        return None  # Diğer durumlar için NaN (opsiyonel)

# Yeni bir "yıl_kategorisi" sütunu oluşturma
merged_df['yıl_kategorisi'] = merged_df['Year'].apply(assign_year_category)

merged_df['yıl_kategorisi'].isnull().sum() # 0

merged_df.drop("Year", inplace=True, axis=1)


# ülkelere göre one hot encoding
merged_df["Country"].nunique() # 62
merged_df.shape # 310


merged_df['yıl_kategorisi'].value_counts()

merged_df = pd.get_dummies(merged_df, columns=["Country"], drop_first=True, dtype="int")

merged_df.shape
merged_df.describe().T
merged_df.head()
merged_df.columns
# multiple linear regression
X = merged_df.drop(['Gender wage gap (%)', 'Male Labour Force Participation Rate', 'Female Labour Force Participation Rate'], axis=1)
y = merged_df[["Gender wage gap (%)"]]

##########################
# Model
##########################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

reg_model = LinearRegression().fit(X_train, y_train)

rs = RobustScaler()
merged_df["Gender wage gap (%)"] = rs.fit_transform(merged_df[["Gender wage gap (%)"]])
merged_df["Women Seat Ratio"] = rs.fit_transform(merged_df[["Women Seat Ratio"]])
merged_df["Maternal Mortality Ratio"] = rs.fit_transform(merged_df[["Maternal Mortality Ratio"]])

merged_df.describe().T

# MSE
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)
# 0.08309789974380676

giz = mean_squared_error(y, y_pred)
print("MSE:", giz)

# RMSE
np.sqrt(mean_squared_error(y, y_pred))
# 0.28826706323096773

# MAE
mean_absolute_error(y, y_pred)
# 0.19587493487647156

# R-KARE
reg_model.score(X, y)
# 0.842719425168627

# valida
##########################
# Tahmin Başarısını Değerlendirme
##########################

# 1- hold out yöntemi:
# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# 0.27978577008352923
# yeni dğeşken eklendiği için hata düştü

# Train RKARE
reg_model.score(X_train, y_train)
# 0.8619443684603638

# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# 0.3199518559941478
# test hatası normalde train hatasından daha yüksek çıkar

# Test RKARE
reg_model.score(X_test, y_test)
# 0.7242618894974271

# 2- Cross validation yöntemi

# 5 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))
# 13.971111151974242