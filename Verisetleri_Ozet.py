# NOTLAR

f_to_m_unpaid_care_work = pd.read_csv("datasets/1- female-to-male-ratio-of-time-devoted-to-unpaid-care-work.csv")
w_in_top_income_groups = pd.read_csv("datasets/2- share-of-women-in-top-income-groups.csv")
f_to_m_labor_force_part = pd.read_csv("datasets/3- ratio-of-female-to-male-labor-force-participation-rates-ilo-wdi.csv")
maternal_mortality = pd.read_csv("datasets/5- maternal-mortality.csv")
gender_wage_gap = pd.read_csv("datasets/6- gender-gap-in-average-wages-ilo.csv")
w_entrepreneurship = pd.read_csv("datasets/Labor Force-Women Entrpreneurship.csv", sep=";")
male_labor_force = pd.read_csv("datasets/Labour Force Participation - Male.csv")
female_labor_force = pd.read_csv("datasets/Labour Force Participation Female.csv")
placement = pd.read_csv("datasets/Placement.csv")
parliament = pd.read_excel("datasets/Parliament.xlsx")
adolescent_fertility_rate = pd.read_excel("datasets/Adolescent_Fertility_Rate.xlsx")
human_dev_indices = pd.read_excel("datasets/Human Development Composite Indices.xlsx")


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


#########################

# Aşağıdaki verisetlerinde ülke isimlerini temsil eden Entity değişkeninin ismi, uyumluluk sağlamak için Country olarak değiştirildi.
    # f_to_m_unpaid_care_work
    # w_in_top_income_groups
    # gender_wage_gap
    # TODO sanırım bunun daha iyisi main'de mevcut
entity_dfs = [f_to_m_unpaid_care_work, w_in_top_income_groups, gender_wage_gap]

for df in entity_dfs:
    df.replace({"Entity":"Country"})

# Ülke kodlarını temsil eden değişkenler, veriseti birleştirme ve makine öğrenmesi süreçlerinde sorun yaratma olasılığına karşın drop edildi.
    # f_to_m_unpaid_care_work - Code
    # w_in_top_income_groups - Code
    # f_to_m_labor_force_part - Code
    # maternal_mortality - Code
    # gender_wage_gap - Code
    # w_entrepreneurship - No
    # male_labor_force - ISO3
    # female_labor_force - ISO3
    # parliament - Country Code
    # adolescent_fertility_rate: Country Code

# Female labor force participation ve male labor force participation verisetleri birleştirildi
    # merged_labor_force

# Kadınların parlamento katılım oranları veriseti ve çocuk hamileliği veriseti eklendi.
    # Parliament
    # Adolescent_fertility_rate
