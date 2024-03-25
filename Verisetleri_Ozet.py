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


1)  1- female-to-male-ratio-of-time-devoted-to-unpaid-care-work
    f_to_m_unpaid_care_work.head()
    """
    Değişkenler:
        'Entity'
            Country
        'Code'
            Country code
        'Year'
            2014
        'Female to male ratio of time devoted to unpaid care work (OECD (2014))'
    """

2)  2- share-of-women-in-top-income-groups
    w_in_top_income_groups.head()
    """
    Değişkenler:
        'Entity'
            Country
        'Code'
            Country code
        'Year'
        'Share of women in top 0.1%'
        'Share of women in top 0.25%'
        'Share of women in top 0.5%'
        'Share of women in top 1%'
        'Share of women in top 10%'
        'Share of women in top 5%'
    """

3)  3- ratio-of-female-to-male-labor-force-participation-rates-ilo-wdi
    f_to_m_labor_force_part.head()
    """
    Değişkenler:
        'Country'
        'Code'
        'Year'
        'Ratio of female to male labor force participation rate (%) (modeled ILO estimate)'
    """

#4 (1 ile aynı olduğu için silindi)

4)  5- maternal-mortality
    maternal_mortality.head()
    """
    Değişkenler:
        'Country'
        'Code'
        'Year'
        'Maternal Mortality Ratio (Gapminder (2010) and World Bank (2015))'
    """

5)  6- gender-gap-in-average-wages-ilo
    gender_wage_gap.head()
    """
    Değişkenler:
        'Entity'
            Country
        'Code'
            Country code
        'Year'
        'Gender wage gap (%)'
    """

6)  Labor Force-Women Entrpreneurship
    w_entrepreneurship.head()
    """
    Değişkenler:
        'No'
            Country code
        'Country'
        'Level of development'
            Developed 27, Developing 24
        'European Union Membership'
            Not member 31, Member 20
        'Currency'
            National Currency 36, Euro 15
        'Women Entrepreneurship Index'
        'Entrepreneurship Index'
        'Inflation rate'
        'Female Labor Force Participation Rate'
"""
    #Labour Force Participation - Male
    #Labour Force Participation Female

7)  Merged_labor_force
    """
    """

8)  Placement
    placement.head()
    """
    Değişkenler:
        'gender'
            M/F
        'ssc_percentage'
        'ssc_board'
            Others/Central
        'hsc_percentage'
        'hsc_board'
        'hsc_subject'
            Commerce 113
            Science 91
            Arts 11
        'degree_percentage'
        'undergrad_degree'
            Comm&Mgmt    145
            Sci&Tech      59
            Others        11
        'work_experience'
            No     141
            Yes     74
        'emp_test_percentage'
        'specialisation'
            Mkt&Fin    120
            Mkt&HR      95
        'mba_percent'
        'status'
            Placed/Not placed
    """
#Women Ent_Data3 (7 ile aynı olduğu için silindi)

# Aşağıdaki verisetlerinde ülke isimlerini temsil eden Entity değişkeninin ismi, uyumluluk sağlamak için Country olarak değiştirildi.
    # f_to_m_unpaid_care_work
    # w_in_top_income_groups
    # gender_wage_gap
    #