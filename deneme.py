placement = pd.read_csv("datasets/Placement.csv")

placement.head()
placement.columns
'''Data Dictionary
gender : Gender of the candidate
ssc_percentage : Senior secondary exams percentage (10th Grade)
ssc_board : Board of education for ssc exams
hsc_percentage : Higher secondary exams percentage (12th Grade)
hsc_borad : Board of education for hsc exams
hsc_subject : Subject of study for hsc
degree_percentage : Percentage of marks in undergrad degree
undergrad_degree : Undergrad degree majors
work_experience : Past work experience
emp_test_percentage : Aptitude test percentage  **yetenek testi yüzdesi
specialization : Postgrad degree majors - (MBA specialization)
mba_percent : Percentage of marks in MBA degree
status (TARGET) : Status of placement. Placed / Not Placed  **işe yerleştirilme durumu
Özellikle gelişmekte olan ülkelerde eğitimli ve yetenekli bireylere duyulan ihtiyacın artması nedeniyle, 
yeni mezunların işe alınması kuruluşlar için rutin bir uygulamadır. 
Geleneksel işe alım yöntemleri ve seçim süreçleri hatalara açık olabilir ve 
tüm süreci optimize etmek için bazı yenilikçi yöntemlere ihtiyaç vardır.'''
placement.shape
placement.isnull().sum() #eksik değer yok
placement.describe().T
placement.info()
placement.gender.value_counts()   #139 erkek, 76 kadın
placement.nunique()

def grab_col_names(dataframe, cat_th=10, car_th=20):
    #cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    #num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_car: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car

p_cat_cols, p_num_cols, p_cat_but_car = grab_col_names(placement)   #placement dataframine ait kategorik ve nümerik değişkenler old. için başına p koydum
p_cat_cols
'''['gender',
 'ssc_board',
 'hsc_board',
 'hsc_subject',
 'undergrad_degree',
 'work_experience',
 'specialisation',
 'status']'''
p_num_cols
'''['ssc_percentage',
 'hsc_percentage',
 'degree_percentage',
 'emp_test_percentage',
 'mba_percent']'''

#status grafiği

sns.countplot(x='status', data=placement, hue='gender')
plt.show()

#specialisation
sns.countplot(x='specialisation', data=placement, hue='gender')
plt.show()

#bunların yerine kategoriklerin hepsini cinsiyet ayrımında subplot ile yapalım
plt.figure(figsize = (15, 10))
plt.suptitle("Analysis Of Variable Gender",fontweight="bold", fontsize=18)
plt.subplot(4,2,1)
sns.countplot(x='status', hue='gender', palette='Set2', data=placement)
plt.subplot(4,2,2)
sns.countplot(x='ssc_board', hue='gender', palette='Set2', data=placement)
plt.subplot(4,2,3)
sns.countplot(x='hsc_board', hue='gender', palette='Set2', data=placement)
plt.subplot(4,2,4)
sns.countplot(x ='hsc_subject', hue='gender', palette='Set2', data=placement)
plt.subplot(4,2,5)
sns.countplot(x='undergrad_degree', hue='gender', palette='Set2', data=placement)
plt.subplot(4,2,6)
sns.countplot(x='work_experience', hue='gender', palette='Set2', data=placement)
plt.subplot(4,2,7)
sns.countplot(x='specialisation', hue='gender', palette='Set2', data=placement)
plt.tight_layout()  # Grafiklerin sığabileceği şekilde ayarlama
plt.show()


#nümerik kolonların histogramı
placement.hist(figsize=(12,8))
plt.show()



# Kadın ve erkek gruplarını ayırma ve her bir sütun için nümerik kolonların histogramları ayrı ayrı çizme
fig, axs = plt.subplots(len(p_num_cols), 2, figsize=(12, 2 * len(p_num_cols)))

for i, column in enumerate(p_num_cols):
    male_data = placement[placement['gender'] == 'M']
    female_data = placement[placement['gender'] == 'F']

    male_data[column].hist(ax=axs[i, 0], color='blue')
    axs[i, 0].set_title('Male ' + column)
    axs[i, 0].set_ylabel('Frequency')

    female_data[column].hist(ax=axs[i, 1], color='red')
    axs[i, 1].set_title('Female ' + column)
    axs[i, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    print(f"{col_name} sütunu için alt ve üst limit değerleri: ({low_limit}, {up_limit})")
    return low_limit, up_limit

for col in p_num_cols:
    outlier_thresholds(placement, col)


def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

for col in p_num_cols:
    grab_outliers(placement, col)
#outlier yok


placement.groupby(['gender', 'status'])[p_num_cols].mean()

#başarı yüzdelerinin ortalamasından oluşan bir sütun ekleyelim
#all_percent = (ssc_percentage + hsc_percentage + degree_percentage + emp_test_percentage + mba_percent) / 4

placement['all_percent'] = (placement['ssc_percentage'] + placement['degree_percentage'] + placement['emp_test_percentage'] + placement['mba_percent']) / 4

placement.info()
placement.head()
p_num_cols.append('all_percent')

placement.groupby(['gender', 'status'])[p_num_cols].mean()
placement.groupby(['gender', 'status'])['all_percent'].mean()
placement[(placement['all_percent'] < 80) & (placement['status'] == 'Placed')].sort_values(by='all_percent')[['gender', 'all_percent', 'status']]



# Nümerik sütunların box plotlarını çizelim
plt.figure(figsize=(10, 6))  # Grafik boyutunu ayarlayalım
sns.boxplot(data=placement, orient="h")  # Box plotları yatay olarak çizelim
plt.title('Nümerik Sütunların Box Plotları')  # Grafik başlığını ekleyelim
plt.xlabel('Değerler')  # X ekseni etiketini ekleyelim
plt.show()  # Grafiği gösterelim

#korelasyon matrisi
plt.figure(figsize=(8, 6))
sns.heatmap(placement[p_num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Korelasyon Matrisi')
plt.show()

placement.info()



import plotly.graph_objects as go
# Veri setindeki nümerik kolonların heatmap için bir örnek veri oluşturulması
data = placement[p_num_cols].corr()
# Heatmap oluşturma
fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=data.columns,
        colorscale='Viridis'))
# Grafiği gösterme
fig.update_layout(title="Nümerik Kolonlar için Korelasyon Heatmap'i")
fig.show()
#aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa!