import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, RocCurveDisplay   #plot_roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


placement = pd.read_csv("datasets/Placement.csv")
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
placement.head()
placement.columns
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
placement.groupby('gender')['undergrad_degree'].value_counts()


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

placement.describe().T
placement.head()

#ortalamaları sınıflandırma
class_boundaries = [0, 40, 60, 70, 80, 90, 100]
class_labels = [0, 1, 2, 3, 4, 5]

for col in p_num_cols:
    new_name = col.split("_")[0] + "_label"
    placement[new_name] = pd.cut(placement[col], bins=class_boundaries, labels=class_labels)


binary_cols = [col for col in placement.columns if placement[col].dtype not in [int, float]
               and placement[col].nunique() == 2]

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    label_encoder(placement, col)

placement.head()

placement = pd.get_dummies(placement, columns=["hsc_subject", "undergrad_degree"], drop_first=True, dtype="int")

p_num_cols

#standartlaştırma
scaler = StandardScaler()
placement[p_num_cols] = scaler.fit_transform(placement[p_num_cols])

placement.head()

#model

y = placement["status"]
X = placement.drop("status", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=42)
log_model = LogisticRegression().fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))

RocCurveDisplay.from_estimator(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

# AUC
roc_auc_score(y_test, y_prob)
#0.93

# Model Validation: 5-Fold Cross Validation
log_model = LogisticRegression().fit(X, y)
cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# Accuracy: 0.8325

cv_results['test_precision'].mean()
# Precision: 0.8675

cv_results['test_recall'].mean()
# Recall: 0.8983

cv_results['test_f1'].mean()
# F1-score: 0.8809

cv_results['test_roc_auc'].mean()
# AUC: 0.9221

# #randomForest
# y = placement["status"]
# X = placement.drop(["status"], axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
#
# from sklearn.ensemble import RandomForestClassifier
# rf_model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
# y_pred = rf_model.predict(X_test)
# print(accuracy_score(y_pred, y_test))
# #0.7906



# Modelinizin katsayılarını alın
coefficients = log_model.coef_[0]

# Özellik adlarını alın
feature_names = X.columns

# Katsayıları görselleştirin
plt.figure(figsize=(10,6))
plt.barh(feature_names, coefficients, color='skyblue')
plt.xlabel('Katsayı Değeri')
plt.ylabel('Özellik')
plt.title('Logistic Regresyon Modeli Özellik Katsayıları')
plt.grid(True)
plt.show()


#özellik önemi

# Özellik katsayılarını al
feature_importance = np.abs(log_model.coef_[0])

# Özellik isimleri
feature_names = X_train.columns

# Özellik katsayılarını ve isimlerini birleştir
feature_importance_dict = dict(zip(feature_names, abs(feature_importance)))

# Özellik katsayılarını sırala
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# Grafik için verileri ayır
features = [x[0] for x in sorted_feature_importance]
importance = [x[1] for x in sorted_feature_importance]


plt.figure(figsize=(10,6))
plt.barh(features, importance, color='skyblue')
plt.xlabel('Katsayı Değeri')
plt.ylabel('Özellik')
plt.title('Logistic Regresyon Modeli Özellik Katsayıları')
plt.grid(True)
plt.show()
