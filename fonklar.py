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

# Verilen metin
metin = "Labour force participation rate, male (% ages 15 and older) (2021)"

# İstenilen kısımları alarak yeni bir kelime oluştur
yeni_kelime = metin.split(" ")[4] + " " + metin.split(" ")[-1]

# Yeni kelimeyi yazdır
print(yeni_kelime)
