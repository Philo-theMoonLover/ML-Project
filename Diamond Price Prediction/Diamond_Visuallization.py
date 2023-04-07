import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# https://www.kaggle.com/datasets/shivam2503/diamonds
datas = pd.read_csv('Diamond-Price-Prediction.csv')
print(datas.head())
print('--------------------------')
print(datas.Price.describe(percentiles=[0.25, 0.5, 0.75, 0.85, 0.90, 1]))

s = (datas.dtypes == "object")
object_cols = list(s[s].index)

# Make copy to avoid changing original data
label_data = datas.copy()
# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
for col in object_cols:
    label_data[col] = label_encoder.fit_transform(label_data[col])

# Correlation matrix
cmap = sns.diverging_palette(70,20,s=50, l=40, n=6,as_cmap=True)
corrmat = label_data.corr()
f, ax = plt.subplots(figsize=(12,12))
sns.heatmap(corrmat,cmap=cmap,annot=True)
sns.pairplot(datas)
plt.show()

plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.title('Price Distribution')
sns.histplot(datas.Price/100)
plt.subplot(1,2,2)
plt.title('Box Plot')
sns.boxplot(x=datas.Price/100)
plt.show()


plt.figure(figsize=(20,10))
plt.subplot(1,3,1)
plt1 = datas.Cut.value_counts().plot(kind = 'bar', color='c')
plt.title('Cut Histogram')
plt1.set(xlabel = 'Cut', ylabel = 'Frequency of Cut')
plt.subplot(1,3,2)
plt1 = datas.Color.value_counts().plot(kind = 'bar', color='c')
plt.title('Color Histogram')
plt1.set(xlabel = 'Color', ylabel = 'Frequency of Color')
plt.subplot(1,3,3)
plt1 = datas.Cut.value_counts().plot(kind = 'bar', color='c')
plt.title('Clarity Histogram')
plt1.set(xlabel = 'Clarity', ylabel = 'Frequency of Clarity')
plt.show()

ax = sns.regplot(x='Carat',y='Price', data=datas, fit_reg=True, scatter_kws={"color":"#a9a799"}, line_kws={"color": "#835656"})
ax.set_title("Regression line on Price vs Carat")
plt.show()
