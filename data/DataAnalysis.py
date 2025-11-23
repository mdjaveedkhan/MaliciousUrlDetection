#importing required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

def dataAnalysis():
    # Loading data into dataframe

    data = pd.read_csv("phishing.csv")

    data = data.drop(['Index'], axis=1)
    plt.figure(figsize=(15, 15))
    sns.heatmap(data.corr(), annot=True)
    plt.savefig('static/vis/FeatCorr.jpg')
    plt.clf()

    # Phishing Count in pie chart

    data['class'].value_counts().plot(kind='pie', autopct='%1.2f%%')
    plt.title("Phishing Count")
    plt.savefig('static/vis/Phiscnt.jpg')
    plt.clf()

    # pairplot for particular features

    df = data[['PrefixSuffix-', 'SubDomains', 'HTTPS', 'AnchorURL', 'WebsiteTraffic', 'class']]
    sns.pairplot(data=df, hue="class", corner=True);
    plt.savefig('static/vis/pairplot.jpg')
    plt.clf()

#dataAnalysis()


