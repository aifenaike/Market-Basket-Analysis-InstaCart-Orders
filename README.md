# Instacart Market Basket Analysis

## Introduction

Instacart is an American technology company that operates grocery delivery and pick up service in the U.S. and Canada. Customers shop for groceries through the Instacart mobile app or [Instacart.com](https://www.instacart.com/) from various retailer partners. The order is shopped and delivered by an Instacart personal shopper.


### Objectives:
- Analyze the anonymized [data](https://www.kaggle.com/c/instacart-market-basket-analysis/data) of 3 million grocery orders from more than 200,000 Instacart users open sourced by Instacart 
- Find out hidden association between products for better cross-selling and upselling
- Perform customer segmentation for targeted marketing and anticipate customer behavior
- Build a Machine Learning model to predict which previously purchased product will be in user’s next order

### Business Questions
- Can we use association mining and machine learning to understand groceries purchase? 
- Can we predicting products that a user will buy again, try for the first time or add to cart next during a session?
- Can we optimize marketing and aisle packaging given the associative behaviours of customers for certain products?


### Project Organization
```
.
├── figures/                                    : Directory containing all plots 
├── Data Description and Overview.ipynb         : Initial analysis to understand data
├── Exploratory Data Analysis.ipynb             : EDA to analyze customer purchase pattern
├── Customers Segmentation and Profiling .ipynb    : Customer Segmentation based on product aisles
├── Market Basket Analysis.ipynb                : Market Basket Analysis to find products association
└── utils.py                                    : Script containing helper functions for memory optimization and EDA
└── modeling_utils.py                           : Script containing utility functions for modeling
├── feature_extraction.py                       : Feature engineering and extraction for a ML model
├── preprocessing.py                            : Data preparation for modeling
├── LICENSE                                     : License
└── README.md                                   : Project Report 

```
<br />
 
 ### ML Models

Using the extracted features, I created a dataframe with all the products the user has previously purchased, user level level features, product level features, asile and department level features, user-product level features, and information about the current order like the day of the week, hour of the day, etc.  The Target variable would be 'reordered' which shows how many of the previously purchased items, user ordered a particular product.

Due to the size of the dataframe, I downcast it to reduce memory usage and fit the data into my memory. Because StandardScaler needs 16 GB of RAM to function, I picked MinMaxScaler. I used CatBoost to handle the enormous amounts of data, because it can be parallelized (ensemble), assigns feature priority, and follows a regular approach for generating models. In addition, I constructed a neural network to determine how well this model would perform while ignoring some inherent unpredictability in both of these models. I used cost-sensitive optimization to balance the performance evaluation by giving the evaluation metric class weights.

By adjusting the threshold, I was able to maximize the F1 score. For model evaluation, I also used the AUC Score and logloss. Below is a performance comparison of both of these models utilizing the classification report, ROC curve, and confusion matrix. To comprehend significant features that aid in predicting product reorder, the feature critical plot from the CatBoost model is also shown. Both models perform almost equally well, with CatBoost slightly outperforming the other in terms of ROC-AUC.


 **CatBoost Model's Performance and Feature Importance:**

<p align="center">
  <img width="400" height="200" src="https://github.com/aifenaike/Market-Basket-Analysis-InstaCart-Orders/blob/main/figure/Screenshot%20(325).png">
</p>

<p align="center">
  <img width="600" height="300" src="https://github.com/aifenaike/Market-Basket-Analysis-InstaCart-Orders/blob/main/figure/ROC_Curve.png">
</p>

<p align="center">
  <img width="500" height="750" src="https://github.com/aifenaike/Market-Basket-Analysis-InstaCart-Orders/blob/main/figure/CatBoost%20Feature%20Importance%20Plot.png">
</p>
