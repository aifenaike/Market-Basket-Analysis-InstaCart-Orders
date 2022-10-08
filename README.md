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
├── Modeling_ensemble.ipynb                     : Predict customer reorder tendencies using CatBoost
├── utils.py                                    : Script containing helper functions for memory optimization and EDA
├── modeling_utils.py                           : Script containing utility functions for modeling
├── feature_extraction.py                       : Feature engineering and extraction for a ML model
├── preprocessing.py                            : Data preparation for modeling
├── LICENSE                                     : License
└── README.md                                   : Project Report 

```
<br />

## Customer Segmentation and Profiling

Customer segmentation is the practice of classifying customers into groups based on shared traits so that businesses may effectively and appropriately market to each group. Utilizing information on the things that consumers purchase, we can segment the population. I used aisles that symbolize different product categories because there are thousands of customers and hundreds of thousands of products.

As KMeans does not work well on higher dimensions, I then used principal component analysis to minimize the number of dimensions. I performed KMeans clustering using 10 major components. With the use of the Elbow approach, I determined that five clusters was the ideal quantity.

<p align="center">
  <img width="600" height="300" src="https://github.com/aifenaike/Market-Basket-Analysis-InstaCart-Orders/blob/main/figure/Optimal_k.png">
</p>

The clustering can be visualized along first two principal components as below.

<p align="center">
  <img width="600" height="400" src="https://github.com/aifenaike/Market-Basket-Analysis-InstaCart-Orders/blob/main/figure/Cluster_Segments.png">
</p>

The clustering results into 5 customer cohorts and after checking most frequent products in them. I noticed some distinctions between the clusters:

- Customers in cohort 1 buy more `fresh vegetables` and `fruits` than the other clusters. As shown by absolute data, Cohort 1 is also the cluster including those customers buying far more goods than any others.

- Customers in cohort 2 buy more `chips pretzels` than people of the other clusters.

- Customers in cohort 3 buy the least amount of `packaged cheese` and `milk` than people of the other clusters. However, this cohort represents customers with the most purchase in `water seltzer sparkling water`.

- Absolute Data shows us customers in cohort 3 buy a lot of `soft drinks` which is not even listed in the top 8 products but mainly characterize this cluster. Coherently given that they also purchase `water seltzer sparkling water` than the others, I believe that they buy more consumable liquid products than the others.

- The mean orders for customers in cohort 5 are low compared to other clusters which tells us that either they are not frequent users of Instacart or they are new users and do not have many orders yet.

## Market Basket Analysis

A modeling technique called market basket analysis is based on the idea that you are either more or less likely to buy one group of goods after purchasing another. Market basket analysis may give the store details about a customer's purchasing habits. In addition to impacting sales promotions, loyalty programs, store layout, and discount schemes, this information can also be used for cross-selling and up-selling.

Market basket analysis examines the items that customers frequently purchase together and analyzes the data to determine which products should be promoted or marketed cross-sell. The phrase refers to the amount of groceries that supermarket patrons load into their trolleys when out shopping.

When attempting to identify a relationship between various items in a collection or identify recurring patterns in a transaction database, relational database, or other information repository, association rule mining is used.

The method used by major retailers like Amazon, Flipkart, and others to analyze customer purchasing patterns by identifying associations between the various items that customers place in their "shopping baskets" is known as market basket analysis, and it is the most popular method for discovering these patterns. The identification of these relationships can assist merchants in creating marketing plans by providing information on the products that customers typically buy in tandem. The tactics could consist of:

- Changing the store layout according to trends
- Customers behavior analysis
- Catalog Design
- Cross marketing on online stores
- Customized emails with add-on sales, etc.

### Metrics

**Support** : Its the default popularity of an item. In mathematical terms, the support of item A is the ratio of transactions involving A to the total number of transactions.

**Confidence** : Likelihood that customer who bought both A and B. It is the ratio of the number of transactions involving both A and B and the number of transactions involving B.
- Confidence(A => B) = Support(A, B)/Support(B)

**Lift** : Increase in the sale of A when you sell B.
- Lift(A => B) = Confidence(A, B)/Support(B)
      
- Lift (A => B) = 1 means that there is no correlation within the itemset.
- Lift (A => B) > 1 means that there is a positive correlation within the itemset, i.e., products in the itemset, A, and B, are more likely to be bought together.
- Lift (A => B) < 1 means that there is a negative correlation within the itemset, i.e., products in itemset, A, and B, are unlikely to be bought together.

**Apriori Algorithm:**  Apriori algorithm assumes that any subset of a frequent itemset must be frequent. It is the underlying Market Basket Analysis algorithm used in this project. Let's say that a transaction that has "Apples, Mango, Grapes" also contains "Grapes, Mango." Therefore, if "Grapes, Apple, Mango" are often, then "Grapes, Mango" must likewise be frequent, according to the a priori principle.

I used the apriori method from the `Mlxtend` Python library to identify associations between the top 100 most common items, and as a result, 28 product pairs (a total of 56 rules) with lift greater than 1 were identified. Following are the top 10 product combos with the highest lift:


| Product A  | Product B | Lift |
| ------------- | ------------- | ---- |
| Limes  | Large Lemons  | 3 |
| Organic Strawberries | Organic Raspberries | 2.21 |
| Organic Avocado | Large Lemon | 2.12 |
| Organic Strawberries | Organic Blueberries | 2.11 |
| Organic Hass Avocado | Organic Raspberries | 2.08 |
| Banana | Organic Fuji Apple | 1.88 |
| Bag of Organic Bananas | Organic Raspberries | 1.83 |
| Organic Hass Avocado | Bag of Organic Bananas | 1.81 |
| Honeycrisp Apple | Banana | 1.77 |
| Organic Avocado | Organic Baby Spinach | 1.70 |


## ML Model to Predict Product Reorders

We can utilize this anonymized transactional data of customer orders over time to predict which previously purchased products will be in a user’s next order. This would help recommend the products to a user. 

 
 ### ML Models

Using the extracted features, I created a dataframe with all the products the user has previously purchased, user level level features, product level features, asile and department level features, user-product level features, and information about the current order like the day of the week, hour of the day, etc.  The Target variable would be 'reordered' which shows how many of the previously purchased items, user ordered a particular product.

 I used CatBoost to handle the enormous amounts of data, because it can be parallelized (ensemble), assigns feature priority, and follows a regular approach for generating models. In addition, I constructed a neural network to determine how well this model would perform while ignoring some inherent unpredictability in both of these models. I used cost-sensitive optimization to balance the performance evaluation by giving the evaluation metric class weights.

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
