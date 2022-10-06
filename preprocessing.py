#Import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
pd.options.mode.chained_assignment = None

#import utility functions from script
from utils import *

#Data file directory
root = 'C:/Users/ALEXANDER/Documents/Machine Learning/Instacart/Data/'

##Ignore Warning
import warnings
warnings.filterwarnings("ignore")


#Read datasets
aisles = read(root,path='aisles')
departments = read(root,path='departments')
orders = read(root,path='orders')
order_products_prior = read(root,path='order_products__prior')
order_products_train = read(root,path='order_products__train')
products = read(root,path="products")

product_features = pd.read_pickle(root + 'product_features.pkl')

user_features = pd.read_pickle(root + 'user_features.pkl')

user_product_features = pd.read_pickle(root + 'user_product_features.pkl')

orders = orders.astype({'order_id': np.int32,
                        'user_id': np.int64,
                        'eval_set': 'category',
                        'order_number': np.int16,
                        'order_dow': np.int8,
                        'order_hour_of_day': np.int8,
                        'days_since_prior_order': np.float32})


order_products_train = order_products_train.astype({'order_id': np.int32,
                                        'product_id': np.uint16,
                                        'add_to_cart_order': np.int16,
                                        'reordered': np.int8})

order_products_prior = order_products_prior.astype({'order_id': np.int32,
                                        'product_id': np.uint16,
                                        'add_to_cart_order': np.int16,
                                        'reordered': np.int8})


#Merge data with meta-data
Train_purchases = orders.merge(order_products_train, on = 'order_id', how = 'inner')

#Drop redundant columns
Train_purchases.drop(['eval_set', 'add_to_cart_order', 'order_id'], axis = 1, inplace = True)

#Ensure similar user_id in product features and Train purchase
place_holder_df = user_product_features[user_product_features['user_id'].isin(Train_purchases['user_id'].unique())]
place_holder_df = place_holder_df.merge(Train_purchases, on = ['user_id', 'product_id'], how = 'outer')


#Missing value imputation
place_holder_df['order_number'].fillna(place_holder_df.groupby('user_id')['order_number'].transform('mean'), inplace = True)
place_holder_df['order_dow'].fillna(place_holder_df.groupby('user_id')['order_dow'].transform('mean'), inplace = True)
place_holder_df['order_hour_of_day'].fillna(place_holder_df.groupby('user_id')['order_hour_of_day'].transform('mean'), inplace = True)
place_holder_df['days_since_prior_order'].fillna(place_holder_df.groupby('user_id')['days_since_prior_order'].\
                                                             transform('mean'), inplace = True)

place_holder_df = place_holder_df[place_holder_df.reordered != 0]

place_holder_df['reordered'].fillna(0, inplace = True)

#Merge product and user features
place_holder_df = place_holder_df.merge(product_features, on = 'product_id', how = 'left')
place_holder_df = place_holder_df.merge(user_features, on = 'user_id', how = 'left')

#save data
place_holder_df.to_pickle(root + 'final_dataset.pkl')

