#Import relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc

#import utility functions from script
from utils import *

#Data file directory
root = 'C:/Users/ALEXANDER/Documents/Machine Learning/Instacart/Data/'


#Read datasets from root directory
aisles = read(root,path='aisles')

departments = read(root, path ='departments')

order_products_prior = read(root,path='order_products__prior')

order_products_train = read(root,path = 'order_products__train')

products = read(root,path= 'products')

orders = read(root, path='orders')

#Correct data types of attributes
orders = orders.astype({ 'order_id': np.int32,
                        'user_id': np.int64,
                        'eval_set': 'category',
                        'order_number': np.int16,
                        'order_dow': np.int8,
                        'order_hour_of_day': np.int8,
                        'days_since_prior_order': np.float32})

order_products_prior = order_products_prior.astype({'order_id': np.int32,
                                'product_id': np.uint16,
                                'add_to_cart_order': np.int16,
                                'reordered': np.int8})

order_products_train = order_products_train.astype({'order_id': np.int32,
                            'product_id': np.uint16,
                            'add_to_cart_order': np.int16,
                            'reordered': np.int8})



#Merge meta-information on products and orders with prior purchase dataset
prior_order= order_products_prior.merge(orders, on ='order_id', how='inner')
prior_order = prior_order.merge(products, on = 'product_id', how = 'left')


#Feature1: The No of times a user buy the product
prior_order['user_buy_product_times'] = prior_order.groupby(['user_id', 'product_id']).cumcount() + 1


feat_agg_1 = {
    #Product's average add-to-cart-order
    'add_to_cart_order' : [('mean_add_to_cart_order','mean')], 
    
    #Reorder percentage of a product
    'reordered' : [('total_orders','count'),
                ('total_reorders','sum'),  #Total times the product was reordered
                ('reorder_percentage','mean')],  #Reorder percentage of a product
    # Total unique users of a product       
    'user_id': [('unique_users' ,lambda x: x.nunique())],
     
    
    'user_buy_product_times': [('order_first_time_total_cnt', lambda x: sum(x==1)), 
                                    ('order_second_time_total_cnt' ,lambda x: sum(x==2))],
    #Is the product Organic?      
    'product_name':[('is_organic', lambda x: 1 if 'Organic' in x else 0)]}


#Create new features using unique product_id as keys
prod_order = prior_order.groupby('product_id').agg(feat_agg_1)
#Drop index of subgroup
prod_order.columns = prod_order.columns.droplevel(0)
prod_order.reset_index(inplace = True)

# Percentage of users that buy the product second time
prod_order['second_time_percent'] = prod_order['order_second_time_total_cnt']/prod_order['order_first_time_total_cnt']

#Feature set II
feat_agg_2 = {
    
    #Mean and std of aisle add-to-cart-order
    'add_to_cart_order' : [('aisle_mean_add_to_cart_order','mean'),
                                   ('aisle_std_add_to_cart_order','std')],
    
    #Reorder percentage, Total orders and reorders of a product aisle
    'reordered' : [('aisle_total_orders','count'), ('aisle_total_reorders','sum'), ('aisle_reorder_percentage','mean')],

    #Aisle unique users       
    'user_id': [('aisle_unique_users' ,lambda x: x.nunique())]}

#Create new features using unique aisle_id as keys
aisle_order = prior_order.groupby('aisle_id').agg(feat_agg_2)
aisle_order.columns = aisle_order.columns.droplevel(0)
aisle_order.reset_index(inplace = True)


##Feature set III
feat_agg_3 = {
    #Mean and std of department add-to-cart-order

    'add_to_cart_order' : [('department_mean_add_to_cart_order','mean'),
                                   ('department_std_add_to_cart_order','std')],
    
     #Reorder percentage, Total orders and reorders of a product department      
    'reordered' : [('department_total_orders','count'), ('department_total_reorders','sum'),
                          ('department_reorder_percentage','mean')],
    
    #Department unique users       
    'user_id': [('department_unique_users' ,lambda x: x.nunique())]}

#Create new features using unique department_id as keys
dpt_order = prior_order.groupby('department_id').agg(feat_agg_3)
dpt_order.columns = dpt_order.columns.droplevel(0)
dpt_order.reset_index(inplace = True)

#Merge all newly created features and meta-data
prod_order = prod_order.merge(products, on = 'product_id', how = 'left')
prod_order = prod_order.merge(aisle_order, on = 'aisle_id', how = 'left')
prod_order = prod_order.merge(aisles, on = 'aisle_id', how = 'left')
prod_order = prod_order.merge(dpt_order, on = 'department_id', how = 'left')
prod_order = prod_order.merge(departments, on = 'department_id', how = 'left')

#drop inconsistent column names
prod_order.drop(['product_name', 'aisle_id', 'department_id'], axis = 1, inplace = True)

# free some memory
del aisle_order, dpt_order, aisles, departments
gc.collect()

# when no prior order, the value is null. Imputing as 0
prior_order['days_since_prior_order'] = prior_order['days_since_prior_order'].fillna(0)


#Feature set IV
feat_agg_4 = {
    #User's average and std day-of-week of order
    'order_dow': [('avg_dow','mean'), ('std_dow','std')],
    

    #User's average and std hour-of-day of order    
    'order_hour_of_day': [('avg_doh','mean'), ('std_doh','std')],
           

    #User's average and std days-since-prior-order 
    'days_since_prior_order': [('avg_since_order','mean'), ('std_since_order','std')],
           
    # Total orders by a user
    'order_number': [('total_orders_by_user', lambda x: x.nunique())],
    #Total products user has bought       
    'product_id': [('total_products_by_user', 'count'),
                         ('total_unique_product_by_user',lambda x: x.nunique())], #Total unique products user has bought

    #user's total reordered products
    'reordered': [('total_reorders_by_user','sum'), 
                        ('reorder_propotion_by_user','mean')]}# User's overall reorder percentage 


user_feats = prior_order.groupby('user_id').agg(feat_agg_4)
user_feats.columns = user_feats.columns.droplevel(0)
user_feats.reset_index(inplace = True)

#Feature set V

feat_agg_5 = {'reordered': [('average_order_size','count'), #Average order size of a user
                        ('reorder_in_order','mean')]} #User's mean of reordered items of all orders


user_feats2 = prior_order.groupby(['user_id', 'order_number']).agg(feat_agg_5)
user_feats2.columns = user_feats2.columns.droplevel(0)
user_feats2.reset_index(inplace = True)


user_feats3 = user_feats2.groupby('user_id').agg({'average_order_size' : 'mean', 
                                   'reorder_in_order':'mean'})
user_feats3 = user_feats3.reset_index()

user_feats = user_feats.merge(user_feats3, on = 'user_id', how = 'left')

#Last 3 orders of a customer
last_three_orders = user_feats2.groupby('user_id')['order_number'].nlargest(3).reset_index()
last_three_orders = user_feats2.merge(last_three_orders, on = ['user_id', 'order_number'], how = 'inner')
last_three_orders['rank'] = last_three_orders.groupby("user_id")["order_number"].rank("dense", ascending=True)


last_order_feats = last_three_orders.pivot_table(index = 'user_id', columns = ['rank'], \
                                                 values=['average_order_size', 'reorder_in_order']).\
                                                reset_index(drop = False)
last_order_feats.columns = ['user_id','orders_3', 'orders_2', 'orders_1', 'reorder_3', 'reorder_2', 'reorder_1']


user_feats = user_feats.merge(last_order_feats, on = 'user_id', how = 'left')

#Feature set VI
feat_agg_6 = {'reordered': [('total_product_orders_by_user','count'), 
                          ('total_product_reorders_by_user','sum'),
                          ('user_product_reorder_percentage', 'mean')],
            'add_to_cart_order': [('avg_add_to_cart_by_user','mean')],
            'days_since_prior_order':[('avg_days_since_last_bought','mean')],
            'order_number': [('last_ordered_in','max')]}

user_product_feats = prior_order.groupby(['user_id', 'product_id']).agg(feat_agg_6)
user_product_feats.columns = user_product_feats.columns.droplevel(0)
user_product_feats.reset_index(inplace = True)


last_orders = prior_order.merge(last_three_orders, on = ['user_id', 'order_number'], how = 'inner')
last_orders['rank'] = last_orders.groupby(['user_id', 'product_id'])['order_number'].rank("dense", ascending=True)
product_purchase_history = last_orders.pivot_table(index = ['user_id', 'product_id'],\
                                                   columns='rank', values = 'reordered').reset_index()


product_purchase_history.columns = ['user_id', 'product_id', 'is_reorder_3', 'is_reorder_2', 'is_reorder_1']
product_purchase_history.fillna(0, inplace = True)
user_product_feats = user_product_feats.merge(product_purchase_history, on=['user_id', 'product_id'], how = 'left')
user_product_feats.fillna(0, inplace = True)


#Save features 
prod_order.to_pickle(root + 'product_features.pkl')
user_feats.to_pickle(root +'user_features.pkl')
user_product_feats.to_pickle(root +'user_product_features.pkl')