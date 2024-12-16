#!/usr/bin/env python
# coding: utf-8

# In[5]:


df = pd.read_csv(r"C:\Users\Sowmi\Downloads\Data (1) (1).csv")


# In[97]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[7]:


df.head()


# In[9]:


df.info()


# In[11]:


df.describe()


# In[13]:


summary = df['product_wg_ton'].describe()

Q1 = summary['25%']
Q3 = summary['75%']
IQR = Q3 - Q1


# In[15]:


lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


# In[19]:


lower_bound


# In[21]:


upper_bound


# In[23]:


outliers = df[(df['product_wg_ton'] < lower_bound) | (df['product_wg_ton'] > upper_bound)]
print("Outliers:")
print(outliers)


# In[25]:


pd.set_option('display.max_columns',50)


# In[ ]:





# In[34]:


df.shape


# # PRE PRUNING

# In[30]:


df.drop(['WH_Manager_ID','Ware_house_ID'],axis = 1,inplace=True)


# In[69]:


#checking unique values
'''df['WH_Manager_ID'].nunique()
its showing error .. because i dropped that column'''


# In[32]:


df.sample()


# # HANDLING MISSING VALUE 

# In[36]:


df.isnull().sum()
#so here, approved_wh_govt_certificate has low missing value, we can use mode
#wh_est_year is not necessary , we can drop.


# In[38]:


df.drop(['wh_est_year'],axis=1,inplace = True)


# In[40]:


df.isnull().sum()


# In[42]:


df['workers_num'] = df['workers_num'].fillna(df['workers_num'].mean())


# In[86]:


df.isnull().sum()


# In[46]:


df['approved_wh_govt_certificate'] = df['approved_wh_govt_certificate'].fillna(df['approved_wh_govt_certificate'].mode().iloc[0])


# # checking for duplicates

# In[50]:


dup = df.duplicated().sum()


# In[52]:


dup


# # FEATURE ENGINEERING

# In[54]:


df.sample()


# In[56]:


df['WH_regional_zone']=df['WH_regional_zone'].apply(lambda x : x[-1])
#lambda returns only last character using 'slicing', ie... x[-1]


# In[58]:


df.sample()


# In[60]:


#still WH_regional_zone is in object datatype... inorder to change it into numeric.. we'll use following func.
df['WH_regional_zone']=pd.to_numeric(df['WH_regional_zone'])


# In[62]:


df.info()
#now.. it is converted into int64


# # Main aim is to encode object type items into numeric... so that we use "one hot encoding", which is we call in naive term 'dummies'

# In[64]:


obj_type = df.select_dtypes(include = 'object')
obj_type.columns


# # creating a list of objects

# In[66]:


obj_list = ['Location_type', 'WH_capacity_size', 'zone', 'wh_owner_type',
       'approved_wh_govt_certificate']


# # creating dummies ... ie., 'one hot encoding'

# In[88]:


df1 = pd.get_dummies(df,columns = obj_list,prefix = obj_list,drop_first = True ,dtype=np.uint8)


# In[76]:


df[obj_list] = df[obj_list].astype('object')


# In[80]:


df1 = df1.fillna('Unknown')


# In[84]:


df.info()


# In[90]:


df1


# In[94]:


df1.shape


# In[92]:


df1.info()


# # checking correlation with heatmap

# In[99]:


plt.figure(figsize=(20,20))
heatmap = sns.heatmap(df1.corr(),cbar = True,annot = True,square = True,fmt='.2f',annot_kws={'size': 10})
plt.title('Heatmap Correlation of all variables', fontsize = 20,fontweight = 100) # title with fontsize 20
plt.show()


# # before one hot encoding

# In[104]:


plt.figure(figsize=(20,20))
heatmap1 = sns.heatmap(df.corr(),cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10})
plt.title('Heatmap Correlation of all variables', fontsize = 20,fontweight = 100) # title with fontsize 20
plt.show()


# # using box plot, we can check outliers

# In[111]:


sns.boxplot(data=df1, orient = 'h' )


# In[113]:


for column in df1.columns:
    sns.boxplot(df1[column])
    plt.title(column)
    plt.show()


# # EDA

# In[120]:


plt.figure(figsize = (10,4))
sns.barplot(x = 'storage_issue_reported_l3m', y = 'product_wg_ton',data = df1)
plt.title('Storage Issues Vs Production')
#more storage issue , then more production.


# In[126]:


plt.figure(figsize = (10,6))
sns.barplot(x = 'transport_issue_l1y', y = 'product_wg_ton',data = df1)
plt.title('Transport issue VS Production')
#more transport issue, then less production.


# In[128]:


plt.figure(figsize = (10,5))
sns.barplot(x = 'temp_reg_mach', y = 'product_wg_ton',data = df1)
plt.title('Temperature Regulation Vs Production')


# In[130]:


plt.figure(figsize = (10,5))
sns.barplot(x = 'wh_breakdown_l3m', y = 'product_wg_ton',data = df1)
plt.title('WH Breakdown Vs Production')


# # dependent n independent variable split

# In[139]:


x = df1.drop(['product_wg_ton'],axis = 1)


# In[141]:


y = df1['product_wg_ton']


# # train n test

# In[152]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)


# # Standardization

# In[150]:


from sklearn.preprocessing import StandardScaler


# In[156]:


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[160]:


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor


# In[166]:


def evaluate_model(true,predicted):
    mse = mean_squared_error(true,predicted)
    sqrt_mse = np.sqrt(mean_squared_error(true,predicted))
    r2_square = r2_score(true,predicted)
    return mse,sqrt_mse,r2_square
    


# In[172]:


models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "XGBRegressor": XGBRegressor(),
    "AdaBoost Regressor": AdaBoostRegressor()
}

model_list = []
r2_list= []


# In[174]:


for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(x_train, y_train) # Train model

    # Make predictions
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    # Evaluate Train and Test
    model_train_mae , model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)

    model_test_mae , model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

    
    print(list(models.keys())[i])
    model_list.append(list(models.keys())[i])
    
    print('Model performance for Training set')
    print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
    print("- R2 Score: {:.4f}".format(model_train_r2))
    
    print('Model performance for Test set')
    print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
    print("- R2 Score: {:.4f}".format(model_test_r2))
    r2_list.append(model_test_r2)
    
    print('='*35)
    print('\n')


# In[176]:


pd.DataFrame(list(zip(model_list, r2_list)), columns=['Model Name', 'R2_Score']).sort_values(by=["R2_Score"],ascending=False)


# In[180]:


rf = RandomForestRegressor()
rf.fit(x_train, y_train) # Train model

# Make predictions
y_train_pred = rf.predict(x_train)
y_test_pred = rf.predict(x_test)

# Evaluate Train and Test dataset
model_train_mae , model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)

model_test_mae , model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)



print('Model performance for Training set')
print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
print("- R2 Score: {:.4f}".format(model_train_r2))

print('Model performance for Test set')
print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
print("- R2 Score: {:.4f}".format(model_test_r2))

print('='*35)
print('\n')


# In[182]:


plt.scatter(y_test,y_test_pred);
plt.xlabel('Actual');
plt.ylabel('Predicted');


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




