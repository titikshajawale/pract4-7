#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Dragon real state price predictor


# In[ ]:


Data Analytics I
Create a Linear Regression Model using Python/R to predict home prices using Boston Housing
Dataset (https://www.kaggle.com/c/boston-housing). The Boston Housing dataset contains
information about various houses in Boston through different parameters. There are 506 samples
and 14 feature variables in this dataset.
The objective is to predict the value of prices of the house using the given features


# In[2]:


import pandas as pd


# In[3]:


housing=pd.read_csv("C:/Users/titik/OneDrive/Documents/data.csv") # read csv file


# In[4]:


housing # display dataset


# In[5]:


housing.head() # showing first 5 reasults in data


# In[6]:


housing.tail()#Showing last 5 rows


# In[7]:


housing.info()#how maany entries to know missing data


# In[8]:


housing['CHAS']#for particular value count


# In[9]:


housing['CHAS'].value_counts()#particular value count


# In[10]:


housing.describe()#give min , max, statistics


# In[11]:


#in abouve result count gives us count of rows and ignor null value
#mean is average
#std is standard deviation
#25% means 25% data is less than that value
#75% means 75% data is less than that value
#max data


# In[12]:


get_ipython().run_line_magic('matplotlib', 'Inline')


# In[13]:


import matplotlib.pyplot as plt


# In[14]:


housing.hist(bins=50,figsize=(20,15))#figuresize


# In[15]:


#train-test splitting for to keep test set  side


# In[16]:


import numpy as np
def split_train_test(data,test_ratio):
    np.random.seed(42)
    # here moste often we take 42
    # we do this random seed becasuse if we print suffled then we will get same output all the time otherwise every time it gives different output after running to avoid this we added that
    #shuffled do the shuffling of the indices in random permutation of dta length
    shuffled=np.random.permutation(len(data))
    print(shuffled)
    test_set_size=int(len(data)* test_ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]        
    


# In[17]:


#train_set,test_set=split_train_test(housing,0.2)# here 0.2 is ratio
#test_ratio means data percent ..,,20 or 80 is good so 20 means 0.2 and 80 means 0.8


# In[18]:


#print(f"Rows in train set:{len(train_set)}\nRows in test set:{len(test_set)}\n")


# In[19]:


# here we will use 405 data entries for to train the data
#here we will use 101 entries for testing


# In[20]:


#split train_test funtion is already presnt in scikit learn


# In[21]:


from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)
print(f"Rows in train set:{len(train_set)}\nRows in test set:{len(test_set)}\n")


# In[22]:


from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing['CHAS']):
    #loc is pandas framework data functionn
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]


# In[23]:


strat_test_set


# In[24]:


strat_test_set.describe()


# In[25]:


strat_test_set.info()


# In[26]:


strat_test_set['CHAS'].value_counts()


# In[27]:


strat_train_set['CHAS'].value_counts()


# In[28]:


95/7


# In[29]:


376/28


# In[30]:


#Looking for Correlations


# In[31]:


corr_matrix=housing.corr()


# In[32]:


corr_matrix['MED'].sort_values(ascending=False)
#if MEDV is 1 then it is strong positive corelation


# In[33]:


from pandas.plotting import scatter_matrix


# In[34]:


attributes=["RM","MED","LST"]


# In[35]:


scatter_matrix(housing[attributes],figsize=(12,8))


# In[36]:


housing.plot(kind="scatter",x="RM",y="MED",alpha=0.8)


# In[37]:


#Trying out Atrribute contribution


# In[38]:


housing["TAXRM"]=housing["TAX"]/housing["RM"]


# In[39]:


housing["TAXRM"]


# In[40]:


housing.head()


# In[41]:


corr_matrix=housing.corr()
corr_matrix['MED'].sort_values(ascending=False)


# In[42]:


housing=strat_train_set.drop("MED",axis=1)
housing_labels=strat_train_set["MED"].copy()


# In[43]:


#missing attributes


# In[44]:


# take care the missing attribute , you have three option:
    #get rid of missing data points
    #get rid of the whole attributes
    # set the value to (0,mean , median)


# In[45]:


housing.dropna(subset=['RM']) #option 1


# In[46]:


housing.shape


# In[47]:


a=housing.dropna(subset=["RM"])
a.shape


# In[48]:


housing.drop("RM",axis=1)#option 2
#note that there is no RM column


# In[49]:


median=housing["RM"].median()


# In[50]:


housing["RM"]


# In[51]:


housing["RM"].fillna(median)
#note that the original housing dataframe will remain unchanged


# In[52]:


housing.shape


# In[53]:


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy='median')
imputer.fit(housing)


# In[54]:


imputer.statistics_


# In[55]:


imputer.statistics_.shape


# In[56]:


X=imputer.transform(housing)


# In[57]:


housing_tr=pd.DataFrame(X,columns=housing.columns)


# In[58]:


housing_tr.describe()


# In[59]:


#tr means transformed dataset


# In[60]:


#scikit learn design


# In[61]:


#primilary three types of the object
#1.estimators-->eg .Imputer .Estimates some parameters based on the dataset
#it has a fit method and transformer method .Fit method -Fits the datsets AND CALCUTAES INTERNAL PARAMEYTER
#2.transformers-->
#tAkes input and return output based on the learning from fit().
#it also has  conveninence function called fit_transform()
#which fits and then transforms.
#3.predictors-->LinearRegression model is an example of predictor.fit() and predict() are two common functions.It also gives score() function which will evalute the prediction


# In[62]:


#feature Scaling
#two types of feeature scaling method
#primarily two method
#1) Min-max Scaling (Normalisation)
#value-min/(max-min)
#Sklearn provides a class called MinMaxScaler for this 
#2)Standardization
#value-min/(std division) ...
#sklearn provides a class called standard scaler for this


# In[63]:


#pipeline


# In[64]:


from sklearn.pipeline import Pipeline


# In[65]:


from sklearn.preprocessing import StandardScaler


# In[66]:


my_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    #.....add as many as you want in pipeline
    ('std_scaler',StandardScaler())
])


# In[67]:


housing_num_tr=my_pipeline.fit_transform(housing_tr)


# In[68]:


housing_num_tr


# In[69]:


#selecting a desired model for g=dragon real estates


# In[70]:


from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor
#model=LinearRegression()
model=DecisionTreeRegressor()
model.fit(housing_num_tr,housing_labels)


# In[71]:


some_data=housing.iloc[:5]


# In[72]:


some_labels=housing_labels.iloc[:5]


# In[73]:


prepared_data=my_pipeline.transform(some_data)


# In[74]:


model.predict(prepared_data)


# In[75]:


list(some_labels)


# In[76]:


some_labels


# In[77]:


#Evaluating the modelling


# In[78]:


from sklearn.metrics import  mean_squared_error


# In[79]:


housing_predictions=model.predict(housing_num_tr)
mse=mean_squared_error(housing_labels,housing_predictions)
rmse=np.sqrt(mse)


# In[80]:


rmse


# In[81]:


#using better evaluation technique-cross validation


# In[82]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)


# In[83]:


rmse_scores


# In[84]:


def print_scores(scores):
    print("Scores are:",scores)
    print("Mean:",scores.mean())
    print("Standard Deviation:",scores.std())


# In[85]:


print_scores(rmse_scores)


# In[86]:


#Decision tree:
#Mean: 4.305817321499487
#Standard Deviation: 0.7865795038177691

#Linear Regression:
#Mean:
#Standard deviation:


# In[87]:


#from sklearn.linear_model import LinearRegression 
#from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model=LinearRegression()
#model=DecisionTreeRegressor()
model=RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)


# In[88]:


some_data=housing.iloc[:5]


# In[89]:


some_labels=housing_labels.iloc[:5]


# In[90]:


prepared_data=my_pipeline.transform(some_data)


# In[91]:


model.predict(prepared_data)


# In[92]:


list(some_labels)


# In[93]:


from sklearn.metrics import  mean_squared_error


# In[94]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)


# In[95]:


rmse_scores


# In[96]:


def print_scores(scores):
    print("Scores are:",scores)
    print("Mean:",scores.mean())
    print("Standard Deviation:",scores.std())


# In[97]:


print_scores(rmse_scores)


# In[98]:


#Convert this notebook into python file and runn the pipeline using visual studio cosde


# In[99]:


#saving the model


# In[100]:


from joblib import dump,load
dump(model,'Dragon.joblib')


# In[101]:


#Testing the model on test data


# In[102]:


x_test=strat_test_set.drop("MED",axis=1)
y_test=strat_test_set['MED'].copy()
x_test_prepared=my_pipeline.transform(x_test)
final_predictions=model.predict(x_test_prepared)
final_mse=mean_squared_error(y_test,final_predictions)
final_rmse=np.sqrt(final_mse)
print(final_predictions ,list(y_test))


# In[103]:


final_rmse


# In[ ]:


# here we have chosen random forest regressor model used because it gives less error


# In[105]:


prepared_data[0]


# In[106]:


from joblib import dump,load
import numpy as np
model=load('Dragon.joblib')
feature=np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.24127765, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])
model.predict(feature)


# In[ ]:




