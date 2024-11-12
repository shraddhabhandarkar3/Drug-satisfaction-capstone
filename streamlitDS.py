#!/usr/bin/env python
# coding: utf-8

# In[27]:


get_ipython().system('pip install streamlit')
get_ipython().system('pip install pyngrok==4.1.1')

import pandas as pd             
import numpy as np                  
import streamlit as st           
import matplotlib.pyplot as plt   
import seaborn as sns           
import requests                     
import xml.etree.ElementTree as ET  
import json  
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report    
import sqlite3    


# In[28]:


file_path = 'drug_clean.csv'


# In[29]:


data = pd.read_csv(file_path)
print(data.head())


# In[30]:


data.head()


# In[31]:


data.columns


# In[32]:


print(data.dtypes)
print(data.describe())
print(data.isnull().sum())


# In[33]:


X = data.drop('Satisfaction', axis=1)
y = data['Satisfaction']

categorical_features = ['Condition', 'Drug', 'Form', 'Indication', 'Type']
numerical_features = ['EaseOfUse', 'Effective', 'Price', 'Reviews']

numerical_transformer = 'passthrough'

categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

model = RandomForestRegressor(n_estimators=100, random_state=42)

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

my_pipeline.fit(X_train, y_train)

preds = my_pipeline.predict(X_test)

score = mean_squared_error(y_test, preds, squared=False)
print('MSE:', score)

import joblib
joblib.dump(my_pipeline, 'my_model.pkl')


# In[35]:


get_ipython().run_cell_magic('writefile', 'streamlit_app.py', '\nimport streamlit as st\nimport pandas as pd\nimport joblib\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.preprocessing import OneHotEncoder\nfrom sklearn.compose import ColumnTransformer\n\nmodel = joblib.load(\'my_model.pkl\')\n\ndata = pd.read_csv(\'Drug_clean.csv\')\n\ndef user_input_features():\n    condition = st.selectbox(\'Condition\', options=data[\'Condition\'].unique())\n    drug = st.selectbox(\'Drug\', options=data[\'Drug\'].unique())\n    form = st.selectbox(\'Form\', options=data[\'Form\'].unique())\n    indication = st.selectbox(\'Indication\', options=data[\'Indication\'].unique())\n    type_ = st.selectbox(\'Type\', options=data[\'Type\'].unique())\n    ease_of_use = st.slider(\'Ease of Use\', 1.0, 5.0, float(data[\'EaseOfUse\'].mean()))\n    effective = st.slider(\'Effectiveness\', 1.0, 5.0, float(data[\'Effective\'].mean()))\n    price = st.number_input(\'Price\', value=float(data[\'Price\'].mean()))\n    reviews = st.number_input(\'Reviews\', value=float(data[\'Reviews\'].mean()))\n\n    features = pd.DataFrame({\n        \'Condition\': [condition],\n        \'Drug\': [drug],\n        \'Form\': [form],\n        \'Indication\': [indication],\n        \'Type\': [type_],\n        \'EaseOfUse\': [ease_of_use],\n        \'Effective\': [effective],\n        \'Price\': [price],\n        \'Reviews\': [reviews]\n    })\n    return features\n\ndef main():\n    st.title("Drug Satisfaction Prediction App")\n\n    input_df = user_input_features()\n\n    if st.button(\'Predict Satisfaction\'):\n        # Get the prediction\n        prediction = model.predict(input_df)\n        st.write(f\'Predicted Satisfaction Score: {prediction[0]}\')\n\n    st.write("## Data Analysis")\n    st.write("### Distribution of Satisfaction Scores")\n    fig, ax = plt.subplots()\n    sns.histplot(data[\'Satisfaction\'], kde=True, ax=ax)\n    st.pyplot(fig)\n\n\nif __name__ == "__main__":\n    main()\n')

