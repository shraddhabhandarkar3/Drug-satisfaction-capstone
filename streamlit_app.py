
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

model = joblib.load('my_model.pkl')

data = pd.read_csv('Drug_clean.csv')

def user_input_features():
    condition = st.selectbox('Condition', options=data['Condition'].unique())
    drug = st.selectbox('Drug', options=data['Drug'].unique())
    form = st.selectbox('Form', options=data['Form'].unique())
    indication = st.selectbox('Indication', options=data['Indication'].unique())
    type_ = st.selectbox('Type', options=data['Type'].unique())
    ease_of_use = st.slider('Ease of Use', 1.0, 5.0, float(data['EaseOfUse'].mean()))
    effective = st.slider('Effectiveness', 1.0, 5.0, float(data['Effective'].mean()))
    price = st.number_input('Price', value=float(data['Price'].mean()))
    reviews = st.number_input('Reviews', value=float(data['Reviews'].mean()))

    features = pd.DataFrame({
        'Condition': [condition],
        'Drug': [drug],
        'Form': [form],
        'Indication': [indication],
        'Type': [type_],
        'EaseOfUse': [ease_of_use],
        'Effective': [effective],
        'Price': [price],
        'Reviews': [reviews]
    })
    return features

def main():
    st.title("Drug Satisfaction Prediction App")

    input_df = user_input_features()

    if st.button('Predict Satisfaction'):
        # Get the prediction
        prediction = model.predict(input_df)
        st.write(f'Predicted Satisfaction Score: {prediction[0]}')

    st.write("## Data Analysis")
    st.write("### Distribution of Satisfaction Scores")
    fig, ax = plt.subplots()
    sns.histplot(data['Satisfaction'], kde=True, ax=ax)
    st.pyplot(fig)


if __name__ == "__main__":
    main()
