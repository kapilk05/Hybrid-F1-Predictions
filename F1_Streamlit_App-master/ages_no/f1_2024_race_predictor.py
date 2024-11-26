import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Function to load the data
@st.cache_data
def load_data():
    data = pd.read_csv('2024_Races_with_predictions_full_streamlit.csv', encoding='ISO-8859-1')
    return data

# Function to filter the dataframe for the selected race and calculate the top 3 drivers
def get_top_3_drivers(selected_race, df):
    race_df = df[df['race'] == selected_race]
    top_3_drivers = race_df.nlargest(3, 'prediction_probability')[['Driver', 'prediction_probability']]
    return top_3_drivers

# Define the app function to be called from the main Streamlit app
def app():
    st.title('2024 F1 Race Predictions')
    st.image("maxresdefault.jpg")
    st.write("""
             Welcome to the 2024 F1 Race Predictions app. Here you can select a specific race to see 
             the top 3 predicted drivers based on our model's prediction probability. Simply choose a race 
             from the dropdown below and click "Show Top 3 Drivers" to view the predictions and see the 
             corresponding bar chart displaying their probability of winning.
             """)
    
    df = load_data()

    # Select a unique list of races to choose from
    races = df['race'].unique()
    selected_race = st.selectbox('Select a Race', races)

    if st.button('Show Top 3 Drivers'):
        top_3_drivers = get_top_3_drivers(selected_race, df)
        st.write(top_3_drivers)
        
        # Plotting the bar chart
        st.write("Bar Chart of Top 3 Drivers' Winning Probabilities")
        fig, ax = plt.subplots()
        ax.bar(top_3_drivers['Driver'], top_3_drivers['prediction_probability'], color='skyblue')
        plt.xlabel('Driver')
        plt.ylabel('Probability of Winning')
        plt.title('Top 3 Drivers Prediction Probability')
        st.pyplot(fig)
