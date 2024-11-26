import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

# Set page config to remove the default Streamlit title
st.set_page_config(
    page_title="F1 Race Predictor 2024",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS to hide the Streamlit app header
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Function to load the model from a pickle file
@st.cache_resource
def load_model():
    model_path = 'random_forest_grid_search.pkl'
    try:
        with open(model_path, 'rb') as file:
            grid_search_cv = pickle.load(file)
        return grid_search_cv.best_estimator_
    except Exception as e:
        st.error(f"Failed to load the model. Error: {e}")
        return None

# Function to load the race data
@st.cache_data
def load_data():
    data = pd.read_csv('f1_predictions_with_race_specific_gnn.csv', encoding='ISO-8859-1')
    return data

# Function to filter the dataframe for the selected race and calculate the top 3 drivers
def get_top_3_drivers(selected_race, df):
    race_df = df[df['race'] == selected_race]
    top_3_drivers = race_df.nlargest(3, 'gnn_predictions')[['Driver', 'gnn_predictions']]
    return top_3_drivers

# Function to create the race graph
def create_race_graph(df, race_name):
    G = nx.Graph()

    # Filter the dataframe for the specific race
    race_df = df[df['race'] == race_name]

    # Add Race node
    G.add_node(race_name, type='Race', weather=race_df['Weather_Conditions_Dry'].iloc[0])

    # Add Driver nodes and connect to Race
    for _, row in race_df.iterrows():
        driver_id = f"Driver_{row['Driver']}"
        G.add_node(driver_id, type='Driver',
                   name=row['Driver'],
                   age=row['age'],
                   years_in_f1=row['years_in_f1'],
                   starting_grid_position=row['starting_grid_position'],
                   points_in_previous_race=row['points_in_previous_race'],
                   gnn_predictions=row['gnn_predictions'])
        G.add_edge(driver_id, race_name)

    # Add Constructor nodes and connect to Drivers
    for _, row in race_df.iterrows():
        constructor_id = f"Constructor_{row['engine_manufacturer']}"
        if constructor_id not in G:
            G.add_node(constructor_id, type='Constructor',
                       engine_manufacturer=row['engine_manufacturer'])
        G.add_edge(f"Driver_{row['Driver']}", constructor_id)

    return G

# Function to visualize the race graph in 3D
def visualize_race_graph_3d(G, race_name):
    pos = nx.spring_layout(G, dim=3)

    edge_trace = []
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_trace.append(go.Scatter3d(x=[x0, x1], y=[y0, y1], z=[z0, z1],
                                       line=dict(width=0.5, color='#888'),
                                       mode='lines',
                                       hoverinfo='text',
                                       text=f'{edge[0]} - {edge[1]}',
                                       showlegend=False))

    node_x, node_y, node_z, node_text, node_color = [], [], [], [], []
    gnn_predictions = []  # Store the GNN predictions for color scaling

    for node in G.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)

        hover_text = f"Node: {node}<br>"
        for key, value in G.nodes[node].items():
            hover_text += f"{key}: {value}<br>"
        node_text.append(hover_text)

        if G.nodes[node]['type'] == 'Driver':
            # Use GNN prediction for the color gradient
            gnn_prediction = G.nodes[node].get('gnn_predictions', 0)
            gnn_predictions.append(gnn_prediction)
            node_color.append(gnn_prediction)  # Use GNN prediction for color
        else:
            gnn_predictions.append(0)  # Default for race and constructor nodes
            node_color.append(0)  # Set color to 0 for race and constructor nodes

    # Normalize GNN predictions for color mapping
    norm_pred = (gnn_predictions - np.min(gnn_predictions)) / (np.max(gnn_predictions) - np.min(gnn_predictions))

    # Use node GNN prediction for color mapping
    node_trace = go.Scatter3d(x=node_x, y=node_y, z=node_z,
                              mode='markers',
                              hoverinfo='text',
                              text=node_text,
                              marker=dict(
                                  showscale=True,
                                  colorscale='YlGnBu',  # Gradient color scale
                                  color=norm_pred,  # Color based on normalized GNN prediction
                                  size=10,  # Fixed size for driver nodes
                                  colorbar=dict(
                                      thickness=15,
                                      title='GNN Prediction',
                                      xanchor='left',
                                      titleside='right'
                                  ),
                                  line_width=2))

    # Manually set the race node color to black
    race_node_x = [pos[race_name][0]]
    race_node_y = [pos[race_name][1]]
    race_node_z = [pos[race_name][2]]
    race_node_trace = go.Scatter3d(x=race_node_x, y=race_node_y, z=race_node_z,
                                     mode='markers',
                                     marker=dict(size=15, color='orange'),  # Race node color
                                     hoverinfo='text',
                                     text=f"Race: {race_name}")

    # Constructor nodes trace
    constructor_node_x, constructor_node_y, constructor_node_z = [], [], []
    for node in G.nodes():
        if G.nodes[node]['type'] == 'Constructor':
            x, y, z = pos[node]
            constructor_node_x.append(x)
            constructor_node_y.append(y)
            constructor_node_z.append(z)

    constructor_node_trace = go.Scatter3d(x=constructor_node_x, y=constructor_node_y, z=constructor_node_z,
                                             mode='markers',
                                             marker=dict(size=10, color='red'),  # Constructor node color
                                             hoverinfo='text',
                                             text=[f"Constructor: {G.nodes[node]['engine_manufacturer']}" for node in G.nodes() if G.nodes[node]['type'] == 'Constructor'])

    data = [node_trace, race_node_trace, constructor_node_trace] + edge_trace

    layout = go.Layout(
        title=f"F1 Race Graph: {race_name}",
        showlegend=False,  # Remove the extra legend
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=0),
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        )
    )

    fig = go.Figure(data=data, layout=layout)
    return fig

# Define app functions for each page
def race_graph_page():
    st.title('2024 F1 Race Predictions - Race Graph')
    # st.image("homepage.jpg")
    st.write("""
             Welcome to the 2024 F1 Race Predictions app. Here you can select a specific race to see 
             an interactive 3D graph representing the race, drivers, and constructors. The graph shows 
             the relationships between these elements and color-codes drivers based on their predicted 
             probability of winning using a Graph Neural Network (GNN) model.
             """)
    
    df = load_data()

    # Select a unique list of races to choose from
    races = df['race'].unique()
    selected_race = st.selectbox('Select a Race', races)

    if st.button('Show Race Graph'):
        G = create_race_graph(df, selected_race)
        fig = visualize_race_graph_3d(G, selected_race)
        st.plotly_chart(fig, use_container_width=True)

def top_3_drivers_page():
    st.title('2024 F1 Race Predictions - Top 3 Drivers')
    # st.image("homepage.jpg")
    st.write("""
             Here you can select a specific race to see the top 3 predicted drivers based on our Graph Neural Network (GNN) model's 
             predictions. Simply choose a race from the dropdown below and click "Show Top 3 Drivers" 
             to view the predictions and see the corresponding bar chart displaying their probability of winning.
             """)
    
    df = load_data()

    # Select a unique list of races to choose from
    races = df['race'].unique()
    selected_race = st.selectbox('Select a Race', races)

    if st.button('Show Top 3 Drivers'):
        top_3_drivers = get_top_3_drivers(selected_race, df)
        st.write(top_3_drivers)
        
        # Plotting the bar chart
        st.write("Bar Chart of Top 3 Drivers' Winning Probabilities (GNN Predictions)")
        fig, ax = plt.subplots()
        ax.bar(top_3_drivers['Driver'], top_3_drivers['gnn_predictions'], color='red')
        plt.xlabel('Driver')
        plt.ylabel('GNN Prediction (Probability of Winning)')
        plt.title('Top 3 Drivers GNN Prediction Probability')
        st.pyplot(fig)

def predict_driver_page():
    st.title('2024 F1 Race Predictions - Driver Winning Probability')
    # st.image("homepage.jpg")

    model = load_model()
    if model is None:
        st.stop()

    df = pd.read_csv('2024_Races_with_predictions_full_streamlit.csv', encoding='ISO-8859-1')

    # User input for features
    st.subheader("Enter Driver and Race Details")

    races = df['race'].unique()
    selected_race = st.selectbox('Select a Race', races)

    drivers = df['Driver'].unique()
    selected_driver = st.selectbox('Select a Driver', drivers)

    starting_grid_position = st.number_input('Starting Grid Position', min_value=1, max_value=20, value=1)
    points_in_previous_race = st.number_input('Points in Previous Race', min_value=0, max_value=26, value=0)
    laps_in_previous_race = st.number_input('Laps in Previous Race', min_value=0, max_value=100, value=0)

    if st.button('Predict Winning Probability'):
        # Prepare the input data for prediction
        input_data = df[(df['race'] == selected_race) & (df['Driver'] == selected_driver)].iloc[0].copy()
        input_data['starting_grid_position'] = starting_grid_position
        input_data['points_in_previous_race'] = points_in_previous_race
        input_data['laps_in_previous_race'] = laps_in_previous_race

        # Remove unnecessary columns
        columns_to_drop = ['index', 'year', 'race', 'Driver', 'prediction_probability', 'predictions']
        input_data = input_data.drop(columns_to_drop)

        # Convert input_data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Check if the model has a preprocessing step
        if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
            input_processed = model.named_steps['preprocessor'].transform(input_df)
        else:
            for col in model.feature_names_in_:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[model.feature_names_in_]
            input_processed = input_df

        # Make prediction
        prediction_probability = model.predict_proba(input_processed)[0][1]

        st.subheader(f"Prediction Result for {selected_driver} at {selected_race}")
        st.write(f"The probability of {selected_driver} winning the {selected_race} is: {prediction_probability:.2%}")

        # Visualize the prediction using a Gauge (Dial) Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction_probability * 100,
            title={'text': f"{selected_driver}'s Winning Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 20], 'color': "lightgray"},
                    {'range': [20, 40], 'color': "lightgreen"},
                    {'range': [40, 60], 'color': "yellow"},
                    {'range': [60, 80], 'color': "orange"},
                    {'range': [80, 100], 'color': "red"}
                ],
            }
        ))

        st.plotly_chart(fig)

# Sidebar navigation
selection = st.sidebar.radio("Select Output", ["Race Graph", "Top 3 Drivers", "Predict Driver"])

# Page selection logic
if selection == "Race Graph":
    race_graph_page()
elif selection == "Top 3 Drivers":
    top_3_drivers_page()
else:
    predict_driver_page()