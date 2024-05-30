import streamlit as st
from functions import new_line
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from catboost import CatBoostClassifier
import sklearn
from collections import Counter

df = pd.read_csv('df_final.csv')

# Function to display page content based on user selection
def show_page(page):
    
    if page == "Home":
        st.markdown('# Player Transfer Prediction in Football')
        st.markdown("### This is Streamlit Application by Anar Abiyev and Jose Burgos for Advanced Data Analysis Methods Laboratory")
    
        st.write('<span style="font-size:20px;">We want to express out gradutate to **Dr. Toka** and **Dr. Rahimian** for there support and supervision throughout the semester.</span>', unsafe_allow_html=True)

        new_line()

        st.markdown("##### You can navigate to EDA and Application pages.")
        st.write('<span style="font-size:16px;">EDA page contains various interactive graph options to get familiar with the dataset.</span>', unsafe_allow_html=True)
        st.write('<span style="font-size:16px;">Application page helps the user to enter player attributes and see the prediction transfer.</span>', unsafe_allow_html=True)
    
    elif page == "EDA":
        st.title("Exploratory Data Analysis")

        st.markdown("##### Dataset:")
        st.write(df)


        value_counts = df['League'].value_counts()
        percentages = (value_counts / len(df)) * 100

        # Create Plotly Express bar chart
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            labels={'x': 'Classes', 'y': 'Counts'},
            title='Value Counts of Classes:'
        )

        # Add percentage annotations
        for i, (count, percentage) in enumerate(zip(value_counts, percentages)):
            fig.add_annotation(
                x=i,
                y=count,
                text=f"{percentage:.1f}%",
                showarrow=False,
                yshift=10
            )

        # Display the plot in Streamlit
        st.plotly_chart(fig)


        value_counts = df['Country'].value_counts()
        percentages = (value_counts / len(df)) * 100

        # Create Plotly Express bar chart
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            labels={'x': 'Countries', 'y': 'Counts'},
            title='Value Counts of Countries:',
            text=[f"{percentage:.1f}%" for percentage in percentages],  # Add percentage annotations
        )

        # Customize the chart
        fig.update_traces(marker_color='#943143', textposition='outside')
        fig.update_layout(
            xaxis_title='Countries',
            yaxis_title='Counts',
            xaxis={'categoryorder':'total descending'},
            width=1800,  # Width of the figure
            height=600,  # Height of the figure
        )

        # Rotate x-axis labels
        fig.update_xaxes(tickangle=90)

        # Display the plot in Streamlit
        st.plotly_chart(fig)

        leagues = ['Premier League', 'Seria A', 'La Liga', 'Bundesliga']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Using hex color codes

        # Create subplots
        fig = make_subplots(rows=4, cols=1, subplot_titles=leagues, vertical_spacing=0.1)

        # Add histograms for each league
        for i, (league, color) in enumerate(zip(leagues, colors), start=1):
            data = df[df['League'] == league]
            fig.add_trace(
                go.Histogram(
                    x=data['Price'],
                    nbinsx=10,
                    name=league,
                    marker_color=color
                ),
                row=i, col=1
            )
            fig.update_xaxes(
                title_text="Price", 
                row=i, col=1,
                categoryorder='category ascending'  # Ensuring the same order for x-axis
            )
            fig.update_yaxes(title_text="Count", row=i, col=1)

        # Update layout
        fig.update_layout(
            height=1000,  # Adjust height as needed
            width=1200,   # Increase width as needed
            title_text="Price Distribution by League",
            showlegend=False
        )
        fig.update_xaxes(tickangle=0)

        # Display the plot in Streamlit
        st.plotly_chart(fig)

        #fig = px.histogram(df, x = target_column)
        #c1, c2, c3 = st.columns([0.5, 2, 0.5])
        #c2.plotly_chart(fig)


        leagues = ['Premier League', 'Seria A', 'La Liga', 'Bundesliga']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Using hex color codes

        # Create a list to hold histograms for each league
        histograms = []

        # Add histogram traces for each league
        for league, color in zip(leagues, colors):
            data_league = df[df['League'] == league]['Price']
            histogram = go.Histogram(
                x=data_league,
                name=league,
                marker_color=color,
                opacity=0.7  # Adjust opacity for better visualization
            )
            histograms.append(histogram)

        # Create layout for the plot
        layout = go.Layout(
            title="Stacked Histogram of Prices by League",
            xaxis=dict(title="Price"),
            yaxis=dict(title="Count"),
            barmode='stack'  # Stacked bars
        )

        # Create the figure
        fig = go.Figure(data=histograms, layout=layout)

        # Display the plot in Streamlit
        st.plotly_chart(fig)
    

        leagues = ['Premier League', 'Seria A', 'La Liga', 'Bundesliga']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Using hex color codes

        # Create a list to hold histograms for each league
        histograms = []

        # Add histogram traces for each league
        for league, color in zip(leagues, colors):
            data_league = df[df['League'] == league]['Age']
            histogram = go.Histogram(
                x=data_league,
                name=league,
                marker_color=color,
                opacity=0.7  # Adjust opacity for better visualization
            )
            histograms.append(histogram)

        # Create layout for the plot
        layout = go.Layout(
            title="Stacked Histogram of Ages by League",
            xaxis=dict(title="Age"),
            yaxis=dict(title="Count"),
            barmode='stack'  # Stacked bars
        )

        # Create the figure
        fig = go.Figure(data=histograms, layout=layout)

        # Display the plot in Streamlit
        st.plotly_chart(fig)

    
        leagues = ['Premier League', 'Seria A', 'La Liga', 'Bundesliga']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Using hex color codes

        # Create a list to hold histograms for each league
        histograms = []

        # Add histogram traces for each league
        for league, color in zip(leagues, colors):
            data_league = df[df['League'] == league]['Position']
            histogram = go.Histogram(
                x=data_league,
                name=league,
                marker_color=color,
                opacity=0.7,  # Adjust opacity for better visualization
                nbinsx=len(df['Position'].unique())  # Adjust the number of bins based on unique positions
            )
            histograms.append(histogram)

        # Create layout for the plot
        layout = go.Layout(
            title="Stacked Histogram of Positions by League",
            xaxis=dict(title="Position"),
            yaxis=dict(title="Count"),
            barmode='stack'  # Stacked bars
        )

        # Create the figure
        fig = go.Figure(data=histograms, layout=layout)

        # Display the plot in Streamlit
        st.plotly_chart(fig)

    
    elif page == "Application":
        st.title("Application")
        st.write('<span style="font-size:20px;">In this page, you can find the details and results of the modelling and try it for yourself.</span>', unsafe_allow_html=True)
        new_line()
        st.write('The approach is to use five models in an ensemble voting method:')
        for item in ['SVM', 'Random Forest', 'Gradient Boosting', 'kNN', 'catboost']:
            st.markdown(f"- {item}")
        new_line(2)
        st.markdown('---')
        new_line(2)
        st.write('The following results have been achived. 75% and 78% in Premier League and Seria A are remarkable. The model has been able to generalize the other classes poorly because of lack of data samples.')
        st.image("results.png", use_column_width=True)

        new_line()
        st.write("We have used the model to predict this summer's transfers:")
        st.image("predictions.jpg", use_column_width=True)
        
        new_line(2)
        st.markdown('---')
        new_line(2)

        st.markdown('##### Try the model by giving input:')

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input('Age:')
        with col2:
            height = st.number_input('Height in cm:')
        with col3:
            weight = st.number_input('Weight in kg:')
       
            
        
        
        col1, col2, col3 = st.columns(3)
        with col1:
            country = st.selectbox("Country:", ["Italy", 'England', 'France', 'Germany','Spain',
                                                      'Belgium', 'Switzerland', 'Portugal', 'Austria', 'Poland',
                                                      'Greece', 'USA', 'Japan', 'Latin American', 'African',
                                                      'Scandinavian', 'Balkan', 'Britanian', 'Others'])
        with col2:
            position = st.selectbox('Positon:', ['ATT', 'MID', 'DEF'])
        with col3:
            preferred_foot = st.selectbox('Preferred Foot:', ['Left', 'Right'])


        st.write('Give values for the following parameters in the scale of 1 to 5 (1 is worst, 5 is best):')
        
        
        col1, col2, col3 = st.columns(3)
        with col1:
            intrep = st.number_input("International Reputation:", min_value=1, max_value=5, value=1, step=1)
        with col2:
            crossing = st.number_input("Crossing:", min_value=1, max_value=5, value=1, step=1)
        with col3:
            finishing = st.number_input("Finishing", min_value=1, max_value=5, value=1, step=1)


        col1, col2, col3 = st.columns(3)
        with col1:
            heading_accuracy = st.number_input("header:", min_value=1, max_value=5, value=1, step=1)
        with col2:
            short_passing = st.number_input("Short Pass:", min_value=1, max_value=5, value=1, step=1)
        with col3:
            volleys = st.number_input("Volleys:", min_value=1, max_value=5, value=1, step=1)

        col1, col2, col3 = st.columns(3)
        with col1:
            dribbling = st.number_input("Dribbling:", min_value=1, max_value=5, value=1, step=1)
        with col2:
            fk_accuracy = st.number_input("Free Kick:", min_value=1, max_value=5, value=1, step=1)
        with col3:
            long_passing = st.number_input("Long Pass:", min_value=1, max_value=5, value=1, step=1)

        col1, col2, col3 = st.columns(3)
        with col1:
            ball_control = st.number_input("Ball control:", min_value=1, max_value=5, value=1, step=1)
        with col2:
            acceleration = st.number_input("Acceleration:", min_value=1, max_value=5, value=1, step=1)
        with col3:
            sprint_speed = st.number_input("Sprint:", min_value=1, max_value=5, value=1, step=1)

        col1, col2, col3 = st.columns(3)
        with col1:
            agility = st.number_input("Agility:", min_value=1, max_value=5, value=1, step=1)
        with col2:
            reactions = st.number_input("Reactions:", min_value=1, max_value=5, value=1, step=1)
        with col3:
            balance = st.number_input("Balance:", min_value=1, max_value=5, value=1, step=1)

        col1, col2, col3 = st.columns(3)
        with col1:
            shot_power = st.number_input("Shot power:", min_value=1, max_value=5, value=1, step=1)
        with col2:
            stamina = st.number_input("Stamina:", min_value=1, max_value=5, value=1, step=1)
        with col3:
            strength = st.number_input("Strength:", min_value=1, max_value=5, value=1, step=1)

        col1, col2, col3 = st.columns(3)
        with col1:
            long_shots = st.number_input("Long shots:", min_value=1, max_value=5, value=1, step=1)
        with col2:
            aggression = st.number_input("Aggression:", min_value=1, max_value=5, value=1, step=1)
        with col3:
            interceptions = st.number_input("Interceptions:", min_value=1, max_value=5, value=1, step=1)

        col1, col2, col3 = st.columns(3)
        with col1:
            positioning = st.number_input("Positioning:", min_value=1, max_value=5, value=1, step=1)
        with col2:
            vision = st.number_input("Vision:", min_value=1, max_value=5, value=1, step=1)
        with col3:
            penalties = st.number_input("Penalties:", min_value=1, max_value=5, value=1, step=1)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            marking = st.number_input("Marking:", min_value=1, max_value=5, value=1, step=1)
        with col2:
            standing_tackle = st.number_input("Standing tackle:", min_value=1, max_value=5, value=1, step=1)
        with col3:
            sliding_tackle = st.number_input("Sliding tackle:", min_value=1, max_value=5, value=1, step=1)
        
        col1, col2 = st.columns(2)
        with col1:
            curve = st.number_input("Curve:", min_value=1, max_value=5, value=1, step=1)
        with col2:
            jumping = st.number_input("Jump:", min_value=1, max_value=5, value=1, step=1)

        col1, col2, col3 = st.columns(3)
        with col1:
            gk_diving = st.number_input("Goal keeper diving:", min_value=1, max_value=5, value=1, step=1)
        with col2:
            gk_handling = st.number_input("Goal keeper handling:", min_value=1, max_value=5, value=1, step=1)
        with col3:
            gk_kicking = st.number_input("Goal keeper kicking:", min_value=1, max_value=5, value=1, step=1)
        
        col1, col2 = st.columns(2)
        with col1:
            gk_positioning = st.number_input("Goal keeper positioning:", min_value=1, max_value=5, value=1, step=1)
        with col2:
            gk_reflexes = st.number_input("Goal keeper reflexes:", min_value=1, max_value=5, value=1, step=1)


        attributes = np.array([age, height, weight, intrep, crossing, finishing, heading_accuracy, short_passing, volleys,
                       dribbling, curve, fk_accuracy, long_passing, ball_control,
                       acceleration, sprint_speed, agility, reactions, balance,
                       shot_power, jumping, stamina, strength, long_shots,
                       aggression, interceptions, positioning, vision, penalties,
                       marking, standing_tackle, sliding_tackle, gk_diving,
                       gk_handling, gk_kicking, gk_positioning, gk_reflexes])
        
        countries = np.zeros(19)

        if country == 'Austria':
            countries[0] = 1
        elif country == 'Balkan':
            countries[1] = 1
        elif country == 'Belgium':
            countries[2] = 1
        elif country == 'Britania':
            countries[3] = 1
        elif country == 'England':
            countries[4] = 1
        elif country == 'France':
            countries[5] = 1
        elif country == 'Germany':
            countries[6] = 1
        elif country == 'Greece':
            countries[7] = 1
        elif country == 'Italy':
            countries[8] = 1
        elif country == 'Japan':
            countries[9] = 1
        elif country == 'Latin America':
            countries[10] = 1
        elif country == 'Netherlands':
            countries[11] = 1
        elif country == 'Others':
            countries[12] = 1
        elif country == 'Poland':
            countries[13] = 1
        elif country == 'Portugal':
            countries[14] = 1
        elif country == 'Scandinavian':
            countries[15] = 1
        elif country == 'Spain':
            countries[16] = 1
        elif country == 'Switzerland':
            countries[17] = 1
        elif country == 'United States':
            countries[18] = 1


        positions = np.zeros(2)
        if position == 'DEF':
            positions[0] = 1
        elif positions == 'MID':
            position[1] = 1

        if preferred_foot == 'Right':
            x_input = np.concatenate([attributes, countries, positions, np.array([1])], axis = 0)
        else:
            x_input = np.concatenate([attributes, countries, positions, np.array([0])], axis = 0)


        scaler = StandardScaler()
        x_input = scaler.fit_transform(x_input.reshape(1, -1))
        st.markdown(
            """
            <style>
            .css-1syj8dg-button-StyledButton {
                padding: 20px 20px; /* Adjust padding to increase button size */
                font-size: 16px; /* Adjust font size */
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        if st.button('Predict'):
            #model1 = joblib.load('models/model_gr_boost.pkl')
            #model2 = joblib.load('models/model_knn.pkl')
            #model3 = joblib.load('models/model_random_frst.pkl')
            model4 = joblib.load('models/model_svm.pkl') 
            model5 = CatBoostClassifier()
            model5.load_model('models/model_catboost.cbm')

            models = [model4, model5]
            answers = []
            for model in models:
                answers.append(model.predict(x_input))

            counter = Counter(answers[0])
            most_common_element, frequency = counter.most_common(1)[0]

            if most_common_element == 0:
                st.write('<span style="font-size:20px;">Predicted League: Premier League</span>', unsafe_allow_html=True)
            if most_common_element == 1:
                st.write('<span style="font-size:20px;">Predicted League: Seria A</span>', unsafe_allow_html=True)
            if most_common_element == 2:
                st.write('<span style="font-size:20px;">Predicted League: Bundesliga</span>', unsafe_allow_html=True)
            if most_common_element == 3:
                st.write('<span style="font-size:20px;">Predicted League: La Liga</span>', unsafe_allow_html=True)


# Sidebar navigation
page_options = ["Home", "EDA", "Application"]
selected_page = st.sidebar.radio("", page_options)

show_page(selected_page)
