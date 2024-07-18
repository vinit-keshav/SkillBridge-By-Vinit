
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, black, grey);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    @keyframes gradient {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }
    .title {
        background: linear-gradient(270deg, gray, black);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 5s ease infinite;
        font-size: 3em;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="title">SkillBridge by Vinit</h1>', unsafe_allow_html=True)

skills_df = pd.read_csv('Skills.csv')
employment_df = pd.read_csv('UAER.csv')
geojson_file = 'India_states.geojson'


le_industry = LabelEncoder()
skills_df['Industry'] = le_industry.fit_transform(skills_df['Industry'])

le_skills = LabelEncoder()
skills_df['Key Skills Required'] = le_skills.fit_transform(skills_df['Key Skills Required'])


X = skills_df[['Industry', 'Key Skills Required']]
y = skills_df['Skill Weight']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

def create_heatmap(data, geojson, value_column, map_title):
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

  
    folium.Choropleth(
        geo_data=geojson,
        name='choropleth',
        data=data,
        columns=['State/UT', value_column],
        key_on='feature.properties.NAME_1',  
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f'{map_title} (%)'
    ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


if 'view' not in st.session_state:
    st.session_state.view = 'Employment Chances Predictor'


st.sidebar.title('Dashboard Navigation')
if st.sidebar.button('Employment Chances Predictor'):
    st.session_state.view = 'Employment Chances Predictor'
if st.sidebar.button('View HeatMap'):
    st.session_state.view = 'View HeatMap'

if st.session_state.view == 'Employment Chances Predictor':
    st.header('Employment Chances Predictor')

    industry_input = st.selectbox('Select your industry of interest:', le_industry.classes_)
    selected_industry = le_industry.transform([industry_input])[0]

    industry_skills_df = skills_df[skills_df['Industry'] == selected_industry]
    industry_skills = le_skills.inverse_transform(industry_skills_df['Key Skills Required'])

    selected_skills = st.multiselect('Select your skills:', industry_skills)

 
    if selected_skills:
        selected_skills_encoded = le_skills.transform(selected_skills)
        skill_weights = []

        for skill in selected_skills_encoded:
            prediction = model.predict([[selected_industry, skill]])
            skill_weights.append(prediction[0])
        

        highest_weight = max(skill_weights)
        total_weight = sum(skill_weights)
        
        base_chance = 50  
        weight_factor = (highest_weight / 10) * 25 
        total_factor = (total_weight / (len(selected_skills) * 10)) * 25  
        
        employment_chance = base_chance + weight_factor + total_factor
        employment_chance = min(employment_chance, 100) 

        st.subheader(f'Employment Chances in {industry_input}:')

      
        skill_chances = [base_chance + (weight / 10) * 25 for weight in skill_weights]
        skill_chances = [min(chance, 100) for chance in skill_chances]  

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(selected_skills, skill_chances, marker='o', linestyle='-', color='skyblue')
        ax.set_xlabel('Skills')
        ax.set_ylabel('Employment Chances (%)')
        ax.set_title(f'Employment Chances in {industry_input}')
        ax.tick_params(axis='x', rotation=90)
        st.pyplot(fig)
       
        st.subheader('Employment Chances (Text):')
        for skill, chance in zip(selected_skills, skill_chances):
            st.write(f"{skill}: {chance:.2f}%")
    else:
        st.write("Please select an industry and skills.")

   
    st.sidebar.title('Employment Chances Scale')
    st.sidebar.markdown("""
    - Employment chances are predicted based on employment rate, the selected industry and weight of each skill.
    """)

elif st.session_state.view == 'View HeatMap':
    st.header('India Employment and Unemployment Rates Heatmap')

    
    rate_type = st.selectbox('Select Rate Type:', ['Employment Rate (%)', 'Unemployment Rate (%)'], key='heatmap_rate_type')


    if rate_type:
        value_column = rate_type
        map_title = rate_type.replace(' (%)', '')
        heatmap = create_heatmap(employment_df, geojson_file, value_column, map_title)
        folium_static(heatmap)


