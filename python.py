
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px
import base64

# Load data
skills_df = pd.read_csv('Skills.csv')
employment_df = pd.read_csv('UAER.csv')
geojson_file = 'India_states.geojson'

# Load the Employment data for gender distribution
data = pd.read_csv('Employment.csv')

# Function to convert an image to base64
def load_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Load background image
bg_image = load_image("errr.avif")  # Ensure this path is correct

# Set page configuration
st.set_page_config(page_title="SkillBridge by Vinit", layout="wide")

# Custom CSS for background image and styling
st.markdown(
    f"""
    <style>
    .main {{
        background-image: url(data:image/jpeg;base64,{bg_image});
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center center;
        color: #FFFFFF;
    }}
    .sidebar .sidebar-content {{
        background: linear-gradient(145deg, #f0f0f0, #cacaca);
        border-radius: 15px;
        box-shadow: 6px 6px 12px #a6a6a6, -6px -6px 12px #ffffff;
    }}
    .title {{
        font-size: 2.5em;
        font-weight: bold;
        color: #FFFFFF;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }}
    .title-3d {{
        font-size: 2.5em;
        font-weight: bold;
        color: #FFFFFF;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7), -2px -2px 4px rgba(255, 255, 255, 0.7);
        animation: pulse 1s infinite;
    }}
    @keyframes pulse {{
        0% {{
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7), -2px -2px 4px rgba(255, 255, 255, 0.7);
        }}
        50% {{
            text-shadow: 4px 4px 8px rgba(0, 0, 0, 0.7), -4px -4px 8px rgba(255, 255, 255, 0.7);
        }}
        100% {{
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7), -2px -2px 4px rgba(255, 255, 255, 0.7);
        }}
    }}
    .stText {{
        color: #FFFFFF;
    }}
    .stPlot {{
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }}
    .stButton button {{
        background: linear-gradient(145deg, #ffffff, #e6e6e6);
        border: none;
        border-radius: 12px;
        box-shadow: 6px 6px 12px #a6a6a6, -6px -6px 12px #ffffff;
        color: #333333;
        font-size: 16px;
        font-weight: bold;
        padding: 10px 20px;
        margin: 10px 0;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
    }}
    .stButton button:hover {{
        box-shadow: 3px 3px 6px #a6a6a6, -3px -3px 6px #ffffff;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
# Sidebar navigation
st.sidebar.title('Dashboard')
if st.sidebar.button('Chances In Different Sectors'):
    st.session_state.view = 'Employment Chances In Different Sectors'
if st.sidebar.button('Employment Rate Predictor'):
    st.session_state.view = 'Employment Rate Predictor'
if st.sidebar.button('View HeatMap'):
    st.session_state.view = 'View HeatMap'
if st.sidebar.button('Gender Distribution'):
    st.session_state.view = 'Gender Distribution'
if 'view' not in st.session_state:
    st.session_state.view = None
# Display title with 3D effect if no option is selected
if st.session_state.view is None:
    st.markdown('<h1 class="title-3d">SkillBridge by Vinit. Select option from DashBoard</h1>', unsafe_allow_html=True)
else:
    st.markdown('<h1 class="title">SkillBridge by Vinit</h1>', unsafe_allow_html=True)
# # Initialize session state for view
# if 'view' not in st.session_state:
#     st.session_state.view = None

# Encode categorical variables
le_industry = LabelEncoder()
skills_df['Industry'] = le_industry.fit_transform(skills_df['Industry'])
le_skills = LabelEncoder()
skills_df['Key Skills Required'] = le_skills.fit_transform(skills_df['Key Skills Required'])

# Prepare data for model
X = skills_df[['Industry', 'Key Skills Required']]
y = skills_df['Skill Weight']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Function to create heatmap
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

# Sidebar navigation
if 'view' not in st.session_state:
    st.session_state.view = 'Employment Chances Predictor'


# Employment Chances Predictor
if st.session_state.view == 'Employment Chances In Different Sectors':
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
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=selected_skills,
            y=skill_chances,
            z=[1]*len(skill_chances),  # Adding a dummy Z axis for 3D effect
            mode='markers+lines',
            marker=dict(size=10, color=skill_chances, colorscale='Viridis', opacity=0.8),
            line=dict(color='skyblue', width=2),
            name='Employment Chances'
        ))

        fig.update_layout(
            scene=dict(
                xaxis_title='Skills',
                yaxis_title='Employment Chances (%)',
                zaxis_title='Dummy Axis'
            ),
            title=f'Employment Chances in {industry_input}',
            margin=dict(l=0, r=0, b=0, t=40),
            width=1200,  # Increase the width
            height=800   # Increase the height
        )

        st.plotly_chart(fig, use_container_width=True)
        st.subheader('Employment Chances (Text):')
        for skill, chance in zip(selected_skills, skill_chances):
            st.write(f"{skill}: {chance:.2f}%")
    else:
        st.write("Please select an industry and skills.")
# Rate Predictor
elif st.session_state.view == 'Employment Rate Predictor':
    # Your provided code
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.metrics import mean_absolute_error, r2_score

    # Load the data from the CSV file
    data = pd.read_csv('ERR.csv')

    # Prepare the features and target variable
    X = data[['GDP_growth_rate', 'inflation_rate', 'literacy_rate', 'population_growth_rate', 'industrial_production_index']]
    y = data['employment_rate']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # List of models to train
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Support Vector Machine": SVR()
    }

    # Dictionary to store the results
    results = {}

    # Train and evaluate each model
    for name, model in models.items():
        if name == "Support Vector Machine":
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {'MAE': mae, 'R2': r2}

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results).T
    results_df['Accuracy (%)'] = results_df['R2'] * 100

    # Streamlit app
    st.title('Employment Rate Prediction')
    st.write('This app predicts the employment rate using various machine learning models.')

    st.write('### Model Performance')
    st.write(results_df)

    # 3D bar chart for MAE, R2, and Accuracy
    st.write('### Model Performance Comparison')
    fig = go.Figure(data=[
        go.Bar(name='MAE', x=results_df.index, y=results_df['MAE']),
        go.Bar(name='R2', x=results_df.index, y=results_df['R2']),
        go.Bar(name='Accuracy (%)', x=results_df.index, y=results_df['Accuracy (%)'])
    ])
    fig.update_layout(barmode='group', title='Model Performance Comparison', yaxis_title='Score')
    st.plotly_chart(fig)

    # Scatter plot for the best-performing model (by R2 score)
    best_model_name = results_df['R2'].idxmax()
    best_model = models[best_model_name]

    if best_model_name == "Support Vector Machine":
        y_pred_best = best_model.predict(scaler.transform(X_test))
    else:
        y_pred_best = best_model.predict(X_test)

    # Input data for 2024
    st.write('### Input Data for 2024')
    gdp_growth_rate = st.number_input('GDP Growth Rate (%)', value=8.2, format="%.1f", disabled=True)
    inflation_rate = st.number_input('Inflation Rate (%)', value=4.5, format="%.1f", disabled=True)
    literacy_rate = st.number_input('Literacy Rate (%)', value=77.7, format="%.1f", disabled=True)
    population_growth_rate = st.number_input('Population Growth Rate (%)', value=0.8, format="%.1f", disabled=True)
    industrial_production_index = st.number_input('Industrial Production Index Growth Rate (%)', value=9.9, format="%.1f", disabled=True)
    # Prepare the new input data
    new_data = pd.DataFrame({
        'GDP_growth_rate': [gdp_growth_rate],
        'inflation_rate': [inflation_rate],
        'literacy_rate': [literacy_rate],
        'population_growth_rate': [population_growth_rate],
        'industrial_production_index': [industrial_production_index]
    })
    if st.button('Predict Employment Rate for 2024'):
        # Predict using the best model
        if best_model_name == "Support Vector Machine":
            new_data_scaled = scaler.transform(new_data)
            employment_rate_pred = best_model.predict(new_data_scaled)
        else:
            employment_rate_pred = best_model.predict(new_data)

        st.write(f"Predicted Employment Rate for 2024: {employment_rate_pred[0]:.2f}%")
        st.write(f"Model used: {best_model_name}")
        st.write(f"Model Accuracy: {results_df.loc[best_model_name, 'Accuracy (%)']:.2f}%")

    # Improved heatmap of the correlation matrix
    st.write('### Correlation Heatmap of Features')
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = X.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, linewidths=0.5)
    ax.set_title('Correlation Heatmap of Features')
    st.pyplot(fig)

# View HeatMap
elif st.session_state.view == 'View HeatMap':
    st.header('India Employment and Unemployment Rates Heatmap')
    rate_type = st.selectbox('Select Rate Type:', ['Employment Rate (%)', 'Unemployment Rate (%)'], key='heatmap_rate_type')
    if rate_type:
        value_column = rate_type
        map_title = rate_type.replace(' (%)', '')
        heatmap = create_heatmap(employment_df, geojson_file, value_column, map_title)
        folium_static(heatmap)

elif st.session_state.view == 'Gender Distribution':
    st.header("Gender Distribution Analysis")

    # Custom CSS for gender distribution page
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(180deg, black, grey);
        }}
        .sidebar .sidebar-content {{
            background-color: rgba(0, 0, 0, 0.5);
        }}
        .title {{
            font-size: 2.5em;
            font-weight: bold;
            color: #FFFFFF;
        }}
        .stText {{
            color: #FFFFFF;
        }}
        .stPlot {{
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Gender distribution
    st.sidebar.title("Gender Distribution Options")
    show_data = st.sidebar.checkbox('Show Data')
    sector = st.sidebar.selectbox(
        "Select Sector for Bar Chart", 
        ["Technology", "Manufacturing", "Retail and Wholesale", "Healthcare", "Education", 
         "Finance and Accounting", "Construction and Real Estate", "Agriculture", 
         "Media and Entertainment", "Supply Chain and Transportation"]
    )
    pie_sector = st.sidebar.selectbox(
        "Select Sector for Pie Chart", 
        ["Technology", "Manufacturing", "Retail and Wholesale", "Healthcare", "Education", 
         "Finance and Accounting", "Construction and Real Estate", "Agriculture", 
         "Media and Entertainment", "Supply Chain and Transportation"]
    )
    show_heatmap = st.sidebar.checkbox('Show Heatmap')

    # Show data
    if show_data:
        st.write(data)

    # 3D Bar Chart
    st.subheader("Employment by Sector")
    sector_data = data.groupby('State/UT').agg({f'{sector} (Male)': 'mean', f'{sector} (Female)': 'mean'}).reset_index()
    states = sector_data['State/UT']
    male_values = sector_data[f'{sector} (Male)']
    female_values = sector_data[f'{sector} (Female)']

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=states,
        y=male_values,
        name='Male',
        marker_color='blue',
        opacity=0.7
    ))
    fig.add_trace(go.Bar(
        x=states,
        y=female_values,
        name='Female',
        marker_color='pink',
        opacity=0.7
    ))

    fig.update_layout(
        title=f'{sector} Employment by State/UT',
        xaxis_title='State/UT',
        yaxis_title='Employment Percentage',
        barmode='group',
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Pie Chart
    st.subheader("Gender Distribution in Sector")
    gender_data = data[[f'{pie_sector} (Male)', f'{pie_sector} (Female)']].mean()
    fig = px.pie(
        names=['Male', 'Female'], 
        values=gender_data, 
        title=f'Gender Distribution in {pie_sector}', 
        color_discrete_sequence=['blue', 'pink'],
        hole=0.3,
        labels={'values': 'Percentage', 'names': 'Gender'}
    )
    fig.update_traces(textinfo='percent+label', pull=[0.1, 0.1])
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)

    # 3D-like Heatmap
    if show_heatmap:
        st.subheader("Heatmap of Employment Rates")
        sectors = ["Technology", "Manufacturing", "Retail and Wholesale", "Healthcare", 
                   "Education", "Finance and Accounting", "Construction and Real Estate", 
                   "Agriculture", "Media and Entertainment", "Supply Chain and Transportation"]
        heatmap_data = data.set_index('State/UT')[[f'{sector} (Male)' for sector in sectors] + 
                                                   [f'{sector} (Female)' for sector in sectors]]

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='YlGnBu',
            colorbar=dict(title='Employment Rate'),
            zmax=100,
            zmin=0,
            opacity=0.8
        ))

        fig.update_layout(
            title='Heatmap of Employment Rates by Sector and Gender',
            xaxis_title='Sectors',
            yaxis_title='States/UTs',
            xaxis=dict(
                tickvals=list(range(len(heatmap_data.columns))),
                ticktext=heatmap_data.columns,
                tickangle=-45
            ),
            yaxis=dict(
                tickvals=list(range(len(heatmap_data.index))),
                ticktext=heatmap_data.index,
                tickangle=0,
                tickfont=dict(size=10)
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            autosize=True,
            height=800
        )

        st.plotly_chart(fig, use_container_width=True)