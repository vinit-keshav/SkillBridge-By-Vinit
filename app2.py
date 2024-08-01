import streamlit as st
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

st.write(f'### Actual vs. Predicted Employment Rate ({best_model_name})')
fig, ax = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred_best, ax=ax)
ax.set_xlabel('Actual Employment Rate')
ax.set_ylabel('Predicted Employment Rate')
ax.set_title(f'Actual vs. Predicted Employment Rate ({best_model_name})')
st.pyplot(fig)

# Improved heatmap of the correlation matrix
st.write('### Correlation Heatmap of Features')
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = X.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, linewidths=0.5)
ax.set_title('Correlation Heatmap of Features')
st.pyplot(fig)

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

# Button for prediction
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