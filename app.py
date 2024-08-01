import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import base64  # Import base64 for image encoding

# Load the data
data = pd.read_csv('Employment.csv')

# Function to convert an image to base64
def load_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Load background image
bg_image = load_image("bgi3.jpg")  # Ensure this path is correct

# Set page configuration
st.set_page_config(page_title="Employment Data Visualization", layout="wide")

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

# Streamlit app
st.markdown('<p class="title">Employment Data Visualization</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Filter Options")
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

# Prepare data for 3D bar chart
sector_data = data.groupby('State/UT').agg({f'{sector} (Male)': 'mean', f'{sector} (Female)': 'mean'}).reset_index()
states = sector_data['State/UT']
male_values = sector_data[f'{sector} (Male)']
female_values = sector_data[f'{sector} (Female)']

# Create 3D bar chart
fig = go.Figure()

# Add bars for male
fig.add_trace(go.Bar(
    x=states,
    y=male_values,
    name='Male',
    marker_color='blue',
    opacity=0.7
))

# Add bars for female
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
    paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
    plot_bgcolor='rgba(0,0,0,0)'   # Transparent plot area
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
    hole=0.3,  # Makes the pie chart look more like a donut for added style
    labels={'values': 'Percentage', 'names': 'Gender'}
)
fig.update_traces(textinfo='percent+label', pull=[0.1, 0.1])
fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))  # Remove margins for a cleaner look
st.plotly_chart(fig, use_container_width=True)

# 3D-like Heatmap
if show_heatmap:
    st.subheader("Heatmap of Employment Rates")

    # Prepare data for heatmap
    sectors = ["Technology", "Manufacturing", "Retail and Wholesale", "Healthcare", 
               "Education", "Finance and Accounting", "Construction and Real Estate", 
               "Agriculture", "Media and Entertainment", "Supply Chain and Transportation"]
    heatmap_data = data.set_index('State/UT')[[f'{sector} (Male)' for sector in sectors] + 
                                               [f'{sector} (Female)' for sector in sectors]]

    # Create 3D-like heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='YlGnBu',
        colorbar=dict(title='Employment Rate'),
        zmax=100,  # Adjust this based on your data range
        zmin=0,
        opacity=0.8  # Add transparency
    ))

    fig.update_layout(
        title='Heatmap of Employment Rates by Sector and Gender',
        xaxis_title='Sectors',
        yaxis_title='States/UTs',
        xaxis=dict(
            tickvals=list(range(len(heatmap_data.columns))),
            ticktext=heatmap_data.columns,
            tickangle=-45  # Rotate x-axis labels for better readability
        ),
        yaxis=dict(
            tickvals=list(range(len(heatmap_data.index))),
            ticktext=heatmap_data.index,
            tickangle=0,  # Ensure labels are not rotated
            tickfont=dict(size=10)  # Adjust font size for clarity
        ),
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        plot_bgcolor='rgba(0,0,0,0)',   # Transparent plot area
        autosize=True,  # Ensure the figure adjusts to container width
        height=800  # Increase height for better visibility
    )

    st.plotly_chart(fig, use_container_width=True)

