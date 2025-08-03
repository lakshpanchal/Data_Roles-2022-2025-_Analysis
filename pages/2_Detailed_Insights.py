from ast import Str
from numpy.matlib import str_
from pandas.core.strings.accessor import str_extractall
import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
import numpy as np

engine  = create_engine("mysql+mysqlconnector://root:Laksh2005@127.0.0.1/data_roles")
def load_data():
    query = "Select * FROM exploded_final_data"
    return pd.read_sql(query,engine)
df = load_data()
filtered_df = df.copy()

def load_month_data():
    query = "Select * FROM data_roles_analysis"
    return pd.read_sql(query,engine)
data_df = load_month_data()


filtered_df = pd.merge(
    filtered_df,
    data_df[['title', 'company_name', 'Year', 'date_time']],
    on=['title', 'company_name', 'Year'],
    how='left'  # Use 'left' if you want all filtered_df rows retained
)

dedup_cols = ['title', 'company_name', 'Year', 'description_tokens', 'industry', 'location']
dedup_df = df[dedup_cols].drop_duplicates().reset_index(drop=True)
dedup_df['job_id'] = dedup_df.index
filtered_df = filtered_df.merge(dedup_df,on=dedup_cols,how='left')

st.title("In Depth Analysis")

# Filters
#Year
year_options = filtered_df['Year'].unique()
selected_year = st.sidebar.selectbox("",["Year"] + sorted(year_options.tolist()))
if selected_year != "Year":
    filtered_df = filtered_df[filtered_df['Year']==selected_year]

#Role
role_options = filtered_df['role_bin'].unique()
selected_role = st.sidebar.selectbox("",['Role']+ sorted(role_options.tolist()))
if selected_role != 'Role':
    filtered_df = filtered_df[filtered_df['role_bin'] == selected_role]

#Industry
industry_options = filtered_df['industry'].unique()
selected_industry = st.sidebar.selectbox(" ",["Industry"] + sorted(industry_options.tolist()))
if selected_industry != "Industry":
    filtered_df = filtered_df[filtered_df['industry']==selected_industry]

#Company
company_options = filtered_df.groupby('company_name')['job_id'].nunique().sort_values(ascending = False).index.tolist()
selected_company = st.sidebar.selectbox("", ['Company']+ company_options)
if selected_company != "Company":
    filtered_df = filtered_df[filtered_df['company_name'] == selected_company]

#Technical Tool
tool_options = filtered_df['technical tool'].unique()
selected_tool = st.sidebar.selectbox(" ", ["Technical Tool"] + sorted(tool_options.tolist()))
if selected_tool != "Technical Tool":
    filtered_df = filtered_df[filtered_df['technical tool']==selected_tool]


#KPIS

#Kpi 1
#Average number of tools per job role
tool_counts = filtered_df.groupby('job_id')['technical tool'].nunique().reset_index()
tool_counts.columns = ['job_id', 'num_tools']
avg_tools = tool_counts['num_tools'].mean()


#Kpi 2
#company with most jobs postings
top_company = filtered_df.groupby('company_name')['job_id'].nunique().sort_values(ascending = False).idxmax()

#Kpi3
#% of jobs requiring a certain tool
contains_tools = filtered_df.groupby('technical tool')['job_id'].nunique().reset_index()
contains_tools.columns = ['technical_tool', 'job_count']
total_count = filtered_df['job_id'].nunique()
contains_tools['percent_jobs'] = round((contains_tools['job_count']/total_count)*100, 2)
top_tool_row = contains_tools.sort_values('percent_jobs', ascending = False).iloc[0]

#Kpi 4
# Technical Stacks with highest Count
contains_stack = filtered_df.groupby('description_tokens')['job_id'].nunique().sort_values(ascending = False).idxmax()

#Kpi 5
contains_seniority = filtered_df.groupby('seniority_bin')['job_id'].nunique().sort_values(ascending=False).idxmax()




st.metric("Top Technical Stack", contains_stack)
col1,col2,col3,col4  = st.columns([1,1,1,1])
with col1:
    st.metric("Avg. Tools Per Job", round(avg_tools,2))
with col2:
    st.metric("Top Hiring Company", top_company)
with col3:
    st.metric(f"% of jobs using {top_tool_row['technical_tool']}", f"{top_tool_row['percent_jobs']}%")
with col4:
    st.metric("Average Seniority", contains_seniority)

#Visualizations
#Tree map for tool usage per industry
def treemap():
    tool_industry_df = filtered_df.groupby(['industry', 'technical tool'])['job_id'].nunique().reset_index()
    tool_industry_df.columns = ['Industry', 'Technical Tool', 'Count']
    fig = px.treemap(tool_industry_df, path = ['Industry','Technical Tool'],values = "Count", color = 'Industry')
    st.plotly_chart(fig, key = "Treemap_tools")

#Heatmap - tool vs role bin
def heat_map():
    top_tools = filtered_df.groupby('technical tool')['job_id'].nunique().sort_values(ascending=False).head(10).index.tolist()
    filtered_top_tools_df = filtered_df[filtered_df['technical tool'].isin(top_tools)]
    heatmap_df = filtered_top_tools_df.groupby(['role_bin','technical tool'])['job_id'].nunique().reset_index()
    heatmap_df.columns = ['Role', 'Tool', 'Count']
    heatmap_df = heatmap_df.pivot(index = 'Role', columns = 'Tool', values = 'Count').fillna(0)
    fig = px.imshow(heatmap_df, labels=dict(x = 'Tool', y = 'Role', color = 'Count') ,color_continuous_scale = 'Reds', aspect='auto')
    st.plotly_chart(fig, key = "Heatmap")



def linear_regression1():
    if selected_tool == "Technical Tool":
        st.warning("Please select a technical tool to visualize its regression.")
        return

    # Copy filtered_df to avoid overwriting global
    temp_df = filtered_df.copy()

    # Step 1: Extract Month and Year
    temp_df['Month'] = pd.to_datetime(temp_df['date_time']).dt.month
    temp_df['Year'] = pd.to_datetime(temp_df['date_time']).dt.year

    # Step 2: Count how many times the tool appears (not unique jobs — just mentions)
    tool_df = temp_df[temp_df['technical tool'] == selected_tool]
    tool_monthly = (
        tool_df.groupby(['Month', 'Year'])
        .size()
        .reset_index(name='Tool_Count')
    )

    # Step 3: Get total number of unique job postings per month-year
    monthly_total = (
        temp_df.groupby(['Month', 'Year'])['job_id']
        .nunique()
        .reset_index(name='Total_Jobs')
    )

    # Step 4: Merge both and normalize
    agg_df = pd.merge(tool_monthly, monthly_total, on=['Month', 'Year'])
    agg_df['normalized_data'] = agg_df['Tool_Count'] / agg_df['Total_Jobs']

    # Step 5: Create proper datetime from Year and Month
    agg_df['Date'] = pd.to_datetime(agg_df['Year'].astype(str) + '-' + agg_df['Month'].astype(str))

    # Step 6: Regression modeling
    X = agg_df['Date'].astype(np.int64).values.reshape(-1, 1)
    y = agg_df['normalized_data'].values

    model = LinearRegression()
    model.fit(X, y)
    agg_df['Predicted_Y'] = model.predict(X)

    # Step 7: Plot
    fig = px.scatter(agg_df, x='Date', y='normalized_data', title=f"Trend of {selected_tool} Over Time")
    fig.add_scatter(x=agg_df['Date'], y=agg_df['Predicted_Y'], mode='lines', name='Regression Line')
    st.plotly_chart(fig, key='Linear Regression')



def linear_regression():


    temp_df = filtered_df.copy()

    if selected_tool == "Technical Tool":
        st.warning("Please select a technical tool to visualize its regression.")
        return

    temp_df['Month'] = pd.to_datetime(filtered_df['date_time']).dt.month

    tool_monthly = filtered_df.groupby(['Month','technical tool','Year'])['job_id'].nunique().reset_index()
    tool_monthly.columns = ['Date','Tool','Year','Count']

    monthly_total = filtered_df.groupby(['Month','Year'])['job_id'].nunique().reset_index()
    monthly_total.columns = ['Date','Year', 'Total Jobs']

    agg_df = pd.merge(tool_monthly,monthly_total, on = ['Date','Year'] )
    agg_df['normalized_data']  = agg_df['Count']/agg_df['Total Jobs']


    agg_df['Date'] = pd.to_datetime(agg_df['Date'].astype(str)+ '-' + agg_df['Year'].astype(str))
    X = agg_df[['Date']].astype('int64')
    y = agg_df[['normalized_data']]
    model = LinearRegression()
    model.fit(X,y)
    agg_df['Predicted_Y'] = model.predict(X)
    fig = px.scatter(agg_df,x = 'Date', y = 'normalized_data')
    fig.add_scatter(x = agg_df['Date'], y = agg_df['Predicted_Y'],mode = 'lines', name = 'Regression Line' )
    st.plotly_chart(fig,key = 'Linear Regression')


treemap()
col1,col2 = st.columns(2)
with col1:
    heat_map()
with col2:
    linear_regression1()

st.dataframe(filtered_df.head(500))


