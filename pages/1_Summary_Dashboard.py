import streamlit as st
import pandas as pd
import plotly.express as px

# configure setup
st.set_page_config(page_title = 'Job Roles Dashboard',layout = 'wide')



# connect to mysql database
def load_data():
    return pd.read_csv("Final_data.csv")
df = load_data()
filtered_df = df.copy()

#dedupe the datatset to get rid of duplicates
dedup_cols = ['title', 'company_name', 'Year', 'description_tokens', 'industry', 'location']
dedup_df = df[dedup_cols].drop_duplicates().reset_index(drop=True)
dedup_df['job_id'] = dedup_df.index
filtered_df = filtered_df.merge(dedup_df,on=dedup_cols,how='left')

st.title("Job Roles High Level Overview")

st.sidebar.subheader("Filter Job Roles")

# FILTERS
# year filter
year_options = filtered_df['Year'].unique()
selected_year = st.sidebar.selectbox("",["Year"] + sorted(year_options.tolist()))
if selected_year != "Year":
    filtered_df = filtered_df[filtered_df['Year']==selected_year]

# Industry filter
industry_options = filtered_df['industry'].unique()
selected_industry = st.sidebar.selectbox(" ",["Industry"] + sorted(industry_options.tolist()))
if selected_industry != "Industry":
    filtered_df = filtered_df[filtered_df['industry']==selected_industry]


#Role filter
role_options = filtered_df['role_bin'].unique()
selected_role = st.sidebar.selectbox(" ", ["Role"] + sorted(role_options.tolist()))
if selected_role != "Role":
    filtered_df = filtered_df[filtered_df['role_bin']==selected_role]

#seniority filter
seniority_options = filtered_df['seniority_bin'].unique()
selected_seniority = st.sidebar.selectbox(" ", ["Seniority Level"]+sorted(seniority_options.tolist()))
if selected_seniority != "Seniority Level":
    filtered_df = filtered_df[filtered_df['seniority_bin'] == selected_seniority]

#technical tool filter
tool_options = filtered_df['technical tool'].unique()
selected_tool = st.sidebar.selectbox(" ", ["Technical Tool"] + sorted(tool_options.tolist()))
if selected_tool != "Technical Tool":
    filtered_df = filtered_df[filtered_df['technical tool']==selected_tool]


#KPIS
#Kpi1 - Average Tools
job_count = filtered_df['job_id'].drop_duplicates().shape[0]

#Kpi2 - most common tool
tool_counts = filtered_df['technical tool'].value_counts().idxmax()

#Kpi3 - company count
company_count = filtered_df['company_name'].nunique()

#kpi4 - total industries
industry_count = filtered_df['industry'].nunique()


col1,col2,col3,col4 = st.columns(4)

with col1:
    st.metric("Total Jobs", job_count)
with col2:
    st.metric('Most Common Tool', tool_counts)
with col3:
    st.metric("Total Companies", company_count)
with col4:
    st.metric("Total Industries", industry_count)

# VISUALIZATIONS
#Viz 1 - Bar Chart top ten in demand tools
def bar_tool():
    total_tool_count = filtered_df.groupby('technical tool')['job_id'].nunique()
    top_5_rows = total_tool_count.sort_values(ascending = False).head(10).to_frame().reset_index()
    top_5_rows.columns = ['Technical Tool', 'Count']
    top_5_rows = top_5_rows.sort_values(by = 'Count', ascending = True )
    fig = px.bar(top_5_rows, x = "Count",  y = "Technical Tool")
    fig.update_layout(yaxis_title = None)
    st.plotly_chart(fig, key = "tool_distribution")

#Viz 2 - Bar Chart Top ten companies
def bar_company():
    total_company_count = filtered_df.groupby('company_name')['job_id'].nunique()
    top_5_comp = total_company_count.sort_values(ascending = False).head(10).to_frame().reset_index()
    top_5_comp.columns = ["Company", "Count"]
    top_5_comp = top_5_comp.sort_values(by = "Count", ascending = True)
    fig = px.bar(top_5_comp, x = "Count", y = "Company")
    fig.update_layout(yaxis_title = None)
    st.plotly_chart(fig, key = "bar_company")

def Pie_industry():
    total_industry = filtered_df.groupby('industry')['job_id'].nunique().reset_index()
    total_industry.columns = ['Industry','Count']
    fig = px.pie(total_industry, names = "Industry", values = "Count", hole = 0.4)
    st.plotly_chart(fig, key = "Industry_distribution")

def location_type(filtered_df):
    filtered_df[['state','country']] = filtered_df['location'].str.split(',',expand = True)
    filtered_df['country'] = filtered_df['country'].str.strip().str.title()
    location_counts = filtered_df.groupby('country')['job_id'].nunique().reset_index()
    location_counts.columns = ['country','count']

    fig = px.choropleth(
        location_counts, locations = 'country', locationmode = 'country names',color = 'count',color_continuous_scale = 'Blues'
        )
    st.plotly_chart(fig, key = "us_state_map")

def stacked_year_trend():
    top_tools = (
        filtered_df.groupby('technical tool')['job_id'].nunique().sort_values(ascending=False).head(10).index
    )
    filtered_top_df = filtered_df[filtered_df['technical tool'].isin(top_tools)]
    stacked_chart = filtered_top_df.groupby(['Year','technical tool'])['job_id'].nunique().reset_index()
    stacked_chart.columns = ['Year','technical_tool','Count']
    fig = px.bar(stacked_chart, x = 'Year', y = "Count", color = "technical_tool")
    fig.update_layout(barmode = 'stack', yaxis_title = None)
    st.plotly_chart(fig, key = "stacked_year_trend")


st.subheader("Year on Year Technical Tool Demand")
stacked_year_trend()

st.subheader("Distribution by Technical Tool and Company")
col1, col2 = st.columns(2)
with col1:
    bar_tool()
with col2:
    bar_company()

st.subheader("Distribution by Industry and Locality")
col1, col2, = st.columns(2)
with col1:
    Pie_industry()
with col2:
    location_type(filtered_df)


st.dataframe(filtered_df)
