import os
import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine

st.title("Summary")
st.write("""
This project analyzes trends in technical tool usage across job postings from 2022 to 2025, using a cleaned and enriched dataset spanning thousands of listings across industries, roles, and companies.

It focuses on identifying how specific technologies (e.g., Python, Airflow, SQL) evolve in popularity over time—not just in raw counts, but through normalized trends that adjust for varying monthly job volumes. This ensures that insights reflect true demand signals rather than skewed totals from high-volume years.

Interactive filters, KPIs, and regression-based trendlines help uncover how different tools rise or fall in usage across time, roles, and industries.
""")
