#!/usr/bin/env python
# coding: utf-8

# In[6]:


# import libraries
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from jupyter_dash import JupyterDash
from dash import Input, Output, State, dcc, html, Dash


# In[7]:


# create a wrangle function to read excel sheet named 'CONSUMPTION'
def wrangle1(filepath):
    df = pd.read_excel(filepath, sheet_name = "CONSUMPTION")

    df.at[0, "Unnamed: 0"] = "Month" 
    
    # set first row as the new header
    new_header = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns = new_header)
    
    # create new column with date format "mm-yy" and convert to string
    df["Month"] = df["Month"].dt.strftime("%b-%y")
   
    # set "Month" column as index
    df = df.set_index("Month")
    
    # set values to floating and round to one decimal place
    df = df.astype(float).round(1)
    
    df["Total Consumption"] = df["WATER CONSUMPTION (m3)"]
    df = df.drop(columns = "WATER CONSUMPTION (m3)")
    
    df["consumption_per_mwh"] = df["Total Consumption"]/df["Generation (MWH)"]
    return df

df = wrangle1(r"C:\Users\HP\Desktop\Water_consump_2023.xlsx")
df.head()


# In[8]:


consump_type = df.columns.tolist()
consump_graph_list = [consump_type[4], consump_type[6], consump_type[7], consump_type[10]]
print(consump_type)


# In[9]:


# create a wrangle function to read excel sheet named 'Other Locations'
def wrangle2(filepath):
    df1 = pd.read_excel(filepath, sheet_name = "Other Locations")
   
    # set first row as the new header
    new_header = df1.iloc[0]
    df1 = pd.DataFrame(df1.values[1:], columns = new_header)
    
    # create new column with date format "mm-yy" and convert to string
    df1["Month"] = df1["Month"].dt.strftime("%b-%y")
    
    # drop 'power house main line', 'noon-operation line', 'treated water' and 'raw water' values
    df1 = df1.drop(
        columns = [
            "P.H Main Line",
            "Non-Operation",
            "Raw water tank consumption",
            "Treated Water"
        ]
    )
    
    # set "Month" column as index
    df1 = df1.set_index("Month")

    # set values to floating with one decimal place
    df1 = df1.astype(float).round(1)
   
    # transpoze entry for easy usage
    df1 = df1.T
    
    return df1

# read excel sheet 
df1 = wrangle2(r"C:\Users\HP\Desktop\Water_consump_2023.xlsx")


# In[10]:


df1.head()


# In[11]:


dates_of_tree_maps = df1.columns
locations = df1.index.tolist()


# In[13]:


app = JupyterDash(__name__)


# In[14]:


app.layout = html.Div([
    html.H1("AKSA ENERGY (GHANA) WATER CONSUMPTION ANALYSIS"),
    html.H2("Select Location to View Water Consumption Trend"),
    dcc.RadioItems(options = consump_graph_list, value = "Boiler", id = "trend-graph-select"),
    dcc.Graph(id = "trend-graph"),
    
    html.H2("Select Date to View Water Consumption by Location"),
    dcc.RadioItems(options = dates_of_tree_maps, value = dates_of_tree_maps[-1], id = "tree-map-display-select", inline = True),
    dcc.Graph(id = "tree-map-graph"),
    
    html.H2("Select Water Consumption and Power Generated Graph of Interest"), 
    dcc.Dropdown(options = ["Water Consumption vrs MWH Generation", "Water Consumed Per MWH Generation"],
                   value = "Water Consumption vrs MWH Generation", id = "MWH-graph-select"),
#     dcc.Graph(id = "MWH-graph"),
#     html.Div("This is a div element.")
])


# In[15]:


def cal_for_tree_map(df1, df, interested_month):
    
    """
    Make calculation for TreeMap Plot.

    Parameters
    ----------
    df1 : pandas.DataFrame
        Input data to be processed.

    df : pandas.DataFrame
        Input data to be processed.

    interested_month: str, optional
        The selected data type among months whose treemap visualization is of interest.
        Default is the most latest month.

    Returns
    -------
    dataframe : pandas.DataFrame
        DataFrame corresponding to the selected data type.
    """
    # initiate an if statement to to select month of interest    
    if interested_month in df.index:
        
        # identify total consumption value to be used in percentage calculation
        total_consumption_of_selected_month = df.loc[interested_month, "Total Consumption"]
        
        # calc. percentage consumption of all locations and name it 'pct_concump'
        df1["pct_consump"] = ((df1[interested_month] / total_consumption_of_selected_month) * 100).round(1)
        
        # add the '% of Total Consumption' statement to calculated values and name it 'percentage (%)'
        df1["percentage (%)"] = df1["pct_consump"].apply(lambda x: str(x) + "% of Total Consumption")
        
        #select the consumption column to which '[M3]' symbol is to be added
        consump_column = df1.columns[-3]
        
        # add the '[M3]' to water consumption values and name it 'amt_consumed_m3'
        df1["amt_consumed_m3"] = df1[f"{consump_column}"].apply(lambda x: str(x) + " [M3]")
    return df1


# In[16]:


@app.callback(Output("tree-map-graph", "figure"), Input("tree-map-display-select", "value"))
def treemap_graph(interested_month):
    if interested_month in df.index:
        df1_copy = df1.copy() # Make a copy of df1 to avoid modifying the original DataFrame

        df1_modified = cal_for_tree_map(df1_copy, df, interested_month).sort_index()  # Apply modifications to df1

        fig = px.treemap(
            df1_copy, path=[locations],
            values=interested_month,
            color=interested_month,
            color_continuous_scale="viridis", hover_data=["amt_consumed_m3"])

        fig.update_layout(
            height=520, width=1085, title=f"Water Consumption by Location for {interested_month}",
            margin=dict(t=40, l=40, r=40, b=40)
        )

        sorted_text_values = df1_modified["percentage (%)"]

        fig.update_traces(
            text=sorted_text_values,  # Use sorted text values for annotations
            textinfo="label+text+value"
        )

    return fig


# In[17]:


def calc_for_trend_graph(selected_data):
    """
    Make calculation for Trend Graph.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data to be processed.
        
    selected_data : str, optional
        The selected data type among 'Boiler', 'Treated Water', or 'Total Consumption'.
        Default is 'Boiler'.

    Returns
    -------
    selected_dataframe : pandas.DataFrame
        DataFrame corresponding to the selected data type.
    """
    if selected_data == "Treated Water":
        return df["Treated Water"]
        
    elif selected_data == "Boiler":
        return df["Boiler"]

    elif selected_data == "Total Consumption":    
        return df["Total Consumption"]


# In[18]:


@app.callback(Output("trend-graph", "figure"), Input("trend-graph-select", "value"))
def server_trend_graph(selected_data = "Boiler"):
    
    # set graph size
    fig = px.line(df, x = df.index, y = df[selected_data], title = "Time Series Water Consumption")
    
    annotations = []
    for i, y in enumerate(df[selected_data]):
        annotations.append(
            dict(x=df.index[i], y=y, text=str(y),
                 showarrow=False,  font=dict(color='black', size=10), xshift=5, yshift=12))
     
    x = np.arange(len(df.index))
    x = sm.add_constant(x)
    y = df[selected_data]
    model = sm.OLS(y, x).fit()
    trendline = model.predict(x)
    
    fig.add_scatter(
        x=df.index,
        y=trendline,
        mode='lines+markers',
        name='Trendline',
        hoverinfo = "skip",
        line = dict(dash = "dash", width = 1.5), marker = dict(size = 0.1))
    
    fig.update_layout(annotations=annotations, plot_bgcolor="bisque", height=550, width=1100 )
    
    fig.update_traces(mode="lines+markers")
    
    fig.update_xaxes(tickangle = -45)
    fig.show()
    return fig


# In[19]:


if __name__ == "__main__":
    app.run_server(debug = False)


# In[ ]:





# In[ ]:




