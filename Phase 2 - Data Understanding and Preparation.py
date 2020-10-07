#!/usr/bin/env python
# coding: utf-8

# # <h1><center> COMMUNICATION THROUGH DATA and STORY THROUGH VISUALIZATION </center></h1>
# 
# ## Data Ingestion & Wrangling

# ### Table of Contents
# * <b>[Data Understanding and Preparation](#sec1)</b>
#     * [The Datasets](#subsec_1_1)
#     * [Questions to Research ](#subsec_1_2)
#         * [Stage I](#subsec_1_2_1)
#         * [Stage II](#subsec_1_2_2)
#     * [Packages Used](#subsec_1_3)
# * <b>[Covid-19 data](#sec2)</b>
#     * [Covid-19 pandemic for the US](#subsec_2_1)
#     * [Reading the data](#subec_2_2)
#     * [Creating Covid-19 data for Maryland State](#subec_2_3) 
# * <b>[Traffic Violation](#sec3)</b>
#     * [Reading "Traffic_Violations" data](#subsec_3_1)  
#          * [Cleaning Data](#subsec_3_1_1)   
# * <b>[Crash Reporting Incidents Data](#sec4)</b>
#     * [Reading data](#subsec_4_1)
#         * [Cleaning Data](#subsec_4_1_1) 
# * <b>[Crash Reporting – Non-Motorists Data](#sec5)</b>
#     * [Reading data](#subsec_5_1)
#         * [Cleaning Data](#subsec_5_1_1) 
#             

# # Data Understanding and Preparation <a class="anchor" id="sec1"></a>
# 
# The objective of the project is to examine five datasets from [dataMontgomery](https://data.montgomerycountymd.gov/). Two of the data sets is related to Covid-19 and other two is related to crash reporting data. Also, Traffic Violation is considered in the project.

# ## Data Sets <a class="anchor" id="subsec_1_1"></a>
# 
# <b> Covid-19 data  :</b>
#     
# 1. <a href = "https://covidtracking.com/data" URL> [Covid-19 pandemic for the US] <!-- link to site -->  
#     
# >1. This data is provided by *The Covid Tracking Project* which is a volunteer Organization launched from the *Atlantic*.  Although there are non-reported columns, I am intersted in knowing total number of confirmed cases and deaths in different States and see the trend in different months. 
# 
# <b> Trafic Violation  :  <a href = "https://data.montgomerycountymd.gov/Public-Safety/Traffic-Violations/4mse-ku6q" URL> [Trafic Violation data]<!-- link to site --> </b>
# 
# > This dataset is updated daily and provides traffic violation information from all electronic traffic violations issued in the County. I am specifically interested in creating mapping visualizations utilizing this dataset.
# 
# 
# <b> Crash Report :</b>
# 
# 1.  <a href = "https://data.montgomerycountymd.gov/Public-Safety/Crash-Reporting-Incidents-Data/bhju-22kf" URL> [Crash Reporting Incident data]<!-- link to site -->
#     
# 2. <a href = "https://data.montgomerycountymd.gov/Public-Safety/Crash-Reporting-Non-Motorists-Data/n7fk-dce5" URL> [Crash Reporting-Non-Motorists data]<!-- link to site -->

# ## Questions to Research <a class="anchor" id="subsec_1_2"></a>
# 
# ### Stage I <a class="anchor" id="subsec_1_2_1"></a>
# 
# 
# <b>Dataset: </b> Covid-19 data
# * Which state has the highest covid case and deaths?
# * How does covid confirmed case and deaths change with the season?
# * How many people hospitalized from March 2020 to Ovtober 2020?
# 
# <b>Dataset: </b> Traffic Violation
# * What type/model of car had highest citation?
# * What days of the week and month had the highest warning and citations?
# * Who (M/F) had highest citations and warnings?
# 
# <b>Dataset: </b> Crash Report
# * Comapre
# 
# ### Stage II <a class="anchor" id="subsec_1_2_1"></a>
# 
# <b>Dataset: </b> Crash Report Incident Data and Crash Report Non-Motorists Data
# * How can we compare Crash Report Incident Data and Crash Report Non-Motorists Data?
# 
# <b>Dataset: </b> Traffic Violation and Crash Report Data
# * How can we compare Crash Report Incident Data and Crash Report Non-Motorists Data?

# # Packages Used <a class="anchor" id="subsec_1_3"></a>

# In[1]:


## import packages:
import numpy as np # for vector operations
from numpy import percentile
import pandas as pd # for Data Frame methods
import matplotlib.pyplot as plt # Two dimensional plotting library
import matplotlib as mpl
import seaborn as sns #  library for making statistical graphics
from scipy import stats #  library of statistical functions


# # 1. Covid-19 data <a class="anchor" id="sec2"></a> 

# ### Reading the data : Covid-19 pandemic for the US <a class="anchor" id="subsec_2_1"></a> 

# In[2]:


covid = pd.read_csv("all-states-history.csv")
# Dimension of the dataframe :
np.shape(covid)


# In[3]:


# Pandas head() method is used to return top n (5 by default) rows of a data frame or series
covid.head()


# ## Cleaning the Data

# ### Confirm the date column is in datetime format
# > The orignal date format was "Y-m-d" and which is converted to "m/d/Y" format to match with the other datasets considered in the project.

# In[4]:


#rename the column
covid = covid.rename(columns={"date":"date_new"})

# Confirm the date column is in datetime format
covid['date_new'] = pd.to_datetime(covid['date_new'], format='%Y-%m-%d')

# observing the datatypes of the dataset
covid.dtypes


# ### Date format
# > The orignal date format was "Y-m-d" and which is converted to "m/d/Y" format to match with the other datasets considered in the project.
# Also, new columns : date in "m/d/Y" format and year is created.

# In[5]:


# creating a new columns, date, year and month in the data frame
covid['date'] = pd.to_datetime(covid['date_new'], format='%Y-%m-%d').dt.strftime('%m/%d/%Y')
covid['year'] = covid['date_new'].dt.year
covid['month'] = covid['date_new'].dt.month
covid.head()


# ## Deleting the Columns
# 
# 
# > Deleted the following columns which is not reported by the Covid Tracking Project and which is stored in new data frame, **covid_new**
# 

# In[6]:


# Dropping columns which are of not interest 
covid_new = covid.drop(columns=['dataQualityGrade',  'totalTestsAntibody', 'totalTestsPeopleAntibody','totalTestsPeopleAntigen','inIcuCumulative','totalTestEncountersViral','totalTestsAntigen','totalTestEncountersViralIncrease'])
covid_new.head()


# ## Creating Covid-19 data for Maryland State <a class="anchor" id="subsec_2_3"></a>

# In[7]:


covidMD = covid_new[covid_new['state']=='MD']


# In[8]:


# Dimension of the dataframe :
print(np.shape(covidMD))
covidMD.head()


# ### *reset_index()* 
# 
# > *reset_index()* is used to reset the row index to make the index start from 0. We can call reset_index() on the dataframe.

# In[9]:


covidMD_new = covidMD.reset_index(drop=True)
#With the argument, drop=True, pandas do not keep the original index.
covidMD_new.head()


# ## Creating the dataframe, *covid_info* for Maryland and for all States
# 
# The new data frames, ***covid_info_MD*** and ***covid_info_St*** are defined by filtering some of the columns and which are used in the later visualizations.

# In[10]:


# selecting the columns for all States
columns_St = covid_new[['state','totalTestResults', 'positive', 'negative', 'hospitalized', 'death', 'recovered', 'date', 'year', 'month']]


# In[11]:


#creating the new dataframe For All States :
covid_info_St = columns_St.copy()
covid_info_St.head()


# In[12]:


# selecting the columns for Maryland :
columns_MD = covid_new[['totalTestResults', 'positive', 'negative', 'hospitalized', 'death', 'recovered', 'date', 'year', 'month']]


# In[13]:


#creating the new dataframe For Maryland :
covid_info_MD = columns_MD.copy()
covid_info_MD.head()


# # 2.Traffic Violation <a class="anchor" id="sec3"></a> 

#  ## Reading "Traffic_Violations" data <a class="anchor" id="subsec_3_1"></a>  

# In[14]:


traffic = pd.read_csv("Traffic_Violations.csv",low_memory=False)
# Dimension of the dataframe :
np.shape(traffic)


# In[15]:


# Pandas head() method is used to return top n (5 by default) rows of a data frame or series
traffic.head()


#  ## Cleaning the Data <a class="anchor" id="subsec_3_1_1"></a>  

# ### Date format
# 
# >In the following code, first I have converted the columns, *Date Of Stop* and *Time Of Stop* to date dormat. Then new columns, date, year, time and month are created. I tried to follow this pattern to other data frames as well.
# 

# In[16]:


# Confirm the date column is in datetime format
traffic['Date Of Stop'] = pd.to_datetime(traffic['Date Of Stop'], format='%m/%d/%Y')

# Confirm the  time column is in datetime format
traffic['Time Of Stop'] = pd.to_datetime(traffic['Time Of Stop'], format='%H:%M:%S')

# creating a new columns, date, year, time in the data frame
traffic['date'] = pd.to_datetime(traffic['Date Of Stop'], format='%m/%d/%Y').dt.strftime('%m/%d/%Y')
traffic['year'] = traffic['Date Of Stop'].dt.year
traffic['month'] = traffic['Date Of Stop'].dt.month
traffic['time'] = traffic['Time Of Stop'].dt.time
traffic.head()


# ## Replace Yes or No by 0 and 1

# In[17]:


#defining new data frame
traffic_replace = traffic.replace( {'Yes' : 1, 'No' : 0})
traffic_replace.head()


# In[18]:


#rename the column
traffic_replace = traffic_replace.rename(columns={"Year":"Year_model"})
traffic_replace.head()


# ## Count Total NAN
#     * Count all NaN in a Data Frame (both columns and rows)
#     * Count total NaN at each column in a data Frame.

# In[19]:


#Dimension of the Dataframe
traffic_replace.shape


# In[20]:


#Total number of NaN
traffic_replace.isnull().sum().sum()


# In[21]:


#Total number NaN at each column
traffic_replace.isnull().sum()


# ## Deleting columns from *traffic_replace* data frame
# > In the following code, the columns with highest NaN's  and nonrelevant columns are deleted.

# In[22]:


delete_columns = traffic_replace[['SeqID', 'SubAgency', 'Search Conducted', 'Search Disposition', 'Search Outcome' , 'Search Reason', 'Search Reason For Stop', 'Search Type', 'Search Arrest Reason', 'Article', 'Driver City', 'Driver State', 'DL State', 'Arrest Type']]


# In[23]:


# Delete these columns from the data frame, traffic_new
traffic_new = traffic_replace.drop(delete_columns ,axis=1)
traffic_new.head()


# ## Exploratory Analysis

# In[24]:


column_violation = traffic_new[['Accident', 'Belts', 'Personal Injury', 'Property Damage', 'Fatal', 'Commercial License', 'HAZMAT', 'Commercial Vehicle', 'Alcohol', 'Work Zone', 'Violation Type']]


# In[25]:


plt.figure(figsize = (10,5))
sns.heatmap(column_violation.corr(), cmap='PuBuGn', annot=True,  linewidths=.1)


# In[26]:


column_violation.describe() 


# ### Correlate with Violation type

# In[27]:


table1 = pd.pivot_table(column_violation, values=['Accident', 'Belts', 'Personal Injury', 'Property Damage', 'Fatal', 'Commercial License', 'HAZMAT', 'Commercial Vehicle', 'Alcohol', 'Work Zone'], columns='Violation Type', aggfunc=np.mean)
table1


# From the above table, it is clear that Belts and Property damage are the highest violation

# In[28]:


plt.figure(figsize = (16,5))
sns.heatmap(traffic_new.corr(), cmap='PuBuGn', annot=True,  linewidths=.1)


# > Most of the correlations are weak. It's interesting how the highest correlation is between Accident and Contributed to Accident. 
# 
# > The next highest correlation is between,
#     * Personal Injury and Contributed Accident.
#     * Property Damage and Contributed Accident.

# # 3. Crash Reporting Incidents Data <a class="anchor" id="sec4"></a> 

#  ## Reading "Crash Reporting Incidents Data" <a class="anchor" id="subsec_4_1"></a>  

# In[29]:


CRI= pd.read_csv("Crash_Reporting_-_Incidents_Data.csv")
# Dimension of the dataframe :
np.shape(CRI)


# In[30]:


# Pandas head() method is used to return top n (5 by default) rows of a data frame or series
CRI.head()


#  ## Cleaning the Data <a class="anchor" id="subsec_4_1_1"></a>  

# In[31]:


#Dimension of the Dataframe
print("Dimension of the data set is :", CRI.shape)

#Total number of NaN
print("Total number of NaNs in the entire data frame is :",CRI.isnull().sum().sum())

#Total number NaN at each column
#print("Total number of NaNs in each column is:\n", CRI.isnull().sum())


# ### Renaming Column
# 
# > Here the column name, "Agency Name" is changed to "Agency" to have same name as in "Traffic Violation" data.

# In[32]:


#Column name 
CRI_rename = CRI.rename(columns={"Agency Name": "Agency"})
CRI_rename.head()


# ### Replace
# 
# > Also, the name "Montgomery County Police" is changed to "MCP" as in traffic violation data.

# In[33]:


CRI_rename= CRI_rename.replace({'Agency' : {'Montgomery County Police' : 'MCP', 'Rockville Police Departme' : 'RPD'}})
CRI_rename.head()


# In[34]:


# Confirm the date column is in datetime format
CRI_rename['Crash Date/Time'] = pd.to_datetime(CRI_rename['Crash Date/Time'], format='%m/%d/%Y %H:%M:%S %p')
# creating a new columns, date, year and time in the data frame
CRI_rename['date'] = pd.to_datetime(CRI_rename['Crash Date/Time'], format='%m/%d/%Y %H:%M:%S %p').dt.strftime('%m/%d/%Y')
CRI_rename['year'] = CRI_rename['Crash Date/Time'].dt.year
CRI_rename['month'] = CRI_rename['Crash Date/Time'].dt.month
CRI_rename['time'] = CRI_rename['Crash Date/Time'].dt.time
CRI_rename.head()


# # 4. Crash Reporting – Non-Motorists Data <a class="anchor" id="sec5"></a> 

#  ## Reading "Crash Reporting - Non-Motorists Data"  <a class="anchor" id="subsec_5_1"></a>  

# In[35]:


CRNM= pd.read_csv("Crash_Reporting_-_Non-Motorists_Data.csv")


# In[36]:


# Pandas head() method is used to return top n (5 by default) rows of a data frame or series
CRNM.head()


# ## Cleaning the Data <a class="anchor" id="subsec_5_1_1"></a>  

# In[37]:


#Dimension of the Dataframe
print("Dimension of the data set is :", CRNM.shape)

#Total number of NaN
print("Total number of NaNs in the entire data frame is :",CRNM.isnull().sum().sum())

#Total number NaN at each column
#print("Total number of NaNs in each column is:\n", CRNM.isnull().sum())


# ### Renaming Column
# 
# > Here the column name, "Agency Name" is changed to "Agency" to have same name as in "Traffic Violation" data.

# In[38]:


#Column name 
CRNM_rename = CRNM.rename(columns={"Agency Name": "Agency"})
CRNM_rename.head()


# ### Replace
# 
# > Also, the name "Montgomery County Police" is changed to "MCP" as in traffic violation data.

# In[39]:


CRNM_rename = CRNM_rename.replace({'Agency' : {'Montgomery County Police' : 'MCP', 'Rockville Police Departme' : 'RPD'}})
CRNM_rename.head()


# In[40]:


# Confirm the date column is in datetime format
CRNM_rename['Crash Date/Time'] = pd.to_datetime(CRNM_rename['Crash Date/Time'], format='%m/%d/%Y %H:%M:%S %p')
# creating a new columns, date, year and time in the data frame
CRNM_rename['date'] = pd.to_datetime(CRNM_rename['Crash Date/Time'], format='%m/%d/%Y %H:%M:%S %p').dt.strftime('%m/%d/%Y')
CRNM_rename['year'] = CRNM_rename['Crash Date/Time'].dt.year
CRNM_rename['month'] = CRNM_rename['Crash Date/Time'].dt.month
CRNM_rename['time'] = CRNM_rename['Crash Date/Time'].dt.time
CRNM_rename.head()


# # Merging Two Data Frames : Crash Incident Data and Crash Non-Motorists Data
# 
# 

# ## Creating new data frame for Crash Incidents Data 
# 
# > The following filiters are made to define new data frame
#     * Certain colomns are selected, viz., Agency, Hit/Run, At Fault, Weather etc.
#     * New column, dataframe is inserted to identify the data frame.
#     * Only Motgomery County Police is considered.

# In[41]:


##Selecting the columns
CRI_columns = CRI_rename[['Agency', 'Collision Type', 'Hit/Run' , 'At Fault' , 'Weather', 'Traffic Control', 'Driver Substance Abuse','year','month']]

## Creating new data frame for Crash Incidents Data 
CRI_new = CRI_columns.copy()

# inserting a new column to identify the data frame
CRI_new.insert(1, 'dataframe', 'Incident')

#
CRI_new = CRI_new[CRI_new['Agency']=='MCP']

CRI_new.head()


# ## Creating new data frame for Crash Non-Motorists Data 
# 
# > The following filiters are made to define new data frame
#     * Certain colomns are selected, viz., Agency, Hit/Run, At Fault, Weather etc.
#     * New column, dataframe is inserted to identify the data frame.
#     * Only Motgomery County Police is considered.

# In[42]:


##Selecting the columns
CRNM_columns = CRNM_rename[['Agency', 'Collision Type', 'At Fault' , 'Weather', 'Traffic Control', 'Pedestrian Movement','year','month']]

## Creating new data frame for Crash Incidents Data 
CRNM_new = CRNM_columns.copy()

# inserting a new column to identify the data frame
CRNM_new.insert(1, 'dataframe', 'Non-motorists')

#
CRNM_new = CRNM_new[CRNM_new['Agency']=='MCP']

CRNM_new.head()


# In[43]:


result = pd.concat([CRI_new, CRNM_new], join='outer', sort=False)
result


# ## The number of crashes involving non-motorists and general collision

# In[44]:


plt.figure(figsize = (10,5))
ax = sns.countplot(x="year", hue="dataframe", data=result)


# In[ ]:





# In[45]:


g = sns.catplot(x="year", hue="dataframe", col="Weather",
                data=result, kind="count",
                height=10, aspect=.7);


# In[ ]:





# #### * Still wrking on joining the data

# In[ ]:




