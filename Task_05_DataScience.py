import numpy as np 
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
for dirname, _, filenames in os.walk('D:\Prodigy_Projects'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


datafile_USA = pd.read_csv('F:/Prodigy_Projects/Task5/dataset/Task_05_DataScience.py')

print(datafile_USA.head())
print(datafile_USA.columns)
print(datafile_USA.dtypes.value_counts())
print(datafile_USA.shape)
print(datafile_USA.describe())
print(datafile_USA.State.unique())
datafile1_USA=datafile_USA[datafile_USA['State']=='CA']
datafile1_USA['IDD'] = datafile_USA['ID'].astype('str').str.extractall('(\d+)').unstack().fillna('').sum(axis=1).astype(int)
print(datafile1_USA)
print(datafile1_USA.head())
print(datafile1_USA.shape)
print(datafile1_USA.columns)
print(datafile1_USA.duplicated().sum())
datafile1_USA=datafile1_USA.dropna(subset=['Precipitation(in)'])    
print(datafile1_USA.shape)
print(datafile1_USA.isna().sum()/len(datafile1_USA)*100)
datafile1_USA=datafile1_USA.dropna(subset=['City','Sunrise_Sunset',
       'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight'])
print(datafile1_USA.isna().sum()/len(datafile1_USA)*100)
print(datafile1_USA['Weather_Condition'].value_counts())
#print(datafile1_USA.Side.unique())

datafile_USA_cat=datafile1_USA.select_dtypes('object')
datafile_USA_num=datafile1_USA.select_dtypes(np.number)
datafile_USA_cat=datafile_USA_cat.drop('ID',axis=1)

datafile_USA_cat=datafile1_USA.select_dtypes('object')
col_name=[]
length=[]

for i in datafile_USA_cat.columns:
    col_name.append(i)
    length.append(len(datafile_USA_cat[i].unique()))
datafile_USA_2=pd.DataFrame(zip(col_name,length),columns=['feature','count_of_unique_values'])
print(datafile_USA_2)
datafile1_USA.drop(['Description','Zipcode','Weather_Timestamp'],axis=1,inplace=True)
del datafile1_USA['Airport_Code']
print(datafile_USA_num.columns)
print(len(datafile_USA_num.columns))
print(datafile_USA_cat.columns)
print(len(datafile_USA['City'].unique()))
datafile_USA_num=datafile1_USA.select_dtypes(np.number)
col_name=[]
length=[]

for i in datafile_USA_num.columns:
    col_name.append(i)
    length.append(len(datafile_USA_num[i].unique()))
datafile2_USA=pd.DataFrame(zip(col_name,length),columns=['feature','count_of_unique_values'])
print(datafile2_USA)
plt.figure(figsize=(15 ,9))
sns.heatmap(datafile_USA_num.corr() , annot=True)
cities = datafile1_USA['City'].unique()
print(len(cities))
accidents_by_cities = datafile1_USA['City'].value_counts()
print(accidents_by_cities)
print(accidents_by_cities[:10])

#Plotiing graph
fig, ax = plt.subplots(figsize=(8,5))
accidents_by_cities[:10].plot(kind='bar')
ax.set(title = 'Top 10 cities By Number of Accidents',
       xlabel = 'Cities',
       ylabel = 'Accidents Count')
plt.show()


accidents_severity = datafile1_USA.groupby('Severity').count()['ID']
print(accidents_severity)
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(aspect="equal"))
label = [1,2,3,4]
plt.pie(accidents_severity, labels=label,
        autopct='%1.1f%%', pctdistance=0.85)
circle = plt.Circle( (0,0), 0.5, color='white')
p=plt.gcf()
p.gca().add_artist(circle)
ax.set_title("Accident by Severity",fontdict={'fontsize': 16})
plt.tight_layout()
plt.show()

print(datafile1_USA['Start_Time'].dtypes)
print(datafile1_USA['End_Time'].dtypes)
datafile1_USA = datafile1_USA.astype({'Start_Time': 'datetime64[ns]', 'End_Time': 'datetime64[ns]'})
print(datafile1_USA['Start_Time'].dtypes)
#print(datafile1_USA['Start_Time'][2408])
#print(datafile1_USA['End_Time'][2408])
datafile1_USA['start_date'] = [d.date() for d in datafile1_USA['Start_Time']]
datafile1_USA['start_time'] = [d.time() for d in datafile1_USA['Start_Time']]

datafile1_USA['end_date'] = [d.date() for d in datafile1_USA['End_Time']]
datafile1_USA['end_time'] = [d.time() for d in datafile1_USA['End_Time']]
datafile1_USA['end_time']

fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(datafile1_USA['Start_Time'].dt.hour, bins = 24)

plt.xlabel("Start Time")
plt.ylabel("Number of Occurence")
plt.title('Accidents Count By Time of Day')

plt.show()

fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(datafile1_USA['Start_Time'].dt.hour, bins = 24)

plt.xlabel("End_Time")
plt.ylabel("Number of Occurence")
plt.title('Accidents Count By Time of Day')

plt.show()

del datafile1_USA['Start_Time']
del datafile1_USA['End_Time']
#%matplotlib inline
import os

fig, ax = plt.subplots(figsize=(8, 5))
weather_conditions = datafile1_USA['Weather_Condition'].value_counts()  # Defined weather_conditions
ax.set(title='Weather Conditions at Time of Accident Occurrence', xlabel='Weather', ylabel='Accidents Count')
weather_conditions.sort_values(ascending=False)[:20].plot(kind='bar')
plt.show()

print(datafile_USA.shape)
print(datafile_USA_num.shape)
print(datafile1_USA.groupby('Severity').count()['IDD'])
datafile_USA_num.plot(kind='scatter', y='Start_Lat', x='Severity')
sns.jointplot(x=datafile_USA_num.Start_Lat.values , y=datafile_USA_num.Start_Lng.values,height=10)
plt.ylabel('Start lattitude', fontsize=12)
plt.xlabel('Start lattitude', fontsize=12)
plt.show()

sns.jointplot(x=datafile_USA_num.End_Lat.values , y=datafile_USA_num.End_Lng.values,height=10)
plt.ylabel('end lattitude', fontsize=12)
plt.xlabel('end longitude', fontsize=12)
plt.show()

