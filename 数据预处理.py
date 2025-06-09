#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#加载csv文件
df = pd.read_csv('global-co2-fossil-plus-land-use.csv')

#检查该数据集的基本信息
print("数据集基本信息：")
print(df.info())
print("\n数据集统计描述：")
print(df.describe())

#查看前几行数据
print("\n数据预览：")
print(df.head())


# In[2]:


#由于年份范围太广，现在筛选出其中2011-2021年份的数据
df_filtered = df[(df['Year'] >= 2011) & (df['Year'] <= 2021)]

#Code列可以去除不影响数据分析
df_filtered = df_filtered.drop(columns=['Code'])

#检查缺失值
print("缺失值统计：")
print(df_filtered.isnull().sum())
#检查重复值
print("重复值的数量：",df_filtered.duplicated().sum())

#处理缺失值：删除含有缺失值的行
df_filtered.dropna(inplace=True)


# In[3]:


#异常值检测：利用IQR方法剔除异常值
def remove_outliers(df,col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3-Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

related_cols = ['Annual CO₂ emissions','Annual CO₂ emissions including land-use change',
               'Annual CO₂ emissions from land-use change']
for col in related_cols:
    if col in df_filtered.columns:
        df_filtered = remove_outliers(df_filtered,col)


# In[10]:


#标准化：Z-score标准化
#scaler = StandardScaler()
#df_filtered['Annual CO₂ emissions_标准化'] = scaler.fit_transform(df_filtered[['Annual CO₂ emissions']])
#df_filtered['Annual CO₂ emissions including land-use change_标准化'] = scaler.fit_transform(df_filtered[['Annual CO₂ emissions including land-use change']])
#df_filtered['Annual CO₂ emissions from land-use change_标准化'] = scaler.fit_transform(df_filtered[['Annual CO₂ emissions from land-use change']])
    
#归一化：Min-Max 归一化
#min_max_scaler = MinMaxScaler()
#df_filtered['Annual CO₂ emissions_归一化'] = min_max_scaler.fit_transform(df_filtered[['Annual CO₂ emissions']])
#df_filtered['Annual CO₂ emissions including land-use change_归一化'] = min_max_scaler.fit_transform(df_filtered[['Annual CO₂ emissions including land-use change']])
#df_filtered['Annual CO₂ emissions from land-use change_归一化'] = min_max_scaler.fit_transform(df_filtered[['Annual CO₂ emissions from land-use change']])


# In[4]:


#最后保存文件
df_filtered.to_csv('filtered_data.csv',index=False)


# # 第二个数据集

# In[3]:


#加载csv文件
df = pd.read_csv('E:\数据\Carbon_(CO2)_Emissions_by_Country.csv')

#检查该数据集的基本信息
print("数据集基本信息：")
print(df.info())
print("\n数据集统计描述：")
print(df.describe())

#查看前几行数据
print("\n数据预览：")
print(df.head())


# In[4]:


# 确保Date列是日期格式
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# 按照Country和Date排序
df_sorted = df.sort_values(by=['Country', 'Date'])

#检查缺失值
print("缺失值统计：")
print(df_sorted.isnull().sum())
#检查重复值
print("重复值的数量：",df_sorted.duplicated().sum())

#缺失值和重复值不存在不用处理

#异常值检测：利用IQR方法剔除异常值
def remove_outliers(df,col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3-Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

related_cols = ['Kilotons of Co2','Metric Tons Per Capita']
for col in related_cols:
    if col in df_sorted.columns:
        df_sorted = remove_outliers(df_sorted,col)
        
#标准化：Z-score标准化
scaler = StandardScaler()
df_sorted['Kilotons of Co2_标准化'] = scaler.fit_transform(df_sorted[['Kilotons of Co2']])
df_sorted['Metric Tons Per Capita_标准化'] = scaler.fit_transform(df_sorted[['Metric Tons Per Capita']])

    
#归一化：Min-Max 归一化
min_max_scaler = MinMaxScaler()
df_sorted['Kilotons of Co2_归一化'] = min_max_scaler.fit_transform(df_sorted[['Kilotons of Co2']])
df_sorted['Metric Tons Per Capita_归一化'] = min_max_scaler.fit_transform(df_sorted[['Metric Tons Per Capita']])


# In[5]:


#最后保存文件
df_sorted.to_csv('sorted_file.csv', index=False)


# # 第三个数据集

# In[6]:


#加载csv文件
df = pd.read_csv('E:\数据\global-data-on-sustainable-energy (1).csv')

#由于只有其中一些数据需要用到，所以要删除一些列，定义要保留的列
keep = ['Entity','Year','Renewable-electricity-generating-capacity-per-capita','Renewable energy share in the total final energy consumption (%)',
       'Electricity from fossil fuels (TWh)','Electricity from nuclear (TWh)','Electricity from renewables (TWh)','Primary energy consumption per capita (kWh/person)',
       'Value_co2_emissions_kt_by_country','gdp_growth','gdp_per_capita']

df = df[keep]

#检查该数据集的基本信息
print("数据集基本信息：")
print(df.info())
print("\n数据集统计描述：")
print(df.describe())

#查看前几行数据
print("\n数据预览：")
print(df.head())


# In[7]:


#检查缺失值
print("缺失值统计：")
print(df.isnull().sum())
#检查重复值
print("重复值的数量：",df.duplicated().sum())

#处理缺失值-用中位数进行填充
df['Renewable-electricity-generating-capacity-per-capita'].fillna(df['Renewable-electricity-generating-capacity-per-capita'].median(),inplace=True)
df['Renewable energy share in the total final energy consumption (%)'].fillna(df['Renewable energy share in the total final energy consumption (%)'].median(),inplace=True)
df['Electricity from fossil fuels (TWh)'].fillna(df['Electricity from fossil fuels (TWh)'].median(),inplace=True)
df['Electricity from nuclear (TWh)'].fillna(df['Electricity from nuclear (TWh)'].median(),inplace=True)
df['Electricity from renewables (TWh)'].fillna(df['Electricity from renewables (TWh)'].median(),inplace=True)
df['Value_co2_emissions_kt_by_country'].fillna(df['Value_co2_emissions_kt_by_country'].median(),inplace=True)
df['gdp_growth'].fillna(df['gdp_growth'].median(),inplace=True)
df['gdp_per_capita'].fillna(df['gdp_per_capita'].median(),inplace=True)


# In[8]:


#异常值检测：利用IQR方法剔除异常值
def remove_outliers(df,col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3-Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

related_cols = ['Renewable-electricity-generating-capacity-per-capita','Renewable energy share in the total final energy']
for col in related_cols:
    if col in df.columns:
        df = remove_outliers(df,col)


# In[9]:


#最后保存文件
df.to_csv('newfile.csv', index=False)


# # 第四个数据集

# In[10]:


#加载csv文件
df = pd.read_csv('E:\数据\energy.csv')

#检查该数据集的基本信息
print("数据集基本信息：")
print(df.info())
print("\n数据集统计描述：")
print(df.describe())

#查看前几行数据
print("\n数据预览：")
print(df.head())


# In[11]:


#检查缺失值
print("缺失值统计：")
print(df.isnull().sum())
#检查重复值
print("重复值的数量：",df.duplicated().sum())
#处理缺失值：删除含有缺失值的行
df.dropna(inplace=True)


# In[12]:


#异常值检测：利用IQR方法剔除异常值
def remove_outliers(df,col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3-Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

related_cols = ['Energy_consumption','Energy_production','GDP','Population','Energy_intensity_per_capita','Energy_intensity_by_GDP','CO2_emission']
for col in related_cols:
    if col in df.columns:
        df = remove_outliers(df,col)


# In[13]:


#最后保存文件
df.to_csv('newfile2.csv', index=False)


# # 第五个数据集

# In[14]:


#加载csv文件
df = pd.read_csv('E:\数据\climate_change_data.csv')

#检查该数据集的基本信息
print("数据集基本信息：")
print(df.info())
print("\n数据集统计描述：")
print(df.describe())

#查看前几行数据
print("\n数据预览：")
print(df.head())


# In[15]:


#检查缺失值
print("缺失值统计：")
print(df.isnull().sum())
#检查重复值
print("重复值的数量：",df.duplicated().sum())
#处理缺失值：删除含有缺失值的行
df.dropna(inplace=True)


# In[16]:


#缺失值和重复值不存在不用处理

#异常值检测：利用IQR方法剔除异常值
def remove_outliers(df,col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3-Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

related_cols = ['Temperature','CO2 Emissions','Humidity','Wind Speed']
for col in related_cols:
    if col in df.columns:
        df = remove_outliers(df,col)


# In[17]:


#最后保存文件
df.to_csv('newfile3.csv', index=False)


# In[ ]:




