#!/usr/bin/env python
# coding: utf-8

# In[71]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family']='SimHei'        # 设置使用黑体字体以正常显示中文
plt.rcParams['axes.unicode_minus']=False   # 正常显示负号

#加载csv文件
df = pd.read_csv('filtered_data.csv')
print(df.head())


# In[39]:


#数据探索性分析


# In[72]:


#计算描述性统计
print(df.describe())

#可视化国家碳排放量变化趋势（随机选取几个国家）
sample_entity = df['Entity'].sample(5,random_state=42).values

plt.figure(figsize=(10,6))
for i in sample_entity:
    data = df[df['Entity'] == i]
    plt.plot(data['Year'],data['Annual CO₂ emissions'],marker='o',label=i)
    
plt.title("国家碳排放量变化趋势")
plt.xlabel("年份")
plt.ylabel("碳排放量")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[27]:


#时序聚类分析
#安装并导入所需库
get_ipython().system('pip install tslearn')


# In[73]:


from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# 构建时序数据
time_series_data = df.pivot(index="Entity", columns="Year", values="Annual CO₂ emissions").fillna(method='ffill', axis=1)

# 标准化数据
scaler = TimeSeriesScalerMeanVariance()
scaled_data = scaler.fit_transform(time_series_data.values)

# DTW + K-means聚类
dtw_km = TimeSeriesKMeans(n_clusters=3, metric="dtw", random_state=42)
clusters = dtw_km.fit_predict(scaled_data)

# 将聚类结果添加到DataFrame
time_series_data["Cluster"] = clusters

# 绘制聚类结果图
for cluster in np.unique(clusters):
    plt.figure(figsize=(8, 4))
    plt.title(f"Cluster {cluster} 碳排放量变化模式")
    for ts in scaled_data[clusters == cluster]:
        plt.plot(ts.ravel(), alpha=0.3)
    plt.xlabel("年份")
    plt.ylabel("标准化碳排放量")
    plt.xticks(ticks=range(len(time_series_data.columns)-1), labels=time_series_data.columns[:-1], rotation=45)
    plt.tight_layout()
    plt.show()


# In[74]:


#单个国家碳排放量绘制与分析
entity_id = sample_entity[0]  # 举例第一个国家
entity_data = df[df["Entity"] == entity_id]

plt.figure(figsize=(10, 5))
plt.plot(entity_data["Year"], entity_data["Annual CO₂ emissions"], marker='o', linestyle='-', linewidth=2)
plt.title(f"国家 {entity_id} 的碳排放量变化曲线")
plt.xlabel("年份")
plt.ylabel("碳排放量")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[75]:


#按国家计算2011-2021年总碳排放量和年平均碳排放量
entity_sum = df.groupby('Entity').agg(
    Total_Carbon_Emissions=('Annual CO₂ emissions','sum'),
    Average_Carbon_Emissions=('Annual CO₂ emissions','mean')
).reset_index()

#显示汇总结果
print(entity_sum)


# In[76]:


#可视化数据
import seaborn as sns
#设置可视化风格
sns.set(style="whitegrid")

plt.rcParams['font.family']='SimHei'        # 设置使用黑体字体以正常显示中文
plt.rcParams['axes.unicode_minus']=False   # 正常显示负号

#按国家绘制总碳排放量柱状图
plt.figure(figsize=(12, 20))
sns.barplot(x='Total_Carbon_Emissions', y='Entity', data=entity_sum, palette='viridis')
plt.xlabel('总碳排放量')
plt.ylabel('国家')
plt.title('各国家2011-2021的总碳排放量')
plt.show()


# # 第二个数据集

# In[77]:


#加载csv文件
df = pd.read_csv('sorted_file.csv')
print(df.head())


# In[78]:


#数据探索性分析
#计算描述性统计
print(df.describe())


# In[79]:


#按地区计算1990-2019年1.1总碳排放量和人均碳排放量
region_group = df.groupby('Region').agg(
    total_co2=('Kilotons of Co2','sum'),
    avg_per_capita_co2=('Metric Tons Per Capita','mean')
).reset_index()

#显示汇总结果
print(region_group)


# In[80]:


#可视化数据
import seaborn as sns
#设置可视化风格
sns.set(style="whitegrid")

plt.rcParams['font.family']='SimHei'        # 设置使用黑体字体以正常显示中文
plt.rcParams['axes.unicode_minus']=False   # 正常显示负号

#按地区绘制总碳排放量柱状图
plt.figure(figsize=(12, 6))
sns.barplot(x='total_co2', y='Region', data=region_group, palette='viridis')
plt.xlabel('总碳排放量')
plt.ylabel('地区')
plt.title('各地区1990-2019年1.1总碳排放量')
plt.show()


# In[81]:


#按地区绘制人均碳排放量柱状图
plt.figure(figsize=(12, 6))
sns.barplot(x='avg_per_capita_co2', y='Region', data=region_group, palette='viridis')
plt.xlabel('人均碳排放量')
plt.ylabel('地区')
plt.title('各地区1990-2019年1.1人均碳排放量')
plt.show()


# In[82]:


#可视化地区碳排放量变化趋势

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
# 按Region和Date进行分组，计算每个Region在每年1.1的总碳排放量
df['Year'] = df['Date'].dt.year  # 提取年份
df_grouped = df.groupby(['Region', 'Year'])['Kilotons of Co2'].sum().reset_index()

plt.figure(figsize=(10,6))
sns.lineplot(data=df_grouped, x='Year', y='Kilotons of Co2', hue='Region', marker='o')
    
plt.title("地区碳排放量变化趋势")
plt.xlabel("年份")
plt.ylabel("碳排放量")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# # 第三个数据集

# In[83]:


#加载csv文件
df = pd.read_csv('newfile.csv')
print(df.head())


# In[84]:


#时序聚类分析
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# 构建时序数据
time_series_data = df.pivot(index="Entity", columns="Year", values="gdp_growth").fillna(method='ffill', axis=1)

# 标准化数据
scaler = TimeSeriesScalerMeanVariance()
scaled_data = scaler.fit_transform(time_series_data.values)

# DTW + K-means聚类
dtw_km = TimeSeriesKMeans(n_clusters=3, metric="dtw", random_state=42)
clusters = dtw_km.fit_predict(scaled_data)

# 将聚类结果添加到DataFrame
time_series_data["Cluster"] = clusters

# 绘制聚类结果图
for cluster in np.unique(clusters):
    plt.figure(figsize=(8, 4))
    plt.title(f"Cluster {cluster} GDP增长变化模式")
    for ts in scaled_data[clusters == cluster]:
        plt.plot(ts.ravel(), alpha=0.3)
    plt.xlabel("年份")
    plt.ylabel("标准化GDP增长量")
    plt.xticks(ticks=range(len(time_series_data.columns)-1), labels=time_series_data.columns[:-1], rotation=45)
    plt.tight_layout()
    plt.show()


# In[53]:


#各国家GDP增长量变化趋势
import random
entities = df['Entity'].unique()  # 获取所有 unique 的 Entity
selected_entities_random = random.sample(list(entities), 5)  # 随机选择5个 Entity
filtered_df_random = df[df['Entity'].isin(selected_entities_random)]

plt.figure(figsize=(10,6))
sns.lineplot(data=filtered_df_random, x='Year', y='gdp_growth', hue='Entity', marker='o')
    
plt.title("各国家GDP增长量变化趋势")
plt.xlabel("年份")
plt.ylabel("GDP增长量")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[85]:


#数据探索性分析
#计算描述性统计
print(df.describe())


# In[86]:


#按国家计算2000-2020年人均可再生电力发电能力、总化石燃料发电量、总核能发电量、总可再生能源发电量、总GDP增长和人均GDP
entity_sum = df.groupby('Entity').agg(
    Average_Renewable_Electricity=('Renewable-electricity-generating-capacity-per-capita','mean'),
    Total_Fossil_Electricity=('Electricity from fossil fuels (TWh)','sum'),
    Total_Nuclear_Electricity=('Electricity from nuclear (TWh)','sum'),
    Total_Renewable_Electricity=('Electricity from renewables (TWh)','sum'),
    Total_GDP_Growth=('gdp_growth','sum'),
    Average_GDP_captia=('gdp_per_capita','mean')
).reset_index()

#显示汇总结果
print(entity_sum)


# In[87]:


#可视化数据
import seaborn as sns
#设置可视化风格
sns.set(style="whitegrid")

plt.rcParams['font.family']='SimHei'        # 设置使用黑体字体以正常显示中文
plt.rcParams['axes.unicode_minus']=False   # 正常显示负号

#绘制各国家人均可再生电力发电能力柱状图
# 随机选择N个Entity，假设随机选择5个实体
n = 5  # 你可以修改这个数字来控制选取的数量
random_entities = entity_sum['Entity'].sample(n=n, random_state=42)  # random_state设置为42确保结果可复现

# 按随机选定的Entity筛选数据
filtered_data = entity_sum[entity_sum['Entity'].isin(random_entities)]

plt.figure(figsize=(12, 6))
sns.barplot(x='Average_Renewable_Electricity', y='Entity', data=filtered_data, palette='viridis')
plt.xlabel('人均可再生电力发电能力')
plt.ylabel('国家')
plt.title('各国家2000-2020年人均可再生电力发电能力')
plt.show()


# In[88]:


#绘制各国家2000-2020总化石燃料发电量和总可再生能源发电量对比图
# 将化石燃料发电量和可再生能源发电量的数据进行合并，以便在同一图上显示
melted_data = filtered_data.melt(id_vars=['Entity'], value_vars=['Total_Fossil_Electricity', 'Total_Renewable_Electricity'],
                                  var_name='Energy_Type', value_name='Generation_Amount')

# 绘制柱状图
plt.figure(figsize=(12, 8))
sns.barplot(x='Generation_Amount', y='Entity', hue='Energy_Type', data=melted_data, palette='viridis')

plt.xlabel('发电量')
plt.ylabel('国家')
plt.title('各国家的2000-2020年总化石燃料和核能发电量比较')
plt.legend(title='能源类型')
plt.show()


# # 第四个数据集

# In[89]:


#加载csv文件
df = pd.read_csv('newfile2.csv')
print(df.head())


# In[90]:


#数据探索性分析
#计算描述性统计
print(df.describe())


# In[93]:


#各国能源消费与能源生产的比较
plt.rcParams['font.family']='SimHei'        # 设置使用黑体字体以正常显示中文
plt.rcParams['axes.unicode_minus']=False   # 正常显示负号

# 随机选择五个国家
random_countries = df['Country'].sample(5, random_state=42)

# 筛选出这五个国家的数据
selected_data = df[df['Country'].isin(random_countries)]

# 绘制能源消费与能源生产的比较
plt.figure(figsize=(10, 6))
sns.barplot(x='Energy_consumption', y='Country', data=selected_data, color='blue', label='Energy Consumption')
sns.barplot(x='Energy_production', y='Country', data=selected_data, color='red', label='Energy Production')
plt.title('各国能源消费与能源生产的比较')
plt.xlabel('能源（消费与生产）')
plt.ylabel('国家')
plt.legend()
plt.show()


# In[94]:


# 绘制能源消费与GDP的关系
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Energy_consumption', y='GDP', data=selected_data, hue='Country', palette='Set1')
plt.title('能源消费与GDP的关系')
plt.xlabel('能源消费')
plt.ylabel('GDP')
plt.legend()
plt.show()


# In[95]:


# 绘制CO2排放与能源消费的关系
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Energy_consumption', y='CO2_emission', data=selected_data, hue='Country', palette='Set2')
plt.title('CO2排放与能源消费的关系')
plt.xlabel('能源消费')
plt.ylabel('CO2排放')
plt.legend()
plt.show()


# # 第五个数据集

# In[96]:


#加载csv文件
df = pd.read_csv('newfile3.csv')
print(df.head())


# In[97]:


#数据探索性分析
#计算描述性统计
print(df.describe())


# In[100]:


# 随机选择5个国家
random_countries = df['Country'].sample(5, random_state=42).tolist()

# 筛选出这些国家的数据
filtered_data = df[df['Country'].isin(random_countries)]

# 设置画布风格
sns.set(style="whitegrid")
plt.rcParams['font.family']='SimHei'        # 设置使用黑体字体以正常显示中文
plt.rcParams['axes.unicode_minus']=False   # 正常显示负号

# 气温与CO2排放的关系
plt.figure(figsize=(8, 6))
sns.scatterplot(data=filtered_data, x='Temperature', y='CO2 Emissions', hue='Country', palette='Set1')
plt.title('气温与二氧化碳排放量的关系')
plt.xlabel('气温')
plt.ylabel('二氧化碳排放量')
plt.legend(title='Country')
plt.show()


# In[102]:


# 各国的降水量比较
plt.figure(figsize=(10, 6))
sns.boxplot(data=filtered_data, x='Country', y='Precipitation', palette='Set2')
plt.title('各国降水量对比')
plt.xlabel('国家')
plt.ylabel('降水量')
plt.show()


# In[103]:


# 国家之间的湿度与风速关系
plt.figure(figsize=(8, 6))
sns.scatterplot(data=filtered_data, x='Humidity', y='Wind Speed', hue='Country', palette='Set3')
plt.title('国家之间的湿度与风速关系')
plt.xlabel('湿度')
plt.ylabel('风速')
plt.legend(title='Country')
plt.show()


# In[ ]:




