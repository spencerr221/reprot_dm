---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(visualization1)=

# Visualization of Dataset

## Overview

In order to have a better understanding of the dataset, we analyze it from various aspects. The graphs below express how different attack target, attack region and other important features change over the years. 

## Details

The first one is about the number of terrorist activities. 

First, there was two obvious continuous growth periods throughout the forty seven years. The first peroid is from 1970 to 1994 and another period starts after 2000's. After those two different upsurges, the activities in second period (in 2000's, the number is over 10000) is noticeably third of that in the former period. It's shown that the attack frequency contributes a pattern like "rampant-governance-convergence-non governance-rampant again".

The frequency of terrorist attacks in 2014 reached the peak in five years. It may be related to the kidnapping in Tikrit City, Saladin Province, Iraq in June 2014 - about 1686 soldiers were kidnapped and all died. The terrorist attack was the deadliest terrorist attack in the year.

```{code-block} python3
plt.subplots(figsize=(15,6))
sns.countplot('Year',data=terror,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Number Of Terrorist Activities Each Year')
plt.show()
```

```{figure} /_static/lecture_specific/VilsPlot/1.png
```

The second one aims at the attacking methods by terrorists. Nearly 80000 terrorist attacks have used explosions/bombing while only a very small part is hijacking.

```{code-block} python3
plt.subplots(figsize=(15,6))
sns.countplot('AttackType_Name',data=terror,palette='inferno',order=terror['AttackType_Name'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Attacking Methods by Terrorists')
plt.show()
```

```{figure} /_static/lecture_specific/VilsPlot/2.png
```


The third one is the main victimized targets in the terrorist attacks.There are many purposes for launching the attack, among which the reason for resource plundering ranks first.

This kind of target data can be connected with real-world events. Even without the method of building models, it can also achieve the purpose of prevention by analyzing the recent economic and political forms of various regions.

```{code-block} python3
plt.subplots(figsize=(15,6))
sns.countplot(terror['Targtype_Name'],palette='inferno',order=terror['Targtype_Name'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Main Targets')
plt.show()
```

```{figure} /_static/lecture_specific/VilsPlot/3.png
```


Then, here comes to the numbers of cases of different attack types in different regions. 

Compared with the display of regional data alone, the combination of attack methods and regions can better reflect the authenticity and has more practical significance. It's can be seen that South Asia, Middle East and North Africa sufferes a lot of attacks and most of the attack is bombing, which is consistent with the previous plot of attackng methods.

We can also think about the main reason why these three places become the focus of casualties although the information is not shown here. The first reason is resource plunder, and another reason is the struggle between political factions and religious groups, which usually makes the government compromise with its opponents by making terrorist attacks in this region.

```{code-block} python3
pd.crosstab(terror.Region_Name,terror.AttackType_Name).plot.barh(stacked=True,width=1,color=sns.color_palette('RdYlGn',9))
fig=plt.gcf()
fig.set_size_inches(12,8)
plt.show()
```

```{figure} /_static/lecture_specific/VilsPlot/4.png
```

Next, you can see the numbers of attacks and person who get killed in different countries. Iraq has the highest number of casualties, which are relatively tragic. Also, It's can be observed that in cities such as Iraq, Pakistan and Afghanistan, the number of conflicts is small but the casualties are heavy. However, What many people don't know is that terrorist attacks have also occurred in Thailand, Turkey and Spain while they are all popular tourist cities.

```{code-block} python3
coun_terror=terror['Country_Name'].value_counts()[:15].to_frame()
coun_terror.columns=['Attacks']
coun_kill=terror.groupby('Country_Name')['Killed'].sum().to_frame()
coun_terror.merge(coun_kill,left_index=True,right_index=True,how='left').plot.bar(width=0.9)
fig=plt.gcf()
fig.set_size_inches(18,6)
plt.show()
```

```{figure} /_static/lecture_specific/VilsPlot/5.png
```

The two pictures below shows activities of top 10 terrorist groups ,number of attacks in regions from 1970 to 2017 respectively. This chart echoes the first analysis, especially the sudden sharp rise in the number of attacks around 2010. Some people speculate that this may be related to the financial crisis in 2008.

During the presentation, one student asked us why there seemed to be no terrorist attacks during 2000-2010 and whether we did anything. At that time, because there was not clear to which picture it was, the answer was not accurate enough. As can be seen from the following two figures, there are many conflicts, but: 1) the unit of the vertical axis is large and the broken line is thick, so the data looks very close to 0. 2) Different broken lines cover each other, which will also affect the judgment. Therefore, we did not intentionally delete some data.

```{code-block} python3
top_groups10=terror[terror['Group_Name'].isin(terror['Group_Name'].value_counts()[1:11].index)]
pd.crosstab(top_groups10.Year,top_groups10.Group_Name).plot(color=sns.color_palette('Paired',10))
fig=plt.gcf()
fig.set_size_inches(18,6)
plt.show()
```

```{figure} /_static/lecture_specific/VilsPlot/6.png
```
```{figure} /_static/lecture_specific/VilsPlot/7.png
```

At last, we can also use this dataset to calculate the terrorism risk in percentage for every months in specific country. Here we use Mexico as an example.This time-related data can be deeply mined but in this exercise, we are not committed to this direction.

```{code-block} python3
country = 'Mexico'
months = dict(terror[terror.Country_Name == country].groupby(['Month']).size())
months = dict(sorted(months.items(), key = lambda kv:(kv[1], kv[0]), reverse=True))
tot = sum(list(months.values()))
print("Entered Country: ",country)
for i in months:
    print(i,' {:.2f} %'.format((months[i]/tot)*100))
```

```{code-block} python3
Entered Country:  Mexico
1  15.46 %
9  14.43 %
11  12.37 %
2  10.31 %
3  9.28 %
10  7.22 %
5  7.22 %
6  6.19 %
7  5.15 %
4  5.15 %
8  4.12 %
12  3.09 %
```

## Summary
The several examples I mention above show that these data can correspond to real-world news, making it highly credible. Last but not the least, the data we show is not all and the complete dataset has more features that can be used for visualization and analysis.

Through the above methods, we can concentrate and extract the information hidden in a large number of seemingly chaotic data, and find out the internal laws of different data characteristics. Hence,The step of data visualization lays a foundation for our later data modeling and prediction. For example, the types of weapons and terrorist organizations can be further studied. 