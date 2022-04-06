#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Project 


# In[1]:


#import required libraries

import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder # Label Encoder
labelencoder = LabelEncoder()  # creating instance of labelencoer
import re # for regular expression


# In[2]:


#import dataset 
data=pd.read_excel(r"C:\Users\User\Desktop\data science\project\\Data (1).xlsx")


# In[3]:


#EDA
data.info()
data.columns
data.dtypes

# Checking duplicates
duplicate = data.duplicated()
duplicate
sum(duplicate) 

# drop irrelevant columns
data=data.drop(columns={"Year","Month","Day"},axis=1)


# #Converting all categorical column into mumeric form.

# In[4]:


# For breakfast column
data["1. What did you eat for breakfast YESTERDAY?"].value_counts()
data = data.rename(columns={"1. What did you eat for breakfast YESTERDAY?": 'breakfast'})

for i in range(0,986): 
    if re.search('Healthy', data['breakfast'][i]):
        data['breakfast'][i] = 1     #Healthy_cereal
    elif re.search('Sugary', data['breakfast'][i]):
        data['breakfast'][i] = 2      #Sugary_cereal
    elif re.search('Toast', data['breakfast'][i]):
        data['breakfast'][i] = 3       #Toast
    elif re.search('Cooked', data['breakfast'][i]):
        data['breakfast'][i] = 4      #Cooked_breakfast
    elif re.search('Fruit', data['breakfast'][i]):
        data['breakfast'][i] = 5      #Fruit
    elif re.search('Nothing', data['breakfast'][i]):
        data['breakfast'][i] = 6       #Nothing
    else:
        data['breakfast'][i] = 7       #Others

data.breakfast.value_counts()


# In[5]:


data["Are you still going to school?"].value_counts()


# In[6]:


data=data.replace({"Are you still going to school?":{"No, I am at home":"no","Yes, most days of the week":"Yes",
            "Yes, sometimes":"Yes","I am in a different school from my own school":"Yes"}})

data['Are you still going to school?']= labelencoder.fit_transform(data['Are you still going to school?'])


# In[7]:


data["How many people live in your home with you (including adults)?"].value_counts()


# In[8]:


data=data.replace({"How many people live in your home with you (including adults)?":{"6+":6}})


# In[9]:


data["What year are you in now?"].value_counts()


# In[10]:
data["3. How many times did you brush your teeth YESTERDAY?"].value_counts()

data=data.replace({"What year are you in now?":{"Year 3":3,"Year 4":4,"Year 5":5,"Year 6":6}})


# Convert all binary columns into numeric form using label encoder

# In[11]:


data['Do you have any other children living in your house with you?']= labelencoder.fit_transform(data['Do you have any other children living in your house with you?'])
data['14. From your house, can you easily walk to a park (for example a field or grassy area)?']= labelencoder.fit_transform(data['14. From your house, can you easily walk to a park (for example a field or grassy area)?'])
data['15. From your house, can you easily walk to somewhere you can play?']= labelencoder.fit_transform(data['15. From your house, can you easily walk to somewhere you can play?'])
data['16. Do you have a garden?']= labelencoder.fit_transform(data['16. Do you have a garden?'])
data["25. Are you able to keep in touch with your family that you don't live with? (grand parents, Uncle, Aunt, Cousins, etc)"].value_counts()
data=data.replace({"25. Are you able to keep in touch with your family that you don't live with? (grand parents, Uncle, Aunt, Cousins, etc)":{9: "Yes"}})
data["25. Are you able to keep in touch with your family that you don't live with? (grand parents, Uncle, Aunt, Cousins, etc)"]= labelencoder.fit_transform(data["25. Are you able to keep in touch with your family that you don't live with? (grand parents, Uncle, Aunt, Cousins, etc)"])

data['26. Are you able to keep in touch with your friends?']= labelencoder.fit_transform(data['26. Are you able to keep in touch with your friends?'])
data['Gender']= labelencoder.fit_transform(data['Gender'])


# Me and my feelings columns convert into numeric form

# In[12]:


data=data.replace({"24. Remember, there are no right or wrong answers, just pick which is right for you. [I am calm]":{"Never": 2, "Sometimes": 1, "Always": 0}})


# In[13]:


data.replace({"Never": 0, "Sometimes": 1, "Always": 2}, inplace=True)


# Calculating sleeping hours from sleeping time and wake up time

# In[14]:


data = data.rename(columns={"4. What time did you fall asleep YESTERDAY (to the nearest half hour)?": 'sleeping_start_time'})

data = data.rename(columns={"5. What time did you wake up TODAY (to the nearest half hour)?": 'wakeup_time'})

data['sleeping_start_time'] = pd.to_datetime(data.sleeping_start_time)
data['wakeup_time'] = pd.to_datetime(data.wakeup_time)

sleeping_hours = ((data['wakeup_time']) - (data['sleeping_start_time']))

sleeping_hours.dt.components
sleeping_hour = pd.DataFrame(sleeping_hours.dt.components.hours)
sleeping_minutes = pd.DataFrame(sleeping_hours.dt.components.minutes)

sleeping_hour['sleeping_minutes'] = sleeping_minutes

sleeping_hour = sleeping_hour.astype('str')
sleeping_hour = sleeping_hour['hours'].str.cat(sleeping_hour['sleeping_minutes'], sep =".")

sleeping_hour.astype(float)
data['sleeping_hours'] = sleeping_hour

data['sleeping_hours']=data['sleeping_hours'].astype(float)
data = data.drop(['sleeping_start_time','wakeup_time'],axis=1)


# In[15]:


data["17. How often do you go out to play outside?"].value_counts()

data['17. How often do you go out to play outside?']= labelencoder.fit_transform(data['17. How often do you go out to play outside?'])


# In[16]:


data["2. Did you eat any fruit and vegetables YESTERDAY? "].value_counts()

data=data.replace({"2. Did you eat any fruit and vegetables YESTERDAY? ":{"No":0,"1 Piece":1,"2 Or More Fruit and Veg":2}})


# In[17]:


# For columns 7 to 14
data.replace({"0 days" :0 ,"1-2 days":1 ,"3-4 days" : 2,"5-6 days":3,"7 days":4}, inplace=True)


# In[18]:


data["18. Do you have enough time for play?"].value_counts()
data=data.replace({"18. Do you have enough time for play?":{"No, I would like to have a bit more":0,"No, I need a lot more":0,"Yes, I have loads":1, "Yes, it's just about enough":1}})


# In[19]:


data["17. How often do you go out to play outside?"].value_counts()

data['17. How often do you go out to play outside?']= labelencoder.fit_transform(data['17. How often do you go out to play outside?'])


# In[20]:


data["19. What type of places do you play in?"].value_counts()

data=data.rename(columns={"19. What type of places do you play in?": "playarea"})


# In[21]:


numbers=data["playarea"].value_counts()
count = data.playarea.str.split(';', expand=True).stack().value_counts()

data.playarea=data["playarea"].astype(str)
for i in range(len(data.playarea)):
    if 'In my house'  in data.playarea[i]:
        data.playarea[i]= 1
    elif 'my garden' in data.playarea[i]:
        data.playarea[i]=2
    else:
        data.playarea[i]=3

data["playarea"].value_counts()


# In[22]:


data["20. Can you play in all the places you would like to?"].value_counts()

data=data.replace({"20. Can you play in all the places you would like to?":{"I can play in some of the places I would like to":1,
                                    "I can only play in a few places I would like to":1,
                                    "I can play in all the places I would like to": 2,
                                    "I can hardly play in any of the places I would like to": 0}})


# In[23]:


data["21. Do you have somewhere at home where you have space to relax?"].value_counts()
data['21. Do you have somewhere at home where you have space to relax?']= labelencoder.fit_transform(data['21. Do you have somewhere at home where you have space to relax?'])


# In[24]:


data["22. Tell us if you agree or disagree with the following: [I am doing well with my school work]"].value_counts()

data.replace({"Agree":1,"Strongly agree":1,"Don't agree or disagree":0,"Disagree":0, "Strongly disagree":0},inplace=True)


# In[25]:


data["27. If yes, how are you keeping in touch (tick all you use)?"].value_counts()

data=data.rename(columns={"27. If yes, how are you keeping in touch (tick all you use)?": "ways_tokeeping_touch"})


# In[26]:


count_number = data["ways_tokeeping_touch"].str.split(';', expand=True).stack().value_counts()



# In[28]:


count_number = data["ways_tokeeping_touch"].str.split(';', expand=True).stack().value_counts()

data.ways_tokeeping_touch=data["ways_tokeeping_touch"].astype(str)

for i in range(len(data.ways_tokeeping_touch)):
    if 'I live near them so I can see them' in data["ways_tokeeping_touch"][i]:
        data["ways_tokeeping_touch"][i]= 1
  
    else:
        data["ways_tokeeping_touch"][i]=2

data.ways_tokeeping_touch=data["ways_tokeeping_touch"].astype(int)


# make two columns from me and my feelings columns

# In[29]:


# Behavioural Dificulties
#rest of the columns of output columns contain behavioural difficulties features
BD_columns = data.iloc[:, 45:51]
BD_columns.loc[:,'BD_subscle'] = BD_columns.sum(axis=1)
data['BD_subscale'] = BD_columns['BD_subscle']


# In[68]:


for i in range(986):
     if data.BD_subscale[i]<=5:
            data.BD_subscale[i]="expected"
     elif data.BD_subscale[i]==6:
             data.BD_subscale[i]="borderline difficulties"
     elif data.BD_subscale[i]>=7:
             data.BD_subscale[i]="clinically significant difficulties"
             


# In[69]:


#Fill all null value by median imputation


# In[70]:


data.fillna(data.median(), inplace=True)


# Behavioural dataset

# In[37]:


#Rearrange columns 


# In[77]:


data1=data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,
                  34,51,52,53,54,55]]



# make new dataframe for emotional dataset

# In[101]:


Behavioural_dataset=data1.iloc[:, :40]
bd_x=Behavioural_dataset.drop(columns={"BD_subscale","ID","Gender"},axis=1)
bd_y=Behavioural_dataset.BD_subscale


# Model building for emotional dataset

# In[102]:


from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


# In[103]:


x_train, x_test, y_train, y_test = train_test_split(bd_x, bd_y, test_size = 0.2,random_state=0)


# In[104]:


# visualize the target variable
g = sns.countplot(Behavioural_dataset.BD_subscale)
g.set_xticklabels(['expected ','borderline difficulties','clinically significant difficulties'])
plt.show()


# Data is highly imbalanced.
# Therefore here I am using oversample method 

# In[105]:


from collections import Counter
from imblearn.over_sampling import SMOTE
# define dataset

# summarize class distribution
counter = Counter(bd_y)
print(counter)
# transform the dataset
oversample = SMOTE()
X, y = oversample.fit_resample(bd_x,bd_y)
# summarize the new class distribution
counter = Counter(y)
print(counter)


# In[106]:


#Features selection for emotional dataset


# In[107]:


import pandas as pd
from sklearn.feature_selection import chi2

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features


feat_importances = pd.Series(fit.scores_, index=dfcolumns)
feat_importances.nlargest(10).plot(kind='barh')


# In[ ]:


#Create new variable with selecting feature.


# In[108]:


x=X.iloc[:,[28,31,27,32,10,9,6,18,23,25]]

# In[109]:

# visualize the selecting variable
import seaborn as sns
correlation=x.corr()
ax=sns.heatmap(correlation,annot=True)

# In[109]:


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2,random_state=85)


# In[113]:


from sklearn.ensemble import RandomForestClassifier
bmodel = RandomForestClassifier(criterion='entropy', max_depth= 11, max_features= 'log2', n_estimators= 100,min_samples_leaf=3)
bmodel.fit(X_train,Y_train)
pred2=bmodel.predict(X_test)
print(classification_report(Y_test, pred2 ))
accuracy_score(Y_test, pred2) #92
predict_=bmodel.predict(X_train)
accuracy_score(Y_train,predict_)


# In[ ]:


import pickle
pickle.dump(bmodel,open('bmodel.pkl','wb'))

bmodel=pickle.load(open("bmodel.pkl",'rb'))
print(bmodel.predict([[1,1,1,1,1,1,1,1,1,1]]))
