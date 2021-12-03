#!/usr/bin/env python
# coding: utf-8

# In[100]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

titanic_df = pd.read_csv('./titanic_train.csv')
titanic_df.head(3)


# In[101]:


print('\n ### train 데이터 정보 ###  \n')
print(titanic_df.info())


# In[102]:


titanic_df['Age'].fillna(titanic_df['Age'].mean(),inplace=True)
titanic_df['Cabin'].fillna('N',inplace=True)
titanic_df['Embarked'].fillna('N',inplace=True)
print('데이터 세트 Null 값 갯수 ',titanic_df.isnull().sum().sum())


# In[103]:


print(' Sex 값 분포 :\n',titanic_df['Sex'].value_counts())
print('\n Cabin 값 분포 :\n',titanic_df['Cabin'].value_counts())
print('\n Embarked 값 분포 :\n',titanic_df['Embarked'].value_counts())


# In[104]:


titanic_df['Cabin'] = titanic_df['Cabin'].str[:1]
print(titanic_df['Cabin'].head(3))


# In[105]:


titanic_df.groupby(['Sex','Survived'])['Survived'].count()


# In[106]:


sns.barplot(x='Sex', y = 'Survived', data=titanic_df)


# In[107]:


sns.barplot(x='Pclass', y='Survived', hue='Sex', data=titanic_df)


# In[108]:


# 입력 age에 따라 구분값을 반환하는 함수 설정. DataFrame의 apply lambda식에 사용. 
def get_category(age):
    cat = ''
    if age <= -1: cat = 'Unknown'
    elif age <= 5: cat = 'Baby'
    elif age <= 12: cat = 'Child'
    elif age <= 18: cat = 'Teenager'
    elif age <= 25: cat = 'Student'
    elif age <= 35: cat = 'Young Adult'
    elif age <= 60: cat = 'Adult'
    else : cat = 'Elderly'
    
    return cat

# 막대그래프의 크기 figure를 더 크게 설정 
plt.figure(figsize=(10,6))

#X축의 값을 순차적으로 표시하기 위한 설정 
group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Elderly']

# lambda 식에 위에서 생성한 get_category( ) 함수를 반환값으로 지정. 
# get_category(X)는 입력값으로 'Age' 컬럼값을 받아서 해당하는 cat 반환
titanic_df['Age_cat'] = titanic_df['Age'].apply(lambda x : get_category(x))
sns.barplot(x='Age_cat', y = 'Survived', hue='Sex', data=titanic_df, order=group_names)
titanic_df.drop('Age_cat', axis=1, inplace=True)


# In[109]:


from sklearn import preprocessing

def encode_features(dataDF):
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(dataDF[feature])
        dataDF[feature] = le.transform(dataDF[feature])
        
    return dataDF

titanic_df = encode_features(titanic_df)
titanic_df.head()


# In[110]:


from sklearn.preprocessing import LabelEncoder

# Null 처리 함수
def fillna(df):
    df['Age'].fillna(df['Age'].mean(),inplace=True)
    df['Cabin'].fillna('N',inplace=True)
    df['Embarked'].fillna('N',inplace=True)
    df['Fare'].fillna(0,inplace=True)
    return df

# 머신러닝 알고리즘에 불필요한 속성 제거
def drop_features(df):
    df.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)
    return df

# 레이블 인코딩 수행. 
def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin','Sex','Embarked']
    for feature in features:
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df

# 앞에서 설정한 Data Preprocessing 함수 호출
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df


# In[111]:


# 원본 데이터를 재로딩 하고, feature데이터 셋과 Label 데이터 셋 추출. 
titanic_df = pd.read_csv('./titanic_train.csv')
y_titanic_df = titanic_df['Survived']
X_titanic_df= titanic_df.drop('Survived',axis=1)

X_titanic_df = transform_features(X_titanic_df)


# In[112]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X_titanic_df, y_titanic_df,                                                   test_size=0.2, random_state=11)


# In[113]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 결정트리, Random Forest, 로지스틱 회귀를 위한 사이킷런 Classifier 클래스 생성
dt_clf = DecisionTreeClassifier(random_state=11)
rf_clf = RandomForestClassifier(random_state=11)
lr_clf = LogisticRegression()
svm_svc = SVC() ##추가해보자.!!

# DecisionTreeClassifier 학습/예측/평가
dt_clf.fit(X_train , y_train)
dt_pred = dt_clf.predict(X_test)
print('DecisionTreeClassifier 정확도: {0:.4f}'.format(accuracy_score(y_test, dt_pred)))

# RandomForestClassifier 학습/예측/평가
rf_clf.fit(X_train , y_train)
rf_pred = rf_clf.predict(X_test)
print('RandomForestClassifier 정확도:{0:.4f}'.format(accuracy_score(y_test, rf_pred)))

# LogisticRegression 학습/예측/평가
lr_clf.fit(X_train , y_train)
lr_pred = lr_clf.predict(X_test)
print('LogisticRegression 정확도: {0:.4f}'.format(accuracy_score(y_test, lr_pred)))


# SVM 학습/예측/평가
svm_svc.fit(X_train , y_train)
svm_pred = lr_clf.predict(X_test)
print('SVM 정확도: {0:.4f}'.format(accuracy_score(y_test, lr_pred)))


# In[114]:


from sklearn.model_selection import KFold


# In[115]:


kfold = KFold(n_splits=5)


# In[116]:


for iter_count, (train_index, test_index) in enumerate(kfold.split(X_titanic_df)):
    print(iter_count)


# In[117]:


X_titanic_df.shape


# In[118]:


for iter_count, (train_index, test_index) in enumerate(kfold.split(X_titanic_df)):
    print(iter_count)
    print('--------------')
    print(train_index)


# In[119]:


for iter_count, (train_index, test_index) in enumerate(kfold.split(X_titanic_df)):
    print(iter_count)
    print('--------------')
    print(test_index)


# In[120]:


X_titanic_df.values[train_index]


# In[121]:


from sklearn.model_selection import KFold

def exec_kfold(clf, folds=5):
    # 폴드 세트를 5개인 KFold객체를 생성, 폴드 수만큼 예측결과 저장을 위한  리스트 객체 생성.
    kfold = KFold(n_splits=folds)
    scores = []
    
    # KFold 교차 검증 수행. 
    for iter_count , (train_index, test_index) in enumerate(kfold.split(X_titanic_df)):
        # X_titanic_df 데이터에서 교차 검증별로 학습과 검증 데이터를 가리키는 index 생성
        X_train, X_test = X_titanic_df.values[train_index], X_titanic_df.values[test_index]
        y_train, y_test = y_titanic_df.values[train_index], y_titanic_df.values[test_index]
        
        # Classifier 학습, 예측, 정확도 계산 
        clf.fit(X_train, y_train) 
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        scores.append(accuracy)
        print("교차 검증 {0} 정확도: {1:.4f}".format(iter_count, accuracy))     
    
    # 5개 fold에서의 평균 정확도 계산. 
    mean_score = np.mean(scores)
    print("평균 정확도: {0:.4f}".format(mean_score)) 
    return mean_score
# exec_kfold 호출


# In[122]:


all_result = []


# In[123]:


len(all_result)


# In[124]:


dt_result = exec_kfold(dt_clf , folds=5) 
dt_result
all_result.append(dt_result)


# In[125]:


svm_result10 = exec_kfold(svm_svc , folds=10) 
svm_result10
all_result.append(svm_result10)


# In[126]:


rf_result = exec_kfold(rf_clf , folds=10) 
rf_result
all_result.append(rf_result)


# In[127]:


len(all_result)


# In[128]:


def all_max(all_list):
    print(all_list)
    max_value = 0
    for model_result in all_list:
        if(model_result > max_value):
            max_value = model_result
        print('현재까지의 max값 >> ' , max_value)
    return max_value


# In[129]:


type(all_result[0])


# In[130]:


max_result = all_max(all_result)
print(max_result)


# In[131]:


def all_max2(all_list):
    max_value = 0
    index = 0
    for i, model_result in enumerate(all_list):
        if(model_result > max_value):
            max_value = model_result
            index = i
        print('현재까지의 max값 >> ' , max_value)
    return max_value, index


# In[132]:


max_result2 = all_max2(all_result)
print(max_result2)


# In[134]:


model_list = ['decision tree', 'svm10', 'random forest']


# In[136]:


# print('최적의 model은 ', model_list[2])
print('최적의 model은 ', model_list[max_result2[1]])


# In[ ]:


# k-fold를 한번으로 묶어보세요.
final_result = callKFold_all()
print("-----")
=================
최적의 model은  random forest, 최대 accuracy는 0.8111


# In[ ]:


final_result2 = callKFold_all2()

cross_val_score()함수 호출!!

반복문 이용!!하여 모든 모델에 대해 수행
=================
최적의 model은  random forest, 최대 accuracy는 0.8111

