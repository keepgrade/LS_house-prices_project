import numpy as np
import pandas as pd

# 집 가격 데이터셋 불러오기.
data_path = r"C:\Users\USER\Desktop\LS_BigData_School\zip_data\house-prices\train.csv"
df = pd.read_csv(data_path)
df.head()
df.info()
df.describe()

# House Price 데이터셋을 가장 잘 설명하는 
# 선형 회귀 모델을 구축하기 

# 1. 데이터 전처리 

# 기본 정보 확인
print(df.shape)
print(df.isnull().sum().sort_values(ascending=False).head(20))

# 결측치 비율이 높은 열 제거 (예: 30% 이상)
null_rate = df.isnull().mean()
df = df.loc[:, null_rate < 0.3]

# 수치형 변수 결측치 → 평균으로 대체
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# 범주형 변수 결측치 → 최빈값으로 대체
cat_cols = df.select_dtypes(include='object').columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# 결측치 제거

# 이상치 제거

# 2.모델 구축

# 

# 3. 모델 평가 

#

# 4. 모델 개선 

# 

# 5. 모델 결과 

# 