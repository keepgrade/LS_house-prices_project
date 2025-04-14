import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# 한글 폰트 설정 (Windows의 경우 예: 맑은 고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'  # 또는 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지
# 집 가격 데이터셋 불러오기.
data_path = r"C:\Users\USER\Desktop\LS_BigData_School\zip_data\house-prices\train.csv"
df = pd.read_csv(data_path)

# 데이터셋 확인
print(df.head())
print(df.info())
print(df.describe())

df["KitchenQual"]

# 사용할 변수 설정 

# 1. 도메인 지식 기반 선택
# 장점: 사람이 직관적으로 이해하기 좋고, 해석이 명확함
# 단점: 중요한 변수를 놓칠 수도 있고, 주관적임

# 2. 상관관계 기반 선택 (수학적)
# 장점: 실제로 집값과 수치적으로 얼마나 관련이 있는지 객관적으로 확인 가능
# 단점: 인과관계를 설명할 수는 없음. 다중공선성 문제 가능성 존재

# 데이터 불러오기
data_path = r"C:\Users\USER\Desktop\LS_BigData_School\zip_data\house-prices\train.csv"
df = pd.read_csv(data_path)

# SalePrice 결측치 있는 행 제거
df = df.dropna(subset=['SalePrice'])

import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrices

# 결과 저장용 리스트
results = []

# 단일 회귀분석을 통한 변수 선택
for col in df.columns:
    if col == 'SalePrice':
        continue

    # 고유값이 하나뿐인 변수는 제외
    if df[col].nunique() <= 1:
        continue

    # 해당 변수의 결측치 제거
    data_subset = df[['SalePrice', col]].dropna()

    # 데이터가 너무 적으면 제외
    if data_subset.shape[0] < 20:
        continue

    # 변수명이 안전하지 않은 경우 Q("...")로 감싸기
    if df[col].dtype == 'object':
        formula = f'SalePrice ~ C(Q("{col}"))'  # 범주형
        # (※ 한 카테고리는 기준으로 빠져 있고, 나머지는 0/1로 코딩됨)
    else:
        formula = f'SalePrice ~ Q("{col}")'     # 수치형

    try:
        model = smf.ols(formula, data=data_subset).fit()

        results.append({
            '변수명': col,
            'R² (결정계수)': round(model.rsquared, 4),
            'p-value': round(model.f_pvalue, 5)
        })

    except Exception as e:
        print(f"[상관 분석 불가] {col}: {e}")

# 결과 정리 및 출력
result_df = pd.DataFrame(results).sort_values(by='R² (결정계수)', ascending=False)

if result_df.empty:
    print("유효한 회귀 결과가 없습니다. 조건을 완화해보세요.")
else:
    print("단변수 회귀분석 결과 (R² 기준 정렬):")
    print(result_df.reset_index(drop=True))

# 변수 유형	문제점
# 비선형 수치형 변수	단순 선형으로만 분석되어 실제 영향력을 저평가
# 범주형 변수	C(Q("col")) 처리로 단순 회귀는 되지만, 카테고리가 많거나 희귀하면 해석이 어렵고 과적합 가능
# 변수 간 상호작용	완전히 배제됨 (조합 효과 고려 안 됨)

# 지금은 단순 회귀분석을 활용해
# ➤ SalePrice에 가장 영향을 많이 주는 핵심 변수들만 골라내는 것에 집중

# 비선형 처리나 범주형 변수 정제는
# 추가로 배워야 한다.


# 상관계수 시각화 
import matplotlib.pyplot as plt
import seaborn as sns

# 수치형 변수만 추출
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# 상관계수 계산
corr_matrix = numeric_df.corr()

# SalePrice와의 상관계수 상위 10개 변수 추출
top_corr = corr_matrix['SalePrice'].abs().sort_values(ascending=False).drop('SalePrice').head(10)

# 막대그래프로 시각화
plt.figure(figsize=(8, 6))
sns.barplot(x=top_corr.values, y=top_corr.index, palette='coolwarm')
plt.title("SalePrice와 상관계수 높은 상위 10개 변수")
plt.xlabel("상관계수 (절댓값 기준)")
plt.ylabel("변수명")
plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.show()


# 수치형 변수만 선택
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# 상관계수 계산
corr_matrix = numeric_df.corr()

# SalePrice와의 상관계수 기준 정렬
top3_corr = corr_matrix['SalePrice'].abs().sort_values(ascending=False).drop('SalePrice').head(3)

print("SalePrice와 상관계수가 높은 수치형 변수 TOP 3:")
print(top3_corr)


# 플롯 생성
plt.figure(figsize=(18, 5))

for i, col in enumerate(top3_corr.index): 
    plt.subplot(1, 3, i+1)
    sns.regplot(x=col, y='SalePrice', data=df, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.title(f"[{col}] vs SalePrice")
    plt.xlabel(col)
    plt.ylabel("SalePrice")

plt.suptitle("상관계수 Top 3 변수와 SalePrice의 단순 회귀분석", fontsize=14)
plt.tight_layout()
plt.show()

# 집값과 가장 관련 있는 Top 3 변수 요약
# 1️⃣ OverallQual – 전반적인 품질 등급
# 정의: 자재 및 마감 품질 종합 점수 (1~10점)
# 상관계수: 0.79 → 집값과 가장 밀접
# 해석: 고급 자재일수록 집값이 확실히 높아짐

# 2️⃣ GrLivArea – 지상 생활 면적
# 정의: 지상층(1층+2층) 총 면적 (ft² 기준)
# 상관계수: 0.71
# 해석: 면적이 넓을수록 집값도 상승, 이상치 유의

# 3️⃣ GarageCars – 차고 수용 차량 수
# 정의: 차고에 주차 가능한 차량 대수 (0~4)
# 상관계수: 0.64
# 해석: 2대 이상 가능 시 고급 주택 가능성↑, 불연속 변수 특성 있음


top3_vars = top3_corr.index.tolist()  # ['OverallQual', 'GrLivArea', 'GarageCars']


import statsmodels.api as sm

# X (독립변수), y (종속변수) 구성
X = df[top3_vars].dropna()
y = df.loc[X.index, 'SalePrice']

# 상수항 추가
X = sm.add_constant(X)

# 회귀 모델 학습
model = sm.OLS(y, X).fit()

# 결과 출력
print(model.summary())



# 예측값
y_pred = model.predict(X)
# 계수 가져오기
intercept = model.params['const']
b1 = model.params['OverallQual']
b2 = model.params['GrLivArea']
b3 = model.params['GarageCars']

# 수식 문자열로 정리
equation = f"SalePrice = {intercept:,.0f} + {b1:,.0f}×OverallQual + {b2:.2f}×GrLivArea + {b3:,.0f}×GarageCars"

import numpy as np
from sklearn.metrics import mean_squared_error

# 예측값
y_pred = model.predict(X)

# 지표 계산
rmse = np.sqrt(mean_squared_error(y, y_pred))
adj_r_squared = model.rsquared_adj
f_statistic = model.fvalue

# 시각화
plt.figure(figsize=(7, 7))
plt.scatter(y, y_pred, alpha=0.3, color='dodgerblue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("실제 SalePrice")
plt.ylabel("예측 SalePrice")
plt.title("다중 회귀모델 예측값 vs 실제값")

# 회귀 수식 ,평가 지표 3종 텍스트로 출력
plt.text(x=y.min(), y=y.max() * 0.95,
         s=f"{equation}\nAdj. R² = {adj_r_squared:.3f}  |  F = {f_statistic:.1f}  |  RMSE = {rmse:,.0f}",
         fontsize=10, color='black',
         bbox=dict(facecolor='white', edgecolor='gray'))
plt.grid(True)
plt.tight_layout()
plt.show()


# 🔍 잔차 계산
residuals = y - y_pred

# 🔸 1. 잔차 vs 예측값 그래프
plt.figure(figsize=(6, 5))
plt.scatter(y_pred, residuals, alpha=0.3, color='darkorange')
plt.axhline(0, color='red', linestyle='--')
plt.title("잔차 vs 예측값")
plt.xlabel("예측값 (Predicted SalePrice)")
plt.ylabel("잔차 (Residuals)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 🔸 2. 잔차 정규성 확인 (QQ Plot)
import statsmodels.api as sm
sm.qqplot(residuals, line='s')
plt.title("잔차의 Q-Q Plot (정규성 확인)")
plt.tight_layout()
plt.show()


# RMSE → 예측 오차의 크기 예측이 실제값과 가깝다 → 정확도 높음 👍