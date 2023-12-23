# Electricity-usage-forecast

2023년 8월 중반 부터 말까지 참여했던 한국 에너지 공단이 주최하고 Dacon이 주관한 '2023 전력사용량 예측 AI 경진대회' 공모전 코드 파일 정리 및 기록입니다. 

## 대회 개요 및 설명

### 배경
안정적이고 효율적인 에너지 공급을 위해서는 전력 사용량에 대한 정확한 예측이 필요합니다.
따라서 한국에너지공단에서는 전력 사용량 예측 시뮬레이션을 통한 효율적인 인공지능 알고리즘 발굴을 목표로 본 대회를 개최합니다.

### 설명
건물 정보와 시공간 정보를 활용하여 특정 시점의 전력 사용량을 예측하는 AI 모델을 개발해주세요.

### 데이터
* train.csv - 100개 건물들의 2022년 06월 01일부터 2022년 08월 24일까지의 데이터
  일시별 기온, 강수량, 풍속, 습도, 일조, 일사 정보
  전력사용량(kWh) 포함
* building_info.csv - 100개 건물 정보
  건물 번호, 건물 유형, 연면적, 냉방 면적, 태양광 용량, ESS 저장 용량, PCS 용량
* test.csv - 100개 건물들의 2022년 08월 25일부터 2022년 08월 31일까지의 데이터
  일시별 기온, 강수량, 풍속, 습도의 예보 정보
* submission.csv - 제출을 위한 양식, 100개 건물들의 2022년 08월 25일부터 2022년 08월 31일까지의 전력사용량(kWh)을 예측

## EDA, Exploartory Data Analysis

### 이상치 처리
건물별 전력 사용량을 seaborn의 heatmap을 사용해서 행은 24hr로 열은 7일로 가지고 있는 그래프를 그려 이상치를 찾아 시계열 데이터의 특성을 고려해 pandas의 보간법으로 값을 변경함. 아래 그래프에서 이상치 확인 가능.(34번 건물은 월요일 21h경에 이상치 발생)
![hm](https://github.com/Junoflows/Electricity-usage-forecast/assets/79469037/5851e3f0-9986-48f9-9344-372540ee7b6e)

### 건물 유형 별, 평균 전력량 사용별 clustering
요일 / 시간 별 전력 중앙 값에 대해 scaling하고 건물 유형별, 평균 전력량 사용별 clustering을 진행해 line graph를 그렸을 때 건물 유형 기타를 제외하고는 대체로 비슷한 양상을 보였음. 그러나 보다 정확한 예측을 위해 각 건물 별로 학습을 따로 진행함.

### 태양열
건물 정보 csv에서 태양광 발전판을 소유하고 있는 회사들에 대해 전일 일조량 평균 값과 당일 전력 사용량 평균값 사이의 관계를 x축 일조량 값, y축을 전력 사용량 그래프로 시각화함. 그래프를 보면 태양광 에너지 저축량이 올라감에 따라 전력 사용량이 줄어드는 반비례 관계에 있는 건물들을 확인할 수 있음. 
![solar](https://github.com/Junoflows/Electricity-usage-forecast/assets/79469037/31e814b6-84d5-4973-a8e0-a01fc04e8c55)

### ESS+ PCS capacity
태양열과 동일한 방법으로 실시했으나 반비례 관계에 있는 건물이 존재하지 않아 사용하지 않음.



## 데이터 전처리


### 변수 추가
+ 시계열 데이터 시간 관련 변수 추가
  월, 일, 시간, 요일, 공휴일
```
eda_df['datetime'] = pd.to_datetime(eda_df['datetime'])
eda_df['hour'] = eda_df['datetime'].dt.hour
eda_df['weekday'] = eda_df['datetime'].dt.weekday #요일
eda_df['date'] = eda_df['datetime'].dt.date
eda_df['day'] = eda_df['datetime'].dt.day
eda_df['month'] = eda_df['datetime'].dt.month
eda_df['weekend'] = eda_df['weekday'].isin([5,6]).astype(int)
```
+ 건물별 요일별 시간별 반전량 평균 열 추가
```
power_mean = pd.pivot_table(train, values = 'power', index = ['num', 'hour', 'day'], aggfunc = np.mean).reset_index()
train['day_hour_mean'] = train.progress_apply(lambda x : power_mean.loc[(power_mean.num == x['num']) & (power_mean.hour == x['hour']) & (power_mean.day == x['day']) ,'power'].values[0], axis = 1)
## 건물별 요일별 시간별 발전량 표준편차 넣어주기
power_hour_std = pd.pivot_table(train, values = 'power', index = ['num', 'hour', 'day'], aggfunc = np.std).reset_index()
train['day_hour_std'] = train.progress_apply(lambda x : power_hour_std.loc[(power_hour_std.num == x['num']) & (power_hour_std.hour == x['hour']) & (power_hour_std.day == x['day']) ,'power'].values[0], axis = 1)
train.head()
```


+ 전력 사용량은 기온의 변화에 따라 즉각적으로 변하지 않음
  기온 대신 냉방도일의 개념을 추가하여 변수 생성
```
def CDH(xs):
    ys = []
    for i in range(len(xs)):
        if i < 11:
            ys.append(np.sum(xs[:(i+1)]-24))
        else:
            ys.append(np.sum(xs[(i-11):(i+1)]-24))
    return np.array(ys)

cdhs = np.array([])
for num in range(1,101,1):
    temp = train[train['num'] == num]
    cdh = CDH(temp['temp'].values)
    cdhs = np.concatenate([cdhs, cdh])
train['CDH'] = cdhs
```

### 결측치 및 이상치 처리
+ 일조, 일사는 test 데이터에는 없으므로 제거
+ 강수량 결측치는 0으로 처리
+ 풍속, 습도 결측치는 보간법으로 대체
```
eda_df.precipitation.fillna(0.0, inplace = True)
eda_df.windspeed = eda_df.windspeed.interpolate(method = 'polynomial', order = 3)
eda_df.humidity = eda_df.humidity.interpolate(method = 'polynomial', order = 3)
```
+ 이상치 처리
  건물별 그래프 확인하여 34번 - 월요일, 56번 건물 - 수요일 이상치 확인
  NUll 값으로 대체한 후 pandas의 interpolation(보간법) 활용하여 이상치 채움

### test 셋에도 같은 과정 적용으로 변수 통일


## autoML - pycaret 사용


### pycaret이란 
ML workflow을 자동화 하는 opensource library로 여러 머신러닝 task에서 사용하는 모델들을 하나의 환경에서 비교하고 튜닝하는 등 간단한 코드를 통해 편리하게 사용할 수 있도록 자동화한 라이브러리.


### 간단한 코드
pycaret은 돌리는데 시간이 오래 걸리기 때문에 가장 결과값이 좋지 않았던 40, 42, 54번 건물을 기준으로 가장 좋은 결과값을 리턴하는 모델을 찾아봄.
공모전 평가 지표인 smape함수를 만들어 custom metric을 추가함.
```
add_metric('smape', 'SMAPE', SMAPE, greater_is_better = False)

test = [40,42,54]
best_models = []

for i in [test]:
    y = train.loc[train.num == i, 'power']
    x = train.loc[train.num == i, features]
    # 마지막 일주일 발전량을 validset으로 24시간*7일 = 168
    y_train, y_test, x_train, x_test = temporal_train_test_split(y = y, X = x, test_size = 168)

    #pycaret 사용 - df 병합
    traindata = x_train.join(y_train)
    testdata = x_test.join(y_test)

    exp = setup(traindata, target= 'power')

    add_metric('smape', 'SMAPE', SMAPE, greater_is_better = False)
    get_metrics()

    best = compare_models(sort = 'MAE')
    model = create_model(best)
    tuned_model = tune_model(model, optimize = 'MAE', n_iter = 100,  choose_better = True)
    best_models.append(tuned_model)
    pm = predict_model(tuned_model, data = testdata)
    building = 'building'+str(i)
    print(building + '|| SMAPE : {}'.format(SMAPE(pm['power'], pm['prediction_label'])))
```

### 결과
가장 결과 값이 좋은 상위 5개의 모델 중 비슷한 성격을 가진 Extra Trees Regressor와 Random Forest Regressor 중 가장 좋은 결과를 보여준 ET를 선택한 대신 사용하지 않은 RF Regressor를 제외하고 나머지 모든 모델을 사용해서 Regression 실시.
![pycaret](https://github.com/Junoflows/Electricity-usage-forecast/assets/79469037/bca5f255-2a07-43df-b023-5a17e1895c87)


## 모델링

### :warning: 주의 사항 
Xgboost, LGBM은 earlystopping_rounds 사용을 위해서 낮은 version install해서 사용함.
* Xgboost = 1.2.1
* LGBM = 3.3.5

### xgboost, lgbm, catboost, extratree
+ 트리 기반 모델로 회귀와 분류에 모두 사용됨
+ 트리 기반 모델이라 변수 scale 의 영향을 덜 받아 따로 scaling 을 안해줘도 됨
https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/catboost-hyperparameters.html  
파라미터 설정 참고한 링크

```
grid = {'iterations': [500],
        'learning_rate' : [0.01, 0.03, 0.05, 0.07],
        'depth': [4,6,8],
        'l2_leaf_reg': [2,4,6,8],
        'subsample': [0.8],
        'colsample_bylevel': [0.8],
        'random_strength' : [1,3,5,7]
         }
```
gridsearchCV 를 이용해 최적의 파라미터 설정

### 손실함수
$MAE = \frac{1}{n}\Sigma_{i=1}^n| \hat{Y_i}-Y_i|$ <br/>
$RSME = (\frac{1}{n}\Sigma_{i=1}^n (\hat{Y_i}-Y_i)^2)^{0.5}$  

과소적합보다 과대 적합이 더 좋은 평가지표인 SMAPE 이므로 RMSE 보다 MAE 가 더 좋은 손실함수로 판단

### Cross Validation
* cv fold 사용
* pds - 마지막 일주일을 validation set으로 설정
```
# 마지막 일주일 발전량을 validset으로 24시간*7일 = 168
pds = PredefinedSplit(np.append(-np.ones(len(x)-168), np.zeros(168)))
```
  
* Blocked timeseries CV에서 mode를 추가해 custom cv를 만듦
  mode1. n주씩 겹치면서 훈련 + 마지막 일주일을 validation set으로 사용
  mode2. 가장 마지막 일주일을 고정 validation set으로 설정 + 훈련 데이터를 점차 줄임



[mode1 그림]
![m2](https://github.com/Junoflows/Electricity-usage-forecast/assets/79469037/4c187841-18f3-409b-b615-ea3a586e01b1)

```
class BlockingTimeSeriesSplit():
    def __init__(self, n_splits = 0, mode = 1):
        self.n_splits = n_splits
        self.mode = mode

    def get_n_splits(self, X, y, groups):
        return self.n_splits

    def split(self, X, y=None, groups=None):
      if self.mode == 0:
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]
      elif self.mode == 1:
        n_samples = len(X)
        indices = np.arange(n_samples)

        weeks = X.week.unique()
        current_week = weeks[0]
        end_week = weeks[-1]

        n_range = end_week - self.n_splits - current_week +2

        while current_week + n_range - 1 <= end_week:
          start = X[X.week == current_week].index[0]
          mid = X[X.week == current_week + n_range - 1].index[0]
          stop = X[X.week == current_week + n_range - 1].index[-1]
          current_week += 1
          yield indices[start:mid], indices[mid:stop+1]
      elif self.mode == 2:
        n_samples = len(X)
        indices = np.arange(n_samples)

        weeks = X.week.unique()
        current_week = weeks[0]
        end_week = weeks[-1]

        n_range = end_week - self.n_splits - current_week +2

        mid = X[X.week == end_week].index[0]
        stop = X[X.week == end_week].index[-1]
        while current_week <= end_week - n_range + 1:
          start = X[X.week == current_week].index[0]
          current_week += 1
          yield indices[start:mid], indices[mid:stop+1]
      elif self.mode == 3:
        n_samples = len(X)
        indices = np.arange(n_samples)

        weeks = X.week.unique()
        start_week = weeks[0]
        end_week = weeks[-1]

        start = X[X.week == start_week].index[0]
        mid = X[X.week == end_week].index[0]
        stop = X[X.week == end_week].index[-1]

        yield indices[start:mid], indices[mid:stop+1]

```
실제 사용은 가장 좋은 성능을 보여준 pds 사용.




## 결과 및 리뷰
+ xgboost 와 lgbm 으로 시작하였고 제일 결과가 좋은 모델은 catboost 였음
+ Pycaret 으로 가장 효과가 좋은 모델을 찾았을 때도 catboost, xgboost, lgbm 순임
+ 데이터 전처리를 마치고 바로 모델을 돌려보는 것보다는 Pycaret 을 돌려보고 성능 좋은 모델을 선별하여 모델링하는게 좋을 것으로 판단됨.
+ 건물별로 특성을 더 잘 파악했다면 어땠을까 하는 아쉬움이 남음
  + 건물별로 주말 외의 쉬는 날이 있다는 걸 고려했다면 더 좋은 성능을 띄었을 것으로 판단됨
