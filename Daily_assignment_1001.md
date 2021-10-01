교육 제목:  파이썬 머신러닝 판다스 데이터 분석

교육 일시:  21. 10. 01

교육장소: 영우글로벌러닝

# 교육내용

## 데이터 사전 처리

### 1. 누락 데이터 처리

NaN 값을 처리할때 당연히 없는게 제일 좋지만 그러다간 데이터의 수가 줄어들고, 해당 값 외에는 쓸만한데도 버려질 가능성이 높다. 이럴때는 fillna()를 활용하여 치환을 하고, 제거해도 될 때는 제거를 한다. 이를 위해서는 데이터를 확인하는 작업이 선행되어야만 한다.

 1. #### 누락 데이터 확인

    `df['col'].value_counts(dropna=False)`: NaN값을 떨구지 않으면서 도수분포를 col에 대해 알아보라는 것으로 각 요소의 도수가 출력된다. 여기서 dropna=True를 하면 NaN값이 몇갠지 알 수가 없다

    `isnull()과notnull()`: df에 대해서 적용되는 메소드로 각 요소의 NaN 여부를 판별한다. df[bull]이면 해당 요소를 반환하기 때문에 이런 불리언 값을 반환하는 메소드는 필터링처럼 사용할 수 있어 유용하다. 그 바로 다음 에제가 이렇다

    `print(df.head().isnull().sum(axis=0))`: 데이터프레임의 앞 데이터만 가져와서 NaN인지 판별한 뒤에, 행을 모두 더한다. 여기서 True=1, False=0이므로 각 열에 따라 NaN의 개수를 반환한다

 2. #### 제거

    isnull()이 참값을 가지는걸 이용해서 반복문을 활용하여 각 열에 따른 NaN값을 출력할 수 있으며,

     ```python
        missing_df = df.isnull()
    for col in missing_df.columns:
        missing_count = missing_df[col].value_counts()    # 각 열의 NaN 개수 파악
        try: 
            print(col, ': ', missing_count[True])   # NaN 값이 있으면 개수를 출력
        except:
            print(col, ': ', 0)                     # NaN 값이 없으면 0개 출력
     ```

    `df.dropna(axis=1,thresh=500)`과 같은 방법으로 threshold를 설정하고 해당 값 이상을 버릴 수 있다.

    `df_age = df.dropna(subset=['age'], how='any', axis=0)  `처럼 dropna는 subset을 선택하여 na를 솎아낼 수 있다. how=any는 하나라도, how=all은 모두 NaN값일 때 삭제된다.

 3. #### 치환

    fillna()를 활용하는데, front 값을 이용하려면 f를 붙인 ffillna(), back 값을 이용하려면 bfillna()를 사용한다 평균값이나 중앙값을 활용하기도 하는데, 이는 데이터셋을 이쁘게 만들어줄수는 있지만 분석에 영향을 미칠 여지가 크므로 유의해서 사용해야 한다.

### 2. 중복 데이터 처리

1. #### 확인

   `duplicated()` 메소드는 선(pre) 행과 비교하여 중복되는 행이면 True 값을 반환한다.첫행은 이전 행이 존재하지 않으므로 무조건 False를 반환. 여러 열이 있는 데이터프레임에도 적용 가능하고 특정 열을 지정하여 사용할수도 있다.

2. #### 제거

   `drop_duplicates()` 메소드는 중복행을 제거하고 고유한 관측값만을 남긴다. 중복되는 데이터가 의미가 없을때만 사용한다. 만약에 중복되는 값이 많이 나올만한 데이터라면 무지성으로 지워서는 안된다. 열을 지정하여 부분집합에 대해서도 사용 가능하다. 예컨대

   `df3=df.drop_duplicates(subset=['c2','c3'])` 이런 식으로 사용 가능하다

   

### 3. 데이터 표준화

1. #### 단위 환산

   미국 표준 단위계에서는 SI단위계와 다른 단위를 많이 쓰는데, 이를 예시로 각 열에 곱하기 연산을 하는 연습을 할 수 있다.

2. #### 자료형 변환

   자료형은 우리가 자주 썼던 float int str 등이 아니라 object로 뜨는 경우가 있는데, 이럴때 적절한 형을 찾아서 형변환을 해주면 다루기 편해진다. 이때 년도처럼 숫자를 쓰는 열이 존재할 수 있는데, 이건 숫자로 다루는 연도의 개념보다는 숫자의 상대적인 크기 자체는 별 의미가 없다. 따라서 범주형으로 바꿔주는게 좋다. 고유값을 바꾸는데는 `astype()` 을 활용한다.

   가장 특기할 만한 점은 NaN이 아닌 문자열이 들어간 데이터를 NaN으로 반환하는 기법인데, numpy에서 제공하는 기능을 활용한다. 이 단원에서 가장 중요하고 앞으로도 많이 쓰일만한 문장

   ``` import numpy as np
   import numpy as np
   df['horsepower'].replace('?', np.nan, inplace=True)      # '?'을 np.nan으로 변경
   df.dropna(subset=['horsepower'], axis=0, inplace=True)   # 누락데이터 행을 삭제
   df['horsepower'] = df['horsepower'].astype('float')      # 문자열을 실수형으로 변환
   ```

### 4. 범주형(카테고리) 데이터 처리

1. #### 구간 분할

   구간을 나눠 성질의 상중하를 나눌수 있을때 자주 사용한다. 예컨대 차종의 마력에 따라 상중하로 나눌수있고, 사람의 연령에 따라 어린이 청년 중년 장년으로 구분가능하다. 이렇게 데이터프레임의 구간(bin)을 나누는 장점은 실수값보다 이산적인 값으로 분석해서 데이터의 양이 줄어들고, 범주에 따라 파악하기가 좋아진다. 앞에서 배운 누락된 문자를 NaN으로 변환하는 데이터 전처리 과정은 선행되어야만 한다

   ``` 
   import pandas as pd
   import numpy as np
   
   # read_csv() 함수로 df 생성
   df = pd.read_csv('./auto-mpg.csv', header=None)
   
   # 열 이름을 지정
   df.columns = ['mpg','cylinders','displacement','horsepower','weight',
                 'acceleration','model year','origin','name'] 
   
   # horsepower 열의 누락 데이터('?') 삭제하고 실수형으로 변환
   df['horsepower'].replace('?', np.nan, inplace=True)      # '?'을 np.nan으로 변경
   df.dropna(subset=['horsepower'], axis=0, inplace=True)   # 누락데이터 행을 삭제
   df['horsepower'] = df['horsepower'].astype('float')      # 문자열을 실수형으로 변환
   
   # np.histogram 함수로 3개의 bin으로 나누는 경계 값의 리스트 구하기
   count, bin_dividers = np.histogram(df['horsepower'], bins=3)
   print(bin_dividers) 
   
   # 3개의 bin에 이름 지정
   bin_names = ['저출력', '보통출력', '고출력']
   
   # pd.cut 함수로 각 데이터를 3개의 bin에 할당
   df['hp_bin'] = pd.cut(x=df['horsepower'],     # 데이터 배열
                         bins=bin_dividers,      # 경계 값 리스트
                         labels=bin_names,       # bin 이름
                         include_lowest=True)    # 첫 경계값 포함 
   
   # horsepower 열, hp_bin 열의 첫 15행을 출력
   print(df[['horsepower', 'hp_bin']].head(15))
   ```

   

2. #### 더미 변수

   dummy는 의미없는 것을 뜻하는데, 수학에서 표기한 것의 local variable처럼 쓰일때 dummy variable이라고 부른다. 여기서도 마찬가지로 새로운 변수가 생긴것이 아니고 그저 우리가 보는 형태에서 컴퓨터가 확인할수있도록 0과 1로 매핑할때, 이 숫자들을 dummy variable이라고 한다. 그리고 이런 0이나 1로만 변환된다 해서 원핫벡터라고 부르며, 이런 변환과정은 원핫인코딩이라고 부른다. 또한 이런 dummy variable을 벡터로 표현할수 있는데, 범주를 나눈 특성상 n개 범주를 나누었다면 n-1은 0이기 때문에, 희소행렬(sparse matrix)이며 희소행렬은 (행,열)좌표와 값 형태로 정리된다. 이때 그 좌표가 원핫벡터다.

3. #### 정규화

   Normalization. 데이터셋의 크기로 나누어 비율로서 표현한 것. -1 to 1, 0 to 1 이런 범위 내로 지정한다

   ```
   import pandas as pd
   import numpy as np
   
   df = pd.read_csv('./auto-mpg.csv', header=None)
   
   df.columns = ['mpg','cylinders','displacement','horsepower','weight',
                 'acceleration','model year','origin','name']  
   
   df['horsepower'].replace('?', np.nan, inplace=True)      
   df.dropna(subset=['horsepower'], axis=0, inplace=True)   
   df['horsepower'] = df['horsepower'].astype('float')      
   
   print(df.horsepower.describe())
   print('\n')
   
   df.horsepower = df.horsepower / abs(df.horsepower.max()) 
   
   print(df.horsepower.head())
   print('\n')
   print(df.horsepower.describe())
   ```

   



## Numpy

numpy는 n dimensional matrix를 구성하기에 좋은 툴이다.

예시로, 1차원 0부터 23까지 정수를 원소로 갖는 어레이를 만들어, 이걸로 3차원 matrix를 만들어보자

```
import numpy as np
print(np.arange(24),type(np.arange(24)))
# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23] <class 'numpy.ndarray'>
a = np.arange(24).reshape(3,4,2)
print(a, type(a))
#[[[ 0  1]
  [ 2  3]
  [ 4  5]
  [ 6  7]]

 [[ 8  9]
  [10 11]
  [12 13]
  [14 15]]

 [[16 17]
  [18 19]
  [20 21]
  [22 23]]] <class 'numpy.ndarray'>
print(a.ndim)
#3
for i in range(0,3):
    print(i,"축을 기준으로\n",a.sum(axis=i))
#0 축을 기준으로
 [[24 27]
 [30 33]
 [36 39]
 [42 45]]
1 축을 기준으로
 [[12 16]
 [44 48]
 [76 80]]
2 축을 기준으로
 [[ 1  5  9 13]
 [17 21 25 29]
 [33 37 41 45]]
print(a.shape)
#3,4,2
print(a.size)
#24
print(a.dtype.name, a.dtype.itemsize)
#int32 4
```

이렇게 여러 성질을 알아볼 수 있다. 그리고 array를 만들떄 강력한데

```ㅇ
x=np.array(range(1,21))
p_1=x[x%3==0] # 해당 범위 중 3배수
p_2=x[x%4==1] # 해당 범위 중 4배수+1

#둘 다 만족하는 식
p_3=x[(x%3==0) & (x%4==1)] 
p_3`=x[np.array((x%3==0)&(x%4==1))]
```

이런 식으로 조건식을 만족하는 array를 뽑아낼 수 있다.



## 데이터프레임의 다양한 응용

### 1. 함수 매핑

리스트와 마찬가지로 함수 매핑이 되는데, 이름이 apply로 조금 다르다.

1. 개별 원소에 함수 매핑

   add_10 함수를 따로 정의한 뒤에

   `sr1= df['age'].apply[add_10]` 이런 식으로 사용하여 시리즈를 만들어낼 수 있다.

   
