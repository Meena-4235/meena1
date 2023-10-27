import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
students=[[85,'m','verygood'],
          [95,'F','Excellent'],
          [75,None,'good'],
          [np.nan,'m','average'],
          [70,'m','good'],
          [np.nan,None,'Verygood'],
          [92,'F','Verygood'],
          [98,'m','Excellent']]
df=pd.DataFrame(students)
df.columns=['marks','gender','result']
imputer=SimpleImputer(missing_values=np.nan,strategy ='most_ frequent')
df.marks=imputer.fit_transform(df['marks'].values.reshape(-1,1))[:,0]
print(df.marks)

import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn import linear_model
df=pd.read_csv("homeprices.csv")
df
import matplotlib.pyplot as plt
#%matplotlib inline
plt.xlabel('areas(sqr ft)')
plt.ylabel('price(US$)')
plt.scatter(df.areas,df.price,color='red',marker='+')
plt.plot(df.areas,reg.predict(df[['areas']]),color='blue')
reg=linear_model.LinearRegression()
reg.fit(df[['areas']],df.price)
reg.predict([[3300]])
reg.coef_
reg.intercept_

# for the prediction
import pandas as pd
d=pd.read_csv("predict.csv")
p=reg.predict(d)
d['prices']=p
d.to_csv("predict.csv",index=False)

# for multiple regression
reg.fit(df[['areas','bedrooms','age']],df.price)
reg.predict([3000,3,40])
