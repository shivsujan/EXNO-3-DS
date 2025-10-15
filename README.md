## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:

1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.

2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.

3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.

4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:

  # 1. FUNCTION TRANSFORMATION

• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation

  # 2. POWER TRANSFORMATION

• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
<img width="412" height="368" alt="image" src="https://github.com/user-attachments/assets/a559d887-6070-4da6-be0a-0d2f463f5a9b" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="254" height="171" alt="image" src="https://github.com/user-attachments/assets/c488694d-a453-410b-95f3-8bae1d5950c1" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="422" height="366" alt="image" src="https://github.com/user-attachments/assets/3046653c-224e-462a-a5a3-d39bdf4f50aa" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="503" height="370" alt="image" src="https://github.com/user-attachments/assets/f646526c-5300-4e89-9cfb-0031396fb76c" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]])) # Orders in Alphabetical Order Blue , Green, Red
df2=pd.concat([df2,enc],axis=1)
df2
```
<img width="523" height="368" alt="image" src="https://github.com/user-attachments/assets/cdd447aa-cc59-45d6-822f-971d611afc41" />

```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="691" height="329" alt="image" src="https://github.com/user-attachments/assets/4e2d3f4e-0d27-477c-89a4-95e577860c9f" />

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
<img width="681" height="369" alt="image" src="https://github.com/user-attachments/assets/84568e91-8f78-445b-b80d-8d95564e8925" />

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb
```
<img width="683" height="365" alt="image" src="https://github.com/user-attachments/assets/3c430c02-9dbf-4fcd-81a6-e2c007a92a71" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
<img width="653" height="366" alt="image" src="https://github.com/user-attachments/assets/a17277c3-93b3-499a-8835-f0f5378fbc6b" />

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
<img width="853" height="424" alt="image" src="https://github.com/user-attachments/assets/db7374d9-3240-49eb-bed1-a51322f9b3da" />

```
df.skew()
```
<img width="326" height="188" alt="image" src="https://github.com/user-attachments/assets/b711dc5a-aaca-4def-8663-76f5bb493f9c" />

```
np.log(df["Highly Positive Skew"])
```
<img width="392" height="410" alt="image" src="https://github.com/user-attachments/assets/76eada5a-0096-475e-b54b-80f637910ab8" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="353" height="411" alt="image" src="https://github.com/user-attachments/assets/d949845e-19f9-4e53-b407-76e051601904" />

```
np.sqrt(df["Highly Positive Skew"])
```
<img width="387" height="417" alt="image" src="https://github.com/user-attachments/assets/8b0ddda6-85db-4169-b9c5-ee40ba04c19c" />

```
np.square(df["Highly Positive Skew"])
```
<img width="397" height="413" alt="image" src="https://github.com/user-attachments/assets/6a471126-7c88-4936-bb75-94d32f6033d0" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="1040" height="419" alt="image" src="https://github.com/user-attachments/assets/6d7440b5-3c83-461b-8050-381f19c5bb6d" />

```
df.skew()
```
<img width="432" height="215" alt="image" src="https://github.com/user-attachments/assets/81ccbef7-6801-4f52-88f2-6a86cce4e626" />

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
<img width="539" height="249" alt="image" src="https://github.com/user-attachments/assets/19489661-965e-4a1f-a2fe-8fab5960c1a9" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
<img width="1479" height="420" alt="image" src="https://github.com/user-attachments/assets/1d555ab2-effe-44d6-bc6a-5de190117114" />

```
import seaborn as sns
import statsmodels.api as sm # STATS MODEL- STATISTICAL MODEL TO VISUALIZE DISTRIBUTION
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45') # QQ - QUANTILE QUANTILE PLOT
plt.show()
```
<img width="525" height="393" alt="image" src="https://github.com/user-attachments/assets/b74feb81-94db-4157-9090-5699d42d2644" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45') # RECIPROCAL
plt.show()
```
<img width="578" height="408" alt="image" src="https://github.com/user-attachments/assets/d261a8fd-955d-40c2-9666-860a4ea84a56" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="601" height="406" alt="image" src="https://github.com/user-attachments/assets/e3deea1c-dc9b-4d42-b01b-94a716fa49af" />

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
<img width="613" height="406" alt="image" src="https://github.com/user-attachments/assets/20730ffd-aef3-42fb-8174-35dad2d2aa4b" />

# RESULT:

Therefore all the codes are Executed Successfully .
