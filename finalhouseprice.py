
import streamlit as st
st.title("Final Project: House price in WA")

st.write("[Xinru Liu](https://github.com/xinruliu123/finalhosueprice)")

import numpy as np
import pandas as pd
import altair as alt
from pandas.api.types import is_numeric_dtype
import seaborn as sns
from scipy.stats import norm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress most warnings in TensorFlow
from pandas.api.types import is_numeric_dtype
pd.set_option('display.float_format', lambda x: '%.3f' % x)
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import matplotlib.pyplot as plt

df=pd.read_csv("data.csv")
st.markdown("## The origional data")
df
    
df = df[df.notna().all(axis=1)].copy() # data cleaning that remove all "" in df
df.sort_values("price", ascending=True) # find if there any value that does not make sense 
df = df.loc[df["price"] != 0] # Remove all pirce=0 row
df.sort_values("price", ascending=True) # check if the 0 price has been removed


st.markdown("## Check the normal distribution")
fig1, ax1 = plt.subplots(1, 2, figsize=(15, 10))# make 2 plot together next to each other to see the difference
sns.distplot(df['price'],fit=norm,kde=True,ax=ax1[0])# the graph of Unlogarithm
sns.distplot(np.log(df['price']),fit=norm,kde=True, ax=ax1[1]) # the graoy of logarithm
st.pyplot(fig1)  
st.write(" the two graph shows that the dependent varible does not fit the normal distrubution ")


st.markdown("## Check outliner Chart")
figure=plt.figure() #data cleaning: check it there is any outliner coorosponding to each x_vars
pair1=sns.pairplot(x_vars=["bedrooms","bathrooms","sqft_living","sqft_lot","floors","waterfront"],            
                                y_vars=["price"],data=df,dropna=True)#source：https://blog.csdn.net/weixin_39910043/article/details/111296923
pair11=sns.pairplot(x_vars=["view","condition","sqft_above","sqft_basement","yr_built","yr_renovated"],
                  y_vars=["price"],data=df,dropna=True)
st.pyplot(pair1)
st.pyplot(pair11)



df = df.drop(df[(df['sqft_living']>10000) &
                                        (df['price']<5000000)].index) # base on pair1 and pair11, it is obvious that price less than 5000000 and sqrt_livinh more than 10000 is not plausible.
df = df.drop(df[(df['bedrooms']<4) & 
                                        (df['price']>1500000)].index) 
df = df.drop(df[(df['bathrooms']<4) & 
                                        (df['price']>200000)].index)
df = df.drop(df[(df['sqft_lot']<1000) & 
                                        (df['price']>200000)].index) #data cleaning, dropping out the outliners
                                
st.markdown("## Check after dropping outlinear")
figure=plt.figure()
pair2=sns.pairplot(x_vars=["bedrooms","bathrooms","sqft_living","sqft_lot","floors","waterfront"],            
                                y_vars=["price"],data=df,dropna=True)#source：https://blog.csdn.net/weixin_39910043/article/details/111296923
pair22=sns.pairplot(x_vars=["view","condition","sqft_above","sqft_basement","yr_built","yr_renovated"],
                  y_vars=["price"],data=df,dropna=True)
st.pyplot(pair2)
st.pyplot(pair22)

# make all comlumn to numeric
numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]

st.sidebar.write("Choose a Chart")
if st.sidebar.checkbox('Chart 1(altair realted to data)'):# if chart 1 is choosen
    st.markdown("**Chart 1**")
    x_axis=st.selectbox("Choose a x-value", numeric_cols)# make a selectbox that allow ppl to choose value from numeric_col
    y_axis=st.selectbox("Choose a y-value", numeric_cols)
    st.altair_chart(alt.Chart(df).mark_circle().encode(
        x = x_axis,
        y = y_axis,
        color = alt.Color('price', scale=alt.Scale(scheme='turbo',reverse=True)),
        tooltip = ["city", "yr_built"]
        ).properties(
            width = 800,
            height = 1000,
            title="House Information in WA"
            ))
            
if st.sidebar.checkbox('Chart 2(which city has most expencive house price)'):
    st.markdown("**Chart 2**")
    st.altair_chart(alt.Chart(df).mark_bar().encode(
        x = "city",
        y = "bedrooms",
        color = "mean(price)"
        )   )
    st.write("From Chart 2 we know that the most expensive housing in WA is Medina")


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

#want to predict if the value seem like price in Medina
y = df["city"].map(lambda g_list: "Medina" if "Medina" in g_list else "Not Medina")#make y the output of if Medina is in city
X = df[["price","sqft_living"]]#prediction is based on input "price" and "sqft_livig"

clf.fit(X,y)

clf.predict(X)

clf.predict_proba(X.iloc[:-1])# predict the probability of all row in df
clf_df=pd.DataFrame(clf.predict_proba(X.iloc[:-1]), columns=clf.classes_)
clf_df
st.write("this is the prediction probability of the data if it will be in Medina")

predictors = ["price", "sqft_living"]

source = df

if st.sidebar.checkbox('Chart 3(realtion between sqrt_living and price'):
    st.markdown("**Chart 3**")
    point_plot = alt.Chart(source).mark_circle().encode(
        x = alt.X("price",scale=alt.Scale(zero=False)),
        y = alt.Y("sqft_living",scale=alt.Scale(zero=False)),
        color='city'
        )
    st.altair_chart(point_plot)
    #the altair chart show relationship btw prive anf sqrt_living



#Standardlize the data so that no one varaible will have large effect
scaler = StandardScaler()
scaler.fit(df[numeric_cols])

df[numeric_cols] = scaler.transform(df[numeric_cols])

#make a new column "is_seattle" to identify any row that is in the city Seattle
df["is_medina"] = df["city"].map(lambda g_list: "Medina" in g_list)

X_train = df[numeric_cols]# train x to the numeric columns
y_train = df["is_medina"]# train output y as is if is Medina or not

X_train.shape# check X_train and y_shape to ensure the input_shape in model layer
y_train.shape

model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape = (13,)),
        keras.layers.Dense(16, activation="sigmoid"),
        keras.layers.Dense(16, activation="sigmoid"),
        keras.layers.Dense(1,activation="sigmoid")
    ]
)# define model using keras with 2 hidden layer with 16 nerouns

model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=0.01),
    metrics=["accuracy"],
)# use binary_crossentropy as the loss function because only predict either ture or false outcome

history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)


plt.style.use('dark_background')
fig, ax = plt.subplots()
ax.plot(history.history['loss'])
ax.plot(history.history['val_loss'])
ax.set_ylabel('loss')
ax.set_xlabel('epoch')
ax.legend(['train', 'validation'], loc='upper right')

if st.sidebar.checkbox('Chart 4(training error chart'):
    st.markdown("**Chart 4**")
    st.pyplot(fig)
    st.write("From the chart we see that the validation is dropping and there is not much bias, since it does not overfitting")


st.markdown("## Reference:")
st.write("Data source:https://www.kaggle.com/shree1992/housedata")