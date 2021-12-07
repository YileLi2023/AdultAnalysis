import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

st.set_page_config(
    page_title="AdultAnalysis",
    layout="wide",
    initial_sidebar_state="auto",
)
df = pd.read_csv("adult.csv")
df_initial = df.copy(deep=True)


# deal income
def dealIncome(income):
    if income == '<=50K':
        return 0
    elif income == '>50K':
        return 1


df['income'] = df.apply(lambda x: dealIncome(x['income']), axis=1)

df = df.drop(['fnlwgt'], axis=1)
df.replace('?', value=np.nan, inplace=True)
df = df.dropna(how='any')

st.write("# Data preprocessing")
st.write("Let's look at the initial data")
try:
    num = int(st.text_input("The amount of data you want to view", value='0', key=None))
    st.write(df_initial.head(num))
    st.write(
        "Then we We have performed data preprocessing on the original data. Because the data quality of the selected "
        "data set is relatively high, our data preprocessing is relatively simple")
    st.write(df.head(num))
except:
    st.error("Please check your input")

option = st.selectbox(
    'Which questions do you like best?',
    ['Question1', 'Question2', 'Question3', 'References'])
if option == 'Question1':
    st.write("## The first thing we want to explore is: How are the years of education of practitioners distributed?")
    chart_1 = alt.Chart(df).mark_line().encode(
        x="education-num",
        y="count()",
    )
    st.altair_chart(chart_1, use_container_width=True)
    st.write("It can be seen from the above table that the education years of employees are mostly distributed around "
             "9 years and 13 years, which exactly corresponds to the two academic qualifications of Bachelors and "
             "HS-grad. ")
elif option == 'Question2':
    st.write("## The second thing we want to explore is: Is there any relationship between the practitioner’s years of "
             "education, weekly working hours, and income?")
    df_visualization = df.copy(deep=True)
    chart_4 = alt.Chart(df_visualization).mark_bar().encode(
        x=alt.X('education-num', bin=alt.Bin(maxbins=30)),
        y='count()',
        color="income:N",
    )
    st.altair_chart(chart_4, use_container_width=True)

    chart_2 = alt.Chart(df_visualization).mark_circle().encode(
        x="education-num",
        y="hours-per-week",
        size="count()",
        color="income:N",
        tooltip=["income"],
    )
    st.altair_chart(chart_2, use_container_width=True)
    st.write("The figure shows that income and years of education are closely related. The longer the years of "
             "education, the higher the probability of income exceeding 50K. At the same time, income and working "
             "hours are also positively correlated. At the same time, we found an interesting phenomenon, that is, "
             "education. The longer the number of years, the lower the weekly working hours to earn more than 50K.")
elif option == 'Question3':
    st.write("## The third thing we want to explore is:Can we use machine learning methods to process the entire data "
             "set to obtain a model that can help us predict future income? ")
    st.write("Fortunately, we have selected indicators that are highly correlated with income in the first two "
             "questions, so we are going to use the selected indicators plus age indicators to classify the data set")
    X = df[['age', 'education-num', 'hours-per-week']]
    y = df['income']
    "indicator:"
    st.write(X.head())
    "label:"
    st.write(y.head())

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    # z - score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(X_train)
    st.write("std:")
    st.write(sc.scale_)
    st.write("mean:")
    st.write(sc.mean_)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train_std, y_train)
    st.write("The accuracy of KNN is: {:.5f}".format(knn.score(X_test_std, y_test)))

    "### The prediction:"
    X_test = pd.concat([X_test, y_test], axis=1)
    X_test['KNN_pre'] = knn.predict(X_test_std)
    st.write(X_test.head())
    X_test["id"] = X_test.index
    st.line_chart(X_test[['income', 'KNN_pre']].iloc[:50,:])
    st.write("Observing the prediction results of the KNN model we selected, we found that the accuracy of the "
             "prediction is about 76.1%. We may be able to adjust the hyperparameters of the model to make the model "
             "more accurate classification.")
st.write("## References")
st.write("GitHub: https://github.com/YileLi2023/AdultAnalysis")
st.write("The section of the app was taken from "
         "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn"
         ".model_selection.train_test_split")
st.write("The section of the app was taken from "
         "https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn"
         ".neighbors.KNeighborsClassifier")
st.write("The section of the app was taken from "
         "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn"
         ".preprocessing.StandardScaler")
