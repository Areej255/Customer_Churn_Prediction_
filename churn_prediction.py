# importing libs
import pandas as pd
import numpy as np
# visualization libs
import seaborn as sns
import matplotlib.pyplot as plt
# ml tools

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# loading dataset
df=pd.read_csv('Telco-Customer-Churn.csv')
# print(df.head())
# print(df.info())

# basic data cleaning
# removing the duplicate rows
df.drop_duplicates(inplace=True)

# now checking for missing values
# print(df.isnull().sum())

# handling missing values
# numerical columns -> fill with median(robust against outliers)
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# categorical columns -> fill with mode(most frequent value)
cat_cols = df.select_dtypes(include=["object"]).columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])


# encode target variable
# convert "churn" column to binary values: Yes -> 1, No -> 0
df["Churn"] = df["Churn"].map({"Yes":1, "No" :0})
# print(df["Churn"])


# print(x_train.dtypes)
# dropping non informative customer id column
# x_train.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
cat_cols = df.select_dtypes(include=["object"]).columns
# print(cat_cols)


# feature and target separation
x = df.drop("Churn", axis=1)
y = df["Churn"]

# onehotencoding
x = pd.get_dummies(x, drop_first=True)

# splitting the dataset into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.2,random_state=42,stratify=y
)

# verifying all features are numeric
# print(x_train.select_dtypes(include=["object"]).columns)

# feature scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
# x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# model training
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(x_train_scaled, y_train)

# making predictions
y_pred = model.predict(x_test_scaled)

# evaluating the model
# accuracy
# logis_reg_score = accuracy_score(y_test,y_pred)
# print(f"Accuracy: {logis_reg_score:.4f}")
# # confusion matrix
# c_m = confusion_matrix(y_test,y_pred)
# print("\nConfusion Matrix:\n", c_m)
# # classification report
# class_report = classification_report(y_test,y_pred)
# print("\nClassification Report:\n", class_report)

# visualizing confusion matrix
# sns.heatmap(c_m, annot=True, fmt="d", cmap="Greens")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix Visualization")
# plt.savefig("confusion_matrix.png")
# plt.close()

# adding business intelligence
# df["Predicted_Churn"] = model.predict(scaler.transform(x))
df["Churn_Probability"] = model.predict_proba(scaler.transform(x))[:, 1]
# high risk customers checking churn probability > 0.7
high_risk_customers = df[df["Churn_Probability"] > 0.7]
# print(high_risk_customers[["customerID", "Churn_Probability"]])

# # feature importance using coefficients from logistic regression
coef = model.coef_[0]
featuress = x.columns

feature_importance = pd.DataFrame({
    "feature": featuress,
    "importance": coef
}).sort_values(by="importance", ascending=False)
# print("\nFeature Importance:\n", feature_importance)

# selecting top 15
top_features = feature_importance.reindex(
    feature_importance.importance.abs().sort_values(ascending=False).index
).head(15)

# plotting
# plt.figure(figsize=(8,6))
# plt.barh(top_features.feature, top_features.importance)
# plt.xlabel("Coefficient Value")
# plt.title("Top 15 Feature Effects (Logistic Regression)")
# plt.show()

# # visualizing feature importance
# plt.figure(figsize=(10,6))
# sns.barplot(x="importance", y="feature", data=feature_importance)
# plt.title("Feature Importance Visualization")
# plt.show()


# comparing multiple models
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)
rf_y_pred = rf_model.predict(x_test)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
# print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# # # comparing accuracies
# print(f"Logistic Regression Accuracy: {logis_reg_score:.4f}")
# print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# # visualizing model comparison
# model_comparison = pd.DataFrame({
#     "Model": ["Logistic Regression", "Random Forest"],
#     "Accuracy": [logis_reg_score, rf_accuracy]
# })
# # plt.figure(figsize=(8,5))
# sns.barplot(x="Model", y="Accuracy", data=model_comparison)
# plt.title("Model Accuracy Comparison")
# plt.ylim(0,1)
# plt.show()

# threshold optimization
# setting threshold to 0.8

from sklearn.metrics import precision_recall_curve
# getting churn probabilities
y_scores = model.predict_proba(x_test_scaled)[:,1]
# optimizing threshold for better precision-recall trade-off
precion, recall, thresholds = precision_recall_curve(y_test, y_scores)
# converting to dataframe for easy ananlysis
pr_df = pd.DataFrame({
    "Threshold": thresholds,
    "Precision": precion[:-1],
    "Recall": recall[:-1]
})
# now choosing threshold where recall > 80% and precision is maximized
filtered = pr_df[(pr_df["Recall"]>=0.8) & (pr_df["Precision"]>=0.40)]
if not filtered.empty:
    optimal_threshold = filtered.iloc[0]
else:
    optimal_threshold = pr_df.iloc[-1]


# optimal_threshold = pr_df[(pr_df["Recall"]>=0.8) & (pr_df["Precision"]>=0.40)].iloc[0] 
# # print("optimal_threshold:", round(optimal_threshold,4))
# # by doing this “By lowering the decision threshold to 9.4%, the model catches almost all churners (95%) but only 40% of flagged customers actually churn.”

# business impact simulation
# assuming cost of retention offer is $100 per customer
retention_cost_per_customer = 100
# assuming average revenue per customer is $500
# revenue saved per retained customer
revenue_per_customer = 500
# true churners caught by the model(true positives)
true_positives = ((y_test == 1)&(y_scores >= optimal_threshold["Threshold"])).sum()
# financial calculations
revenue_saved = true_positives * revenue_per_customer
campaign_cost = ((y_scores >= optimal_threshold["Threshold"]).sum()) * retention_cost_per_customer
net_profit = revenue_saved - campaign_cost
# print(f"Revenue Saved: ${revenue_saved}")
# print(f"Campaign Cost: ${campaign_cost}")
# print(f"Net Profit gain from Retention Campaign: ${net_profit}")

# import os
# import joblib

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# joblib.dump(model, os.path.join(BASE_DIR, "churn_prediction_model.pkl"))
# joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))
# joblib.dump(feature_columns, os.path.join(BASE_DIR, "feature_columns.pkl"))

feature_columns = x.columns.tolist()

# now saving the trained model and scaler for future use
import joblib

joblib.dump(model, "churn_prediction_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(x_train.columns.tolist(), "feature_columns.pkl")

print("PKL FILES SAVED SUCCESSFULLY")



# import os
# import joblib
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression

# train model
# model = LogisticRegression()
# model.fit(x_train, y_train)

# # scaler already fit
# # feature_columns already defined

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# joblib.dump(model, os.path.join(BASE_DIR, "churn_prediction_model.pkl"))
# joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))
# joblib.dump(feature_columns, os.path.join(BASE_DIR, "feature_columns.pkl"))




