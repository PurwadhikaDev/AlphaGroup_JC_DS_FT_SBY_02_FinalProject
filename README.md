# E-commerce Customer Churn Analysis & Prediction

By:

    Benedict Emannuel
    Bryan Stevanus

## 1. Business Problem Understanding
### 1.1. Context

E-commerce is the buying and selling of goods and services over the internet using digital devices. In order to maintain profitability, an e-commerce business can use several metrics to evaluate its performance, identify problems it is currently or will be facing, and make plans and decisions to overcome said problems. One such metric is customer churn.

Customer churn, also known as customer attrition, refers to the phenomenon where customers stop doing business with a company or cease using its products or services over a given period. In the context of e-commerce, customer churn is particularly critical as it directly impacts revenue and long-term business sustainability. A high customer churn negatively impacts the business and puts a lot of pressure on the marketing team. This is because any customer gained would only purchase from the business only for a short while, which pressures the marketing team to seek out more customers, only for those hard sought customers to disappear in a short time frame, thus limiting the long-term gain from the marketing team’s effort.

In contrast, a low customer churn means that many customers continue to purchase products from the business, and any increase in customer from the marketing team’s effort have a positive cumulative impact to the business long-term profitability. High churn rates can indicate underlying issues such as poor customer satisfaction, inadequate service quality, or more attractive offers from competitors. Understanding the reasons behind customer churn involves analyzing customer behavior, transaction history, and engagement patterns to identify factors that contribute to their decision to leave. Effective churn analysis enables businesses to proactively address these issues, implement retention strategies, and ultimately enhance customer loyalty and lifetime value.

According to this article, the financial cost of customer churn comes in 3 fold, which are Lose Recurring Revenue, Lose Expansion Opportunity Revenue, and Cost to Acquire New Customers. Lose Recurring Revenue means that when the customer stops purchasing from the business, the business can not expect any revenue from that customer in the future. Lose Expansion Opportunity Revenue, according to the article, existing customers are 65% more likely to accept upselling compared to to the 13% of new prospects, which means that in addition to missing out recurring revenue, the business also loses out on any oppurtunity increase revenue from upselling their customers. Finally Cost to Acquire New Customers, means the business needs to invest more money to find new customers or reacquire those churned customers.

Lets use an example to better quantify the cost of customer churn. Let's say an e-commerce business has lost 100 customers in a year, a single customer generates $100 a year. That means the company has lost $10000 worth of recurring revenue for the following year. In addition, assuming the business can upsell to their customer to and aditional $20 a year 65% of the time, thats $1300 potential revenue increase lost. And finally, assuming it costs $100 to acquire new customers, then it would cost the business $10000 to regain the same amount of customers. Ultimately it would cost the business a total of $21300 the following year.

### 1.2. Problem Statement

In the dynamic and competitive e-commerce industry, retaining customers is crucial for maintaining steady revenue and achieving long-term success. However, understanding the factors that drive customer churn—when customers stop purchasing from the platform—remains a significant challenge.

A certain ecommerce company called company X, had a good reputation for years, however they received the data about their overall performance for some period. It can be seen that several customer churned and decided to use another ecommerce company service.

Company X wants to know which factor that cause their customer churned, and they want to implement a machine learing method to create a prediction based on the data they received, to prevent more customer churned in the future.

### 1.3. Analysis Goal

    Create a machine learning model that could predict if a customer would churn based on data provided by this dataset.
    Identify which factors that would cause a customer to churn.
    The final goal of the development of this machine learning model is to accurately predict customers that will churn and the relevant features that cause customers to churn. This would provide information to stakeholders to better understand what causes customer churn and devise strategies, such as discounts and promotions, to reduce and prevent customer churn

This project aims to analyze e-commerce customer churn data to identify patterns and predictors of churn, providing actionable insights to reduce churn rates. By employing data analytics and machine learning techniques to predict the churn rates in the future.

The prediction method is classification method, based on the data provided by the company which is a categorical data. We will try to use several algorithms such as XGBoost, KNN, or Random Forest

### 1.4. Stakeholders

    Marketing & Sales Team in E-commerce company: They can use the insights to create targeted campaigns aimed at retaining customers who are at risk of churning. They can leverage the data to identify at-risk customers and work on retention strategies to keep them engaged.

### 1.5. Analytic Apprioach

    Analyze the data to better understand the data and the value ranges of each feature/column

    Prepare the data by cleaning the data from missing and duplicate values

    Determine which preprocessing methods to use for each feature and create a column transformer and pipeline

    Determine which estimator has the best performance based on evaluation metric, specifically F1-score

    Perform hyperparameter tuning to the best estimator to improve model performance

    Evaluate model using evaluation metrics, namely Accuracy, Recall, Precision, and F1-score

    Perform model summary using SHAP to determine which features significantly impact car price prediction.

### 1.6 Metric Evaluation

To better assess a machine learning model's ability to classify each class, we will use confusion matrix, as seen below:
	Actual Positive 	Actual Negative
Predicted Positive 	True Positve 	False Positve
Predicted Negative 	False Negative 	True Negative

This table summarizes a model's performance by showing the number of correct and incorrect predictions for each class. From that table there are 2 types errors:

    Type 1 error = False Positives (FP):
    when a model incorrectly predicts a row with a positive class when in reality it is a negative class, this is called a false positive. This error results in investing resources to retain customers that would not churn, thus wasting resources.
    Type 2 error = False Negatives (FN):
    when a model incorrectly predicts a row with a negative class when in reality it is a positive class, this is called a false negative. This error results in ignoring customers that, in reality, does churn, which leads to lost revenue.

Confusion Matrix Term:

    True Positive (TP): The model predicts a customer will churn, and they actually do churn.

    True Negative (TN): The model predicts a customer will not churn, and they actually do not churn.

    False Positive (FP): The model predicts a customer will churn, but they actually do not churn.

    False Negative (FN): The model predicts a customer will not churn, but they actually do churn.

We are going to use F1-score for metric evaluation step. The F1 score is calculated as the harmonic mean of precision and recall. A harmonic mean is a type of average calculated by summing the reciprocal of each value in a data set and then dividing the number of values in the dataset by that sum. The value of the F1 score lies between 0 to 1 with 1 being a better.

## 2. Table Description

| Feature Name | Description |
|---|---|
| CustomerID | Unique customer ID |
| Churn | Flag indicating whether the customer churned (1) or not (0) |
| Tenure | Tenure of the customer in the organization |
| PreferredLoginDevice | Preferred device used by the customer to login (e.g., mobile, web) |
| CityTier | City tier classification (e.g., Tier 1, Tier 2, Tier 3) |
| WarehouseToHome | Distance between the warehouse and the customer's home |
| PreferredPaymentMode | Preferred payment method used by the customer (e.g., credit card, debit card, cash on delivery) |
| Gender | Gender of the customer |
| HourSpendOnApp | Number of hours spent on the mobile application or website |
| NumberOfDeviceRegistered | Total number of devices registered to the customer's account |
| PreferedOrderCat | Preferred order category of the customer in the last month |
| SatisfactionScore | Customer's satisfaction score with the service |
| MaritalStatus | Marital status of the customer |
| NumberOfAddress | Total number of addresses added to the customer's account |
| OrderAmountHikeFromlastYear | Percentage increase in order value compared to last year |
| CouponUsed | Total number of coupons used by the customer in the last month |
| OrderCount | Total number of orders placed by the customer in the last month |
| DaySinceLastOrder | Number of days since the customer's last order |
| CashbackAmount | Average cashback received by the customer in the last month |


## 3. Analysis & Machine Learning Conclusions

- Tenure has the strongest negative correlation with Churn, with the histogram visualization above we know that the rate of use of ecommerce services has increased in under 2 months.

- The proportion of customers who file a complaint is higher to stop using e-commerce services by 31.7% (3x, in proportion) than those who do not file a complaint.

- Surprisingly, for a satisfaction score of 4, 15.6% (109 out of 696) have churned, and for a satisfaction score of 5, 23.3% (176 out of 754) have churned, which are higher than satisfaction score of 1 and 2.

- Extreme Gradient Boost with Random Over Sampling is the best model for this problem. The base model performs similarly to the model after hyperparameter tuning.

- To evaluate the model, this analysis uses confusion matrix with emphasis on f1-score as evaluation metric. The model produced an f1-score of 0.91, which means the model can predict a customer would churn 91% of the time.
