"Congratulations! You just got some contract work with an Ecommerce company " \
    "based in New York City that sells clothing online but they also have in-store style and clothing advice sessions. " \
    "Customers come in to the store, have sessions/meetings with a personal stylist, then they can go home and order either on a " \
    "mobile app or website for the clothes they want.\n", "\n",
    "The company is trying to decide whether to focus their efforts on their mobile app experience or their website. " \
"They've hired you on contract to help them figure it out! Let's get started!\n",

"We'll work with the Ecommerce Customers csv file from the company. It has Customer info, suchas Email," \
" Address, and their color Avatar. T" \
"hen it also has numerical value columns:\n",

"* Avg. Session Length: Average session of in-store style advice sessions.\n",
"* Time on App: Average time spent on App in minutes\n",
"* Time on Website: Average time spent on Website in minutes\n",
"* Length of Membership: How many years the customer has been a member. \n",

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

customers = pd.read_csv('Ecommerce Customers.csv')

customers.describe()

# sns.jointplot(data= customers, x= 'Time on Website', y= 'Yearly Amount Spent')

# sns.jointplot(data= customers, x= 'Time on App', y= 'Yearly Amount Spent')

# sns.jointplot(data= customers, x= 'Time on App', y= 'Length of Membership', kind = 'hex')

# sns.pairplot(customers)


# **Based off this plot what looks to be the most correlated feature with Yearly Amount Spent?**
# ANS = LENGTH OF THE MEMBERSHIP.


# Create a linear model plot (using seaborn's lmplot) of  Yearly Amount Spent vs. Length of Membership.

# sns.lmplot(x= 'Length of Membership', y= 'Yearly Amount Spent', data= customers)


## Training and Testing Data

# Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets.\n",
##  "** Set a variable X equal to the numerical features of the customers and a variable y equal to the \"Yearly Amount Spent\" column.


a = customers.columns
# print(a)

y = customers['Yearly Amount Spent']
x = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()


aa = lm.fit(x_train, y_train)

COFF =lm.coef_

predict_lm = lm.predict(x_test)

# plt.scatter(y_test, predict_lm)

# plt.xlabel('Y test true values')
# plt.ylabel('Predicted Values')

print(predict_lm)

from sklearn import metrics

print('MAE', metrics.mean_absolute_error(y_test, predict_lm))

print('MSE', metrics.mean_squared_error(y_test, predict_lm))

print('RMSE', np.sqrt(metrics.mean_squared_error(y_test, predict_lm)))

EVC = metrics.explained_variance_score(y_test, predict_lm)

print(EVC)

print(COFF)


# Plot a histogram of the residuals and make sure it looks normally distributed. Use either seaborn distplot, or just plt.hist().
sns.distplot((y_test-predict_lm), bins= 50)

# Conclusion

# We still want to figure out the answer to the original question, do we focus our efforst on mobile app or website
# development? Or maybe that doesn't even really matter, and Membership Time is what is really important.  Let's see
# if we can interpret the coefficients at all to get an idea.

cdf = pd.DataFrame(COFF,x.columns,columns= ['coeff'])

print(cdf)

















plt.show()
