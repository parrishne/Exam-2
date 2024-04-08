import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Step 1: Load the data from the Excel file
file_path = "C:/Users/parrishne03/Downloads/Restaurant Revenue.xlsx"
df = pd.read_excel(file_path)

# Step 2: Preprocess the data and prepare features and target
# Selecting columns A to F as features
X = df[['Number_of_Customers', 'Menu_Price', 'Marketing_Spend', 
        'Average_Customer_Spending', 'Promotions', 'Reviews']]
# Target variable is Monthly_Revenue
y = df['Monthly_Revenue']

# Step 3: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Train the multiple regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions on the test set
predicted_monthly_revenue = model.predict(X_test)

# Step 7: Print the predicted monthly revenue along with the corresponding input data
# Only print once
if len(predicted_monthly_revenue) > 0:
    print("Predicted Monthly Revenue:", predicted_monthly_revenue[0])

# Print out the data used in prediction
# Each column should have one output printed out
data_used = pd.DataFrame(X_test[:1], columns=X.columns)

# Step 8: Fit the multiple regression model with statsmodels for full output
X_train_sm = sm.add_constant(X_train)
model_sm = sm.OLS(y_train, X_train_sm).fit()

# Step 9: Print out the multiple regression model summary
print("Multiple Regression Model Output:")
print(model_sm.summary())

print("Go Brewers!")

