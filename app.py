import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix



df = pd.read_csv("diabetes.csv")

data_array = df.to_numpy()

dependent_data = []
status_data = []

dependent_success = []
dependent_failure = []
independent_success = []
independent_failure = []

success_count = 0
failure_count = 0

for entry in data_array:
    entry_status = 0

    if entry[-1] == 1:
        entry_status = 1
        success_count += 1
        dependent_success.append(entry[5])
        independent_success.append(entry[-1])
    else:
        failure_count += 1
        dependent_failure.append(entry[5])
        independent_failure.append(entry[-1])

    status_data.append(entry_status)
    dependent_data.append(entry[5])


two_d_array = [[status_data[i], dependent_data[i]] for i in range(len(status_data))]


group_1 = [item for item in two_d_array if item[0] == 1]
group_0 = [item for item in two_d_array if item[0] == 0]


def find_outliers(group):
    # Get second elements (the ones we want to check for outliers)
    values = np.array([item[1] for item in group])
    
    # Calculate Q1, Q3, and IQR
    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)
    IQR = Q3 - Q1
    
    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers = [item for item in group if item[1] < lower_bound or item[1] > upper_bound]
    
    # Return the outliers
    return outliers

outliers_1 = find_outliers(group_1)
outliers_0 = find_outliers(group_0)

filtered_data = [item for item in two_d_array if item not in outliers_1 and item not in outliers_0]


binary_variable = np.array([item[0] for item in filtered_data])

continuous_variable = np.array([item[1] for item in filtered_data])



x_data_column = "BMI"

plt.title("Diabetes Outcomes based on BMI")
plt.xlabel(x_data_column)
plt.ylabel('Probability of having diabetes')
plt.scatter(continuous_variable, binary_variable, marker='x')




fig = plt.figure(figsize=(15, 8))
plt.boxplot(dependent_success, showfliers=True)
plt.title("Raw Diabetes data")
plt.boxplot([dependent_success, dependent_failure], vert=False)




x = continuous_variable.reshape(-1, 1)

unweighted_model = model = LogisticRegression(solver='liblinear',  random_state=0).fit(x, binary_variable)
weighted_model = model = LogisticRegression(solver='liblinear',  random_state=0, class_weight='balanced').fit(x, binary_variable)

model = unweighted_model
model.fit(x, binary_variable)

bmi_range = np.linspace(x.min(), x.max())

probabilities = model.predict_proba(bmi_range.reshape(-1, 1))[:, 1]

plt.plot(bmi_range, probabilities, color='red', label='Sigmoid curve')

# plt.show()
print(model.intercept_)
print(model.coef_)





cm = confusion_matrix(binary_variable, model.predict(x))
print(cm)



print(classification_report(binary_variable,model.predict(x)))


