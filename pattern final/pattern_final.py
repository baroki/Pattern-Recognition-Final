
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

#  Dosya yolları
ages_file = "C:/Users/utkus/Desktop/pattern final/Ages.csv"  
data_file = "C:/Users/utkus/Desktop/pattern final/data.csv"  

ages_df = pd.read_csv(ages_file)
data_df = pd.read_csv(data_file)

merged_df = pd.merge(data_df, ages_df, on="Sample Accession")

X = merged_df.iloc[:, 2:-1]  
y = merged_df["Age"]         

subset_size = 1000
X_subset = X[:subset_size]
y_subset = y[:subset_size]

X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42, n_jobs=1)
model.fit(X_train, y_train)

#  Performans verilerini hesaplama
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2):", r2)

# Scatter plot: Gerçek yaş (x-ekseni) vs. Tahmin edilen yaş (y-ekseni)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color="blue", edgecolors="k")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", linewidth=2)
plt.xlabel("Actual Ages")
plt.ylabel("Predicted Ages")
plt.title("Scatter Plot of Predicted and Actual Ages")
plt.grid(alpha=0.3)
plt.show()
