import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# nge load dataset menngunakan pandas
df = pd.read_csv("data.csv")

#  pisah fitur dan target

x = df[["luas"]]
y = df[["harga"]]
# 3. Visualisasi data asli (scatter)
# =========================
plt.scatter(x, y)
plt.xlabel("Luas")
plt.ylabel("Harga")
plt.title("Scatter Plot: Luas vs Harga")
plt.show()

# split data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#  training model

model = LinearRegression()
model.fit(x_train,y_train)

# evaluasi
y_pred = model.predict(x_test )

print("MAE:", mean_absolute_error(y_test,y_pred))
print("MAE:", mean_squared_error(y_test,y_pred))
print("r2:", r2_score(y_test,y_pred))
# 7. Visualisasi garis regresi
# =========================
# Urutkan X biar garisnya rapi
x_sorted = x.sort_values(by="luas")
y_line = model.predict(x_sorted)

plt.scatter(x, y, label="Data Asli")
plt.plot(x_sorted, y_line, color="red", label="Garis Regresi", linewidth=2)
plt.xlabel("Luas")
plt.ylabel("Harga")
plt.title("Regresi Linear: Luas vs Harga")
plt.legend()
plt.show()


# simpan model
joblib.dump(model,"model.pkl")
print("model saved as model.pkl")