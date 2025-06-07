
# 📊 Customer Churn Prediction - End-to-End Data Science Project

This is a simple **end-to-end Data Science project** to predict **Customer Churn** using a Random Forest Classifier.  
The project covers the basic pipeline from **loading data** to **training**, **saving**, and **loading** a machine learning model.

---

## 🚀 Project Workflow

1️⃣ **Load Dataset**  
2️⃣ **Preprocess Data**  
3️⃣ **Encode Categorical Variables**  
4️⃣ **Split Data (Train/Test)**  
5️⃣ **Train Random Forest Model**  
6️⃣ **Save Trained Model using `joblib`**  
7️⃣ **Load Model back and Predict**  

---

## 🗂️ Files

- `data.csv` — your input dataset (Telco Customer Churn or similar)
- `customer_churn_model.pkl` — trained Random Forest model (saved after training)
- `main_script.ipynb` or `.py` — your main code notebook or Python script

---

## 📦 Libraries Used

- `pandas`  
- `scikit-learn` (`sklearn`)  
- `joblib`  

---

## 🔍 How to Run This Project

### 1️⃣ Clone this repository

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2️⃣ Install dependencies

```bash
pip install pandas scikit-learn joblib
```

### 3️⃣ Place your `data.csv` file in the project directory

### 4️⃣ Run the script or Jupyter Notebook

```bash
# If using .py
python main_script.py

# If using Jupyter Notebook
jupyter notebook main_script.ipynb
```

### 5️⃣ After running, the trained model will be saved as:
```
customer_churn_model.pkl
```

---

## ⚙️ Model Workflow Example

```python
# Train and Save Model
rf.fit(X_train, y_train)
joblib.dump(rf, 'customer_churn_model.pkl')

# Load Model Later
rf_loaded = joblib.load('customer_churn_model.pkl')

# Predict on test data
predictions = rf_loaded.predict(X_test)
print(predictions)
```

---

## 🎯 Why Save & Load Models?

✅ No need to retrain every time  
✅ Fast deployment to apps (Flask / FastAPI / Streamlit)  
✅ Easy reproducibility  

---

## 🚧 Possible Improvements

- Add Exploratory Data Analysis (EDA)  
- Try different models (XGBoost, Logistic Regression)  
- Use Feature Engineering  
- Deploy as API or Streamlit app  
- Track metrics using MLflow  

---

## 💻 Example Use Cases

- Predict whether a customer is likely to churn
- Help Telco / SaaS companies reduce churn
- Build an internal churn dashboard

---

## 📚 References

- [Scikit-Learn Documentation](https://scikit-learn.org/stable/documentation.html)  
- [Pandas Documentation](https://pandas.pydata.org/docs/)  
- [Joblib Documentation](https://joblib.readthedocs.io/en/latest/)  

