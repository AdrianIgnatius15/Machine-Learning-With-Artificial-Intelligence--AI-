# 📈 Regression Model Metrics: MSE and R²

## 📉 Mean Squared Error (MSE)

| Feature              | Description                                                                                                             |
| :------------------- | :---------------------------------------------------------------------------------------------------------------------- |
| **What it Measures** | The **average magnitude of the errors** (residuals) in a model's predictions.                                           |
| **Calculation**      | Finds the mean of the **squared** differences between the predicted values ($\hat{y}_i$) and the actual values ($y_i$). |
| **Formula**          | $$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$                                                           |
| **Interpretation**   | **Lower is better**. An MSE of 0 is a perfect model. It is very sensitive to **outliers** because errors are squared.   |
| **Units**            | In the **squared units** of the target variable (e.g., dollars squared, degrees squared).                               |

---

## $R^2$ (R-squared or Coefficient of Determination)

| Feature                 | Description                                                                                                                              |
| :---------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- |
| **What it Measures**    | The **proportion of the variance** in the dependent variable that is predictable from the model.                                         |
| **Comparison Baseline** | Compares your model's performance against the simplest model: one that always predicts the **mean** of the target variable.              |
| **Formula**             | $$R^2 = 1 - \frac{\text{SSR}}{\text{SST}} = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$              |
| **Interpretation**      | Ranges between 0 and 1. **Higher is better**. An $R^2$ of 0.80 means 80% of the target variable's variability is explained by the model. |
| **Units**               | It is a **scale-independent** ratio (a percentage or proportion).                                                                        |
