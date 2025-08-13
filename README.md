# Sales Forecasting Project

This project focuses on forecasting future sales for various products (SKUs) across different warehouses and regions. It employs a hybrid forecasting model that combines the strengths of both traditional time series models and machine learning algorithms to achieve higher accuracy. Specifically, it uses a **SARIMA (Seasonal Autoregressive Integrated Moving Average)** model to capture the primary trend and seasonality in the sales data, and an **XGBoost (Extreme Gradient Boosting)** model to predict the residuals from the SARIMA model, thereby correcting its errors.

-----

## 📖 Dataset

The dataset used in this project is contained in the `Data (1).csv` file. It includes the following columns:

  * **Warehouse id**: Identifier for the warehouse.
  * **Region**: The geographical region of the warehouse (e.g., NORTH, SOUTH, EAST, WEST).
  * **SKU id**: Identifier for the Stock Keeping Unit (product).
  * **Monthly Sales Data**: A series of columns representing monthly sales from April 2018 to May 2021.

-----

## 📈 Methodology

The forecasting approach is a two-stage hybrid model:

1.  **SARIMA Model**: A SARIMA model is first trained on the historical sales data for each SKU-Warehouse combination to capture the underlying time series patterns, including trend and seasonality.

2.  **XGBoost Model for Residuals**: The residuals (the difference between the actual sales and the SARIMA model's predictions) are then used to train an XGBoost model. This allows the model to learn and predict the errors of the SARIMA model based on features like year, month, warehouse, and region.

The final forecast is the sum of the predictions from the SARIMA and XGBoost models, providing a more accurate and robust sales prediction.

-----

## ⚙️ Installation

To run this project, you need to have Python and the following libraries installed. You can install them using pip:

```bash
pip install pandas numpy statsmodels xgboost scikit-learn pmdarima matplotlib seaborn tqdm
```

-----

## 🚀 Usage

The main logic of the project is in the `Sales_forecasting.ipynb` Jupyter notebook. To use it, follow these steps:

1.  **Load the data**: Make sure the `Data (1).csv` file is in the same directory as the notebook.
2.  **Run the notebook**: Execute the cells in the notebook sequentially. The notebook will:
      * Load and preprocess the data.
      * Train the hybrid SARIMA + XGBoost model for each SKU-Warehouse combination.
      * Generate sales forecasts for June 2021.
      * Save the predictions to `Output.csv`.

-----

## 📊 Results

The project successfully generates sales forecasts for each SKU across all warehouses. The hybrid model's performance is evaluated using metrics like **Root Mean Squared Error (RMSE)** and **Mean Absolute Error (MAE)**. The results show that the hybrid model outperforms a standalone SARIMA model, with a lower RMSE and MAE, indicating a more accurate forecast.

The final predictions for June 2021 are saved in the `Output.csv` file.

-----


## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
