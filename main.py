from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, lag
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import Window
import matplotlib.pyplot as plt
import logging
import yfinance as yf
import streamlit as st
import pandas as pd


# get the stock data by yfinance
def get_stock_data(symbol, start_date, end_date):
    try:
        stock_data_pd = yf.download(symbol, start=start_date, end=end_date)
        if stock_data_pd.empty:
            raise ValueError("No data available for the given symbol and date range.")
            return None
        stock_data_pd.reset_index(inplace=True)
        stock_data_pd = stock_data_pd.dropna()
        return stock_data_pd
    except Exception as e:
        st.error(f"An error occurred while fetching stock data: {str(e)}")
        return None


# to transform the dataset to spark dataset 
def transform_dataset(spark, stock_data_pd):
    stock_data_spark = spark.createDataFrame(stock_data_pd)
    stock_data_spark = stock_data_spark.withColumn("date", to_date(stock_data_spark["Date"], 'yyyy-MM-dd'))
    stock_data_spark = stock_data_spark.orderBy("date")
    window_spec = Window().orderBy("date")
    stock_data_spark = stock_data_spark.withColumn("yesterday_volume", lag("Volume").over(window_spec)).na.drop()
    stock_data_spark = stock_data_spark.withColumn("yesterday_close", lag("Close").over(window_spec)).na.drop()
    stock_data_spark = stock_data_spark.dropna()
    return stock_data_spark


# split the dataset to train_dataset and test_dataset
def split_dataset(stock_data_spark, split_date):
    train_data = stock_data_spark.filter(col("date") <= split_date)
    test_data = stock_data_spark.filter(col("date") > split_date)
    return train_data ,test_data


# bulid the linear regression model
def bulid_model(feature_columns, train_data):
    vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    linear_regression = LinearRegression(featuresCol="features", labelCol="Close", predictionCol="prediction")
    pipeline = Pipeline(stages=[vector_assembler, linear_regression])
    model = pipeline.fit(train_data)
    return model


# evalute the model 
def evaluate_model(predictions):
    evaluator = RegressionEvaluator(labelCol="Close", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    mae_evaluator = RegressionEvaluator(labelCol="Close", predictionCol="prediction", metricName="mae")
    mae = mae_evaluator.evaluate(predictions)
    r2_evaluator = RegressionEvaluator(labelCol="Close", predictionCol="prediction", metricName="r2")
    r2 = r2_evaluator.evaluate(predictions)
    st.write("Root Mean Squared Error (RMSE) on test data:", rmse)
    st.write("Mean Absolute Error (MAE) on test data:", mae)
    st.write("R-squared on test data:", r2)


# visualize the predictions
def visualize_predictions(predictions):
    actual_vs_predicted = predictions.select("Date", "Close", "prediction").toPandas()
    fig, ax = plt.subplots()
    ax.plot(actual_vs_predicted["Date"], actual_vs_predicted["Close"], label="Actual Close Price")
    ax.plot(actual_vs_predicted["Date"], actual_vs_predicted["prediction"], label="Predicted Close Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.set_title("Actual vs. Predicted Close Price Over Time")
    ax.legend()
    st.pyplot(fig)


# input the volume and close price to predit the close price in the next day
def predict_next_day_value(model, spark):
    st.subheader("Predict Next Day's Close Price")
    user_volume = st.number_input("Enter Volume:", min_value=0.0)
    user_close = st.number_input("Enter Close Price:", min_value=0.0)
    user_data = pd.DataFrame({"yesterday_volume": [user_volume], "yesterday_close": [user_close]})
    user_spark_data = spark.createDataFrame(user_data)
    user_predictions = model.transform(user_spark_data)
    predicted_close = user_predictions.select("prediction").collect()[0]["prediction"]
    st.write(f"Predicted Close Price for the Next Day: {predicted_close}")


def main():
    st.title("Stock Price Prediction App")
    st.subheader("Enter the information")
    symbol = st.text_input("Enter Stock Symbol (e.g., 2330.TW):", "2330.TW")
    start_date = st.text_input("Enter Start Date (YYYY-MM-DD):", "2000-01-01")
    end_date = st.text_input("Enter End Date (YYYY-MM-DD):", "2023-01-01")
    spark = SparkSession.builder.appName("StockAnalyze").getOrCreate()
    logger = spark._jvm.org.apache.log4j
    logger.LogManager.getLogger("org.apache.spark.sql.execution.window.WindowExec").setLevel(logger.Level.ERROR)

    stock_data_pd = get_stock_data(symbol, start_date, end_date)
    if stock_data_pd is not None:
        stock_data_spark = transform_dataset(spark, stock_data_pd)
        split_date = st.text_input("Enter Split Date (YYYY-MM-DD):", "2021-12-31")
        train_data ,test_data= split_dataset(stock_data_spark, split_date) 
        feature_columns = ["yesterday_volume", "yesterday_close"]
        model = bulid_model(feature_columns, train_data)
        predictions = model.transform(test_data)

        st.subheader("Model Evaluation Metrics")
        evaluate_model(predictions)
        st.subheader("Visualize Predictions")
        visualize_predictions(predictions)
        predict_next_day_value(model, spark)
    
    spark.stop()
    
if __name__ == "__main__":
    main()


