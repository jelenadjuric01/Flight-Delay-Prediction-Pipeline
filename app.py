from pyspark.sql import SparkSession
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql import functions as F
from pyspark.ml.evaluation import RegressionEvaluator
import sys
import time
import os

def main():
    try:# Ensure correct usage of the application with command-line arguments
        if len(sys.argv) != 3:
            print("Usage: spark-submit app.py <test_data_path> <best_model_path>")
            sys.exit(-1)
        # Initialize Spark session
        spark = SparkSession.builder.appName("Practical work - Group 4").getOrCreate()
        spark.sparkContext.setLogLevel("WARN")

        # Read test data
        test_data_path = sys.argv[1]
        try:
            test_data = spark.read.csv(f"{test_data_path}/*.csv", header=True, inferSchema=True)
        except Exception as e:
            print(f"Error reading test data: {e}")
            sys.exit(-1)
        # Drop forbidden and unnecessary columns as done in preprocessing
        columns_to_drop = [
            "ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay", "WeatherDelay", 
            "NASDelay", "SecurityDelay", "LateAircraftDelay", "TailNum", "TaxiOut", "CancellationCode", "Year", 
            "FlightNum", "Cancelled"
        ]
        test_data = test_data.drop(*columns_to_drop)
        # Convert specific columns to integer type for the preprocessing
        columns_to_cast = ["ArrDelay", "DepTime", "DepDelay", "Distance"]
        for column in columns_to_cast:
            test_data = test_data.withColumn(column, F.col(column).cast(IntegerType()))
        # Filter out rows with missing or negative values in critical columns
        test_data = test_data.filter(
            ~F.col('DepTime').isNull() & ~F.isnan('DepTime') & ~F.col('DepTime').eqNullSafe("NA") &
            ~F.col('ArrDelay').isNull() & ~F.isnan('ArrDelay') & ~F.col('ArrDelay').eqNullSafe("NA") &
            ~F.col('DepDelay').isNull() & ~F.isnan('DepDelay') & ~F.col('DepDelay').eqNullSafe("NA") &
            ~F.col('Distance').isNull() & ~F.isnan('Distance') & ~F.col('Distance').eqNullSafe("NA")
        )

        test_data = test_data.filter(
            (F.col('CRSElapsedTime') >= 0) & (F.col('DepDelay') >= 0) & (F.col('ArrDelay') >= 0)
        )

        # Calculate and join average delays for carrier, origin, and destination
        try:
            carrier_avg_delay = test_data.groupBy("UniqueCarrier").agg({"ArrDelay": "mean"}) \
                .withColumnRenamed("avg(ArrDelay)", "Carrier_Avrg_Delay")
            test_data = test_data.join(carrier_avg_delay, on="UniqueCarrier", how="left").drop("UniqueCarrier")

            origin_avg_delay = test_data.groupBy("Origin").agg({"ArrDelay": "mean"}) \
                .withColumnRenamed("avg(ArrDelay)", "Origin_Avrg_Delay")
            test_data = test_data.join(origin_avg_delay, on="Origin", how="left").drop("Origin")

            dest_avg_delay = test_data.groupBy("Dest").agg({"ArrDelay": "mean"}) \
                .withColumnRenamed("avg(ArrDelay)", "Dest_Avrg_Delay")
            test_data = test_data.join(dest_avg_delay, on="Dest", how="left").drop("Dest")
        except Exception as e:
            print(f"Error during data aggregation: {e}")
            sys.exit(-1)
        # Convert all columns to FloatType for compatibility with the model
        for column in test_data.columns:
            test_data = test_data.withColumn(column, F.col(column).cast(FloatType()))

        # Scaling features
        columns_to_scale = [
            'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime', 'CRSArrTime', 'CRSElapsedTime',
            'DepDelay', 'Distance', 'Carrier_Avrg_Delay', 'Origin_Avrg_Delay', 'Dest_Avrg_Delay'
        ]
        assembler = VectorAssembler(inputCols=columns_to_scale, outputCol="features_to_be_scaled")
        test_data = assembler.transform(test_data)
        scaler = MinMaxScaler(inputCol="features_to_be_scaled", outputCol="features")
        # Fit and transform the scaling model
        try:
            scaler_model = scaler.fit(test_data)
            test_data = scaler_model.transform(test_data)
        except Exception as e:
            print(f"Error during scaling: {e}")
            sys.exit(-1)

        # Load the best model
        best_model_path = sys.argv[2]
        try:
            model = CrossValidatorModel.load(best_model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(-1)

        # Perform predictions
        predictions = model.transform(test_data)
        predictions.select("prediction", "ArrDelay").show()
        
        # Performance evaluation
        evaluator = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        print(f"Root Mean Squared Error (RMSE): {rmse}")

        mae_evaluator = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="mae")
        mae = mae_evaluator.evaluate(predictions)
        print(f"Mean Absolute Error (MAE): {mae}")

        r2_evaluator = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName="r2")
        r2 = r2_evaluator.evaluate(predictions)
        print(f"R-squared (R2): {r2}")
        #Write results to results.txt that will be in folder results (both things will be made if they dont exists)
        os.makedirs("results",exist_ok=True)        
        with open("./results/results.txt","w") as file:
            file.write(f"Root Mean Squared Error (RMSE): {rmse}\n")
            file.write(f"Mean Absolute Error (MAE): {mae}\n")
            file.write(f"R-squared (R2): {r2}\n")
            file.write("Predictions (100 rows)\n")
            predictions_for_file = predictions.select("prediction","ArrDelay").limit(100).collect()
            for row in predictions_for_file:
                file.write(f"Prediction: {row['prediction']}, Actual: {row['ArrDelay']}\n")
        print("Results have been written to results/results.txt")
        time.sleep(300)
        #Stop the spark        
        spark.stop()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(-1)


if __name__ == "__main__":
    main()
