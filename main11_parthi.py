import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import boto3
from io import StringIO
import sagemaker
from sagemaker.amazon.amazon_estimator import get_image_uri

def main():
    st.title("Trend Prediction App")
    
    # Initialize AWS S3 client
    s3 = boto3.client("s3")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read the CSV file
        csv_data = StringIO(uploaded_file.getvalue().decode("utf-8"))
        data = pd.read_csv(csv_data)
        
        # Display the data
        st.subheader("Data")
        st.dataframe(data)
        
        # Select the date and value columns
        date_column = st.selectbox("Select the date column", data.columns)
        value_column = st.selectbox("Select the value column", data.columns)
        
        # Convert the date column to datetime format
        data[date_column] = pd.to_datetime(data[date_column])
        
        # Set the date column as the index
        data.set_index(date_column, inplace=True)
        
        # Specify the number of future dates to predict
        num_future_dates = st.number_input("Enter the number of future dates to predict", min_value=1, value=30)
        
        if st.button("Predict"):
            # Upload the data to S3
            bucket_name = "avengers-ba007"
            file_name = "data.csv"
            csv_buffer = StringIO()
            data.to_csv(csv_buffer)
            s3.put_object(Bucket=bucket_name, Key=file_name, Body=csv_buffer.getvalue())
            
            # Create a SageMaker session
            sagemaker_session = sagemaker.Session()
            
            # Specify the Docker image for the SageMaker algorithm
            docker_image_name = get_image_uri(boto3.Session().region_name, "forecasting-deepar")
            
            # Create an estimator
            estimator = sagemaker.estimator.Estimator(
                docker_image_name,
                role="your-sagemaker-execution-role",
                instance_count=1,
                instance_type="ml.c4.xlarge",
                output_path=f"s3://{bucket_name}/output",
                sagemaker_session=sagemaker_session,
            )
            
            # Set the hyperparameters
            estimator.set_hyperparameters(
                time_freq="D",
                epochs=100,
                mini_batch_size=32,
                learning_rate=0.001,
                context_length=30,
                prediction_length=num_future_dates,
            )
            
            # Specify the data channels
            data_channels = {"train": f"s3://{bucket_name}/{file_name}"}
            
            # Train the model
            estimator.fit(inputs=data_channels)
            
            # Deploy the model
            predictor = estimator.deploy(initial_instance_count=1, instance_type="ml.m4.xlarge")
            
            # Make predictions for future dates
            last_date = data.index[-1]
            future_dates = [last_date + timedelta(days=x) for x in range(1, num_future_dates + 1)]
            future_dataset = pd.DataFrame({"date": future_dates})
            future_dataset["date"] = pd.to_datetime(future_dataset["date"])
            future_dataset.set_index("date", inplace=True)
            
            # Convert the future dataset to JSON format
            future_dataset_json = future_dataset.to_json(orient="split")
            
            # Make predictions using the deployed model
            predictions = predictor.predict(future_dataset_json)
            
            # Create a dataframe with future dates and predicted values
            future_data = pd.DataFrame({"Date": future_dates, "Predicted Value": predictions})
            future_data.set_index("Date", inplace=True)
            
            # Display the predicted values
            st.subheader("Predicted Values")
            st.dataframe(future_data)
            
            # Visualize the trend
            st.subheader("Trend Visualization")
            combined_data = pd.concat([data, future_data])
            st.line_chart(combined_data)
            
            # Delete the endpoint
            predictor.delete_endpoint()

if __name__ == "__main__":
    main()