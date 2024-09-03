import os
import mlflow

def run_command(command):
    """
    Run a shell command and capture its output and errors.
    """
    try:
        result = os.system(command)
        if result != 0:
            print(f"Error: Command '{command}' failed with exit code {result}")
    except Exception as e:
        print(f"Error running command: {e}")

def main():
    """
    Main function to run data fetching and preprocessing with MLFlow and DVC integration.
    """
    # Initialize MLFlow
    mlflow.set_experiment("Stock Forecasting Pipeline")

    # Start MLFlow run
    with mlflow.start_run(run_name="Data Fetching and Preprocessing") as run:
        # Log parameters (can be modified as needed)
        mlflow.log_param("Stage", "Data Fetching and Preprocessing")

        # Run data fetching script
        print("Running data fetching script...")
        run_command("python fetch_data.py")

        # Run data preprocessing and engineering script
        print("Running data preprocessing and feature engineering script...")
        run_command("python data_prep_engg.py")

        print("Running LSTM Model training script...")
        run_command("python train_lstm.py")

        # Log metrics or artifacts if applicable
        # For example:
        # mlflow.log_artifact("StockForecasting/data/TSLA.csv")

        # Optionally, log other metrics or parameters
        # mlflow.log_metric("Example_Metric", value)

if __name__ == "__main__":
    main()
