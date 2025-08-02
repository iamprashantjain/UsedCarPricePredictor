import dagshub
import mlflow

# Initialize DagsHub connection
dagshub.init(repo_owner='iamprashantjain', repo_name='UsedCarPricePredictor', mlflow=True)

# Set the tracking URI to DagsHub
mlflow.set_tracking_uri("https://dagshub.com/iamprashantjain/UsedCarPricePredictor.mlflow")

# Create or set an experiment
experiment_name = "Test_Experiment"
mlflow.set_experiment(experiment_name)

