from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier import logger
import os 
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()



STAGE_NAME = "Data Ingestion stage"



class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)

            dataset_dir = Path(data_ingestion_config.unzip_dir) / "Chicken-fecal-images"
            if dataset_dir.exists():
                logger.info(f"Local dataset already extracted at: {dataset_dir}")
            elif data_ingestion_config.source == "local":
                if not Path(data_ingestion_config.local_data_file).exists():
                    raise FileNotFoundError(
                        "Download the dataset ZIP from Google Drive and place it at "
                        f"{data_ingestion_config.local_data_file}"
                    )
                data_ingestion.extract_zip_file()
            else:
                data_ingestion.download_from_s3(
                    data_ingestion_config.bucket_name,
                    "data.zip",
                    data_ingestion_config.local_data_file,
                    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                    region_name="us-east-1",
                )
                data_ingestion.extract_zip_file()

        except Exception as e:
            raise e