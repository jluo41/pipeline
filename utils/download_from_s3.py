
import argparse
from code.pipeline.recfldtkn.base import download_s3_folder_or_file, delete_s3_folder_or_file, upload_s3_folder_or_file


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download data from S3')
    parser.add_argument('--local_path', type=str, help='Specific folder to download')
    args = parser.parse_args()

    # provide you aws credentials to the terminal.
    S3_CLIENT = boto3.client('s3')

    BUCKET_NAME = 'rxinform-analytics-personalization'
    S3_BASE_PATH = '000-SAGEMAKER-TRAINING-PIPELINE/REMOTE_REPO/'
    local_path = args.local_path
    # local_path = '_Data/1-Data_RFT/20250218_SMSAll'
    download_s3_folder_or_file(S3_CLIENT, BUCKET_NAME, S3_BASE_PATH, local_path)   

    # python utils/download_from_s3.py --local_path '_Data/1-Data_RFT/20250218_SMSAll'
