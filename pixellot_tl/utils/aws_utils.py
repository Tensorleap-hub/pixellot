from typing import Optional, List
import os
import boto3
import json
from tqdm import tqdm
from pixellot_tl.config import CONFIG, get_full_path


def _connect_to_s3() -> boto3.client:
    """
    This function establishes a connection to Amazon S3 using the provided authentication credentials.
    Output: Returns a boto3 S3 client object.
    """
    auth_secret_string = os.getenv("AUTH_SECRET")
    if auth_secret_string is None:
        raise ValueError("AUTH_SECRET environment variable not set")
    auth_secret_dic = json.loads(auth_secret_string)
    aws_access_key_id = auth_secret_dic.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = auth_secret_dic.get("AWS_SECRET_ACCESS_KEY")

    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError("AWS credentials are missing in AUTH_SECRET")

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    return s3_client


def download(
    cloud_file_path: str, local_file_path: Optional[str] = None, overwrite: bool = False
) -> str:
    # if local_file_path is not specified saving in home dir
    if local_file_path is None:
        local_file_path = os.path.join(
            get_full_path("tensorleap/data"), CONFIG["bucket_name"], cloud_file_path
        )

    # check if file is already exists
    if os.path.exists(local_file_path) and not overwrite:
        return local_file_path

    bucket_name = CONFIG["bucket_name"]
    dir_path = os.path.dirname(local_file_path)

    # s3_client = boto3.client('s3')
    s3_client = _connect_to_s3()
    os.makedirs(dir_path, exist_ok=True)
    s3_client.download_file(bucket_name, cloud_file_path, local_file_path)

    return local_file_path


def list_files_in_s3_folder(folder_name) -> List:
    bucket_name = CONFIG["bucket_name"]
    # Initialize the S3 client
    s3_client = _connect_to_s3()

    # Ensure the folder name ends with a '/'
    if not folder_name.endswith("/"):
        folder_name += "/"

    # List objects within the specified folder
    file_names = []
    continuation_token = None

    while True:
        list_kwargs = dict(Bucket=bucket_name, Prefix=folder_name)
        if continuation_token:
            list_kwargs["ContinuationToken"] = continuation_token

        response = s3_client.list_objects_v2(**list_kwargs)

        if "Contents" in response:
            file_names.extend(
                [obj["Key"].split("/")[-1] for obj in response["Contents"]]
            )

        if response.get("IsTruncated", False):
            continuation_token = response["NextContinuationToken"]
        else:
            break

    return file_names
