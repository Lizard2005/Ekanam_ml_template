import os
from typing import TYPE_CHECKING

import boto3
from dotenv import load_dotenv

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
else:
    S3Client = object

load_dotenv()


def get_s3_client(
    key_id: str | None = None,
    access_key: str | None = None,
    endpoint_url: str | None = None,
) -> S3Client:
    if key_id is None:
        key_id = os.environ["AWS_ACCESS_KEY_ID"]
    if access_key is None:
        access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    if endpoint_url is None:
        endpoint_url = os.environ.get("AWS_ENDPOINT_URL", "https://s3-msk.tinkoff.ru")

    client: S3Client = boto3.client(
        "s3",
        aws_access_key_id=key_id,
        aws_secret_access_key=access_key,
        endpoint_url=endpoint_url,
    )
    return client


def s3_upload_file(
    file_local_src: str,
    file_s3_dst: str,
    remove_src_file: bool = False,
    bucket_name: str | None = None,
) -> None:
    if bucket_name is None:
        bucket_name = os.environ["AWS_BUCKET_NAME"]
    s3_client = get_s3_client()

    s3_client.upload_file(file_local_src, bucket_name, file_s3_dst)
    if remove_src_file:
        os.remove(file_local_src)


def s3_download_file(
    file_s3_src: str,
    file_local_dst: str,
    bucket_name: str | None = None,
) -> None:
    if bucket_name is None:
        bucket_name = os.environ["AWS_BUCKET_NAME"]
    s3_client = get_s3_client()

    s3_client.download_file(bucket_name, file_s3_src, file_local_dst)


def s3_list_files(
    prefix: str = "",
    bucket_name: str | None = None,
) -> list[str]:
    if bucket_name is None:
        bucket_name = os.environ["AWS_BUCKET_NAME"]
    s3_client = get_s3_client()

    files = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix).get("Contents")
    return [file["Key"] for file in files]
