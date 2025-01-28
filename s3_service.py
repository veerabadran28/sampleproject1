import boto3
import pandas as pd
from typing import Dict, List, Optional
import os
from datetime import datetime

class S3Service:
    def __init__(self, config: Dict):
        self.bucket = config['bucket']
        self.main_folder = config['main_folder']
        self.s3_client = boto3.client('s3')

    def upload_file(self, file, bank: str, year: str, period: str) -> str:
        """Upload file to S3 and return the S3 path."""
        try:
            # Generate S3 path
            filename = file.name
            s3_key = f"{self.main_folder}/{bank}/{year}/{period}/{filename}"
            
            # Ensure file pointer is at the beginning
            file.seek(0)
            
            # Upload file
            self.s3_client.upload_fileobj(file, self.bucket, s3_key)
            
            return f"s3://{self.bucket}/{s3_key}"
            
        except Exception as e:
            raise Exception(f"Error uploading file: {str(e)}")
        
    def get_distinct_values(self, field_type: str) -> List[str]:
        """Get distinct values for bank, year, or period from S3 structure."""
        try:
            # List objects in the main folder
            paginator = self.s3_client.get_paginator('list_objects_v2')
            #print(f"Paginator: {paginator}")
            #print(f"Bucket:{self.bucket}")
            prefix = f"{self.main_folder}/"
            #print(f"Prefix: {prefix}")
            
            unique_values = set()
            
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                #print("page: {page}")
                if 'Contents' in page:
                    for obj in page['Contents']:
                        # Parse path components
                        path_parts = obj['Key'].split('/')
                        if len(path_parts) >= 4:  # Ensure path has enough components
                            if field_type == "bank":
                                unique_values.add(path_parts[1])
                            elif field_type == "year":
                                unique_values.add(path_parts[2])
                            elif field_type == "period":
                                unique_values.add(path_parts[3])

            return sorted(list(unique_values))
            
        except Exception as e:
            raise Exception(f"Error getting distinct values: {str(e)}")

    def list_documents(self) -> pd.DataFrame:
        """List all documents in the main folder."""
        try:
            documents = []
            paginator = self.s3_client.get_paginator('list_objects_v2')
            prefix = f"{self.main_folder}/"
            
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        path_parts = obj['Key'].split('/')
                        if len(path_parts) >= 5:  # Ensure it's a file, not a folder
                            documents.append({
                                'Bank': path_parts[1],
                                'Year': path_parts[2],
                                'Period': path_parts[3],
                                'Filename': path_parts[4],
                                'Last Modified': obj['LastModified'],
                                'Size (MB)': round(obj['Size'] / (1024 * 1024), 2)
                            })
            
            return pd.DataFrame(documents)
            
        except Exception as e:
            raise Exception(f"Error listing documents: {str(e)}")

    def list_filtered_documents(self, bank: str, year: str, period: str) -> List[str]:
        """List documents matching the specified filters."""
        try:
            documents = []
            prefix = f"{self.main_folder}/{bank}/{year}/{period}/"
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    if not obj['Key'].endswith('/'):  # Skip folders
                        documents.append(obj['Key'])
            
            return documents
            
        except Exception as e:
            raise Exception(f"Error listing filtered documents: {str(e)}")

    def get_document_content(self, s3_path: str) -> bytes:
        """Get document content from S3."""
        try:
            # Parse S3 path
            path_parts = s3_path.replace('s3://', '').split('/')
            bucket = path_parts[0]
            key = '/'.join(path_parts[1:])
            
            # Get object
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            return response['Body'].read()
            
        except Exception as e:
            raise Exception(f"Error getting document content: {str(e)}")
        
    def get_document_content_explore(self, file_key: str) -> bytes:
        """Fetch document content from S3."""
        try:
            print(f"Fetching from S3: Bucket={self.bucket}, Key={file_key}")
            response = self.s3_client.get_object(Bucket=self.bucket, Key=file_key)
            return response['Body'].read()
        except Exception as e:
            raise Exception(f"Error fetching content: {str(e)}")
        
    def list_files(self, s3_path: str) -> List[str]:
        """List all files in the given S3 path."""
        try:
            # Parse the bucket name and prefix from the S3 path
            if not s3_path.startswith("s3://"):
                raise ValueError("Invalid S3 path. Must start with 's3://'")

            path_parts = s3_path.replace("s3://", "").split("/", 1)
            bucket = path_parts[0]
            prefix = path_parts[1] if len(path_parts) > 1 else ""

            # List objects using paginator
            paginator = self.s3_client.get_paginator("list_objects_v2")
            files = []

            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        # Append the full S3 path of the file
                        files.append(f"s3://{bucket}/{obj['Key']}")

            return files

        except Exception as e:
            raise Exception(f"Error listing files in S3 path '{s3_path}': {str(e)}")
        
    def file_exists(self, s3_path: str) -> bool:
        """Check if a file exists in S3."""
        try:
            # Parse S3 path
            path_parts = s3_path.replace('s3://', '').split('/')
            bucket = path_parts[0]
            key = '/'.join(path_parts[1:])
            
            try:
                self.s3_client.head_object(Bucket=bucket, Key=key)
                return True
            except self.s3_client.exceptions.ClientError as e:
                if e.response['Error']['Code'] == '404':
                    return False
                else:
                    raise
                    
        except Exception as e:
            print(f"Error checking file existence: {str(e)}")
            return False

    def upload_document(self, local_file_path: str, s3_path: str) -> bool:
        """Upload a document to S3."""
        try:
            # Parse S3 path
            path_parts = s3_path.replace('s3://', '').split('/')
            bucket = path_parts[0]
            key = '/'.join(path_parts[1:])
            
            # Upload file
            with open(local_file_path, 'rb') as file:
                self.s3_client.upload_fileobj(file, bucket, key)
            return True
            
        except Exception as e:
            print(f"Error uploading document: {str(e)}")
            return False