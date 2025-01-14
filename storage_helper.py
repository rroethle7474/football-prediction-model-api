from typing import Union
from io import BytesIO
from azure.storage.blob import BlobServiceClient
import os

from io import BytesIO

class AzureStorageHelper:
    def __init__(self):
        connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        if not connection_string:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable not set")
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
    def upload_file(self, container_name: str, blob_name: str, file_data: Union[bytes, BytesIO, str]) -> None:
        """
        Upload a file to Azure Blob Storage.
        
        Args:
            container_name (str): Name of the container
            blob_name (str): Name to give the blob in storage
            file_data (Union[bytes, BytesIO, str]): File data to upload
        """
        container_client = self.blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(file_data, overwrite=True)
        
    def download_file(self, container_name: str, blob_name: str) -> bytes:
        """
        Download a file from Azure Blob Storage.
        
        Args:
            container_name (str): Name of the container
            blob_name (str): Name of the blob to download
            
        Returns:
            bytes: The file content as bytes
        """
        container_client = self.blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_name)
        return blob_client.download_blob().readall()
    
    def file_exists(self, container_name: str, blob_name: str) -> bool:
        """
        Check if a file exists in Azure Blob Storage.
        
        Args:
            container_name (str): Name of the container
            blob_name (str): Name of the blob to check
            
        Returns:
            bool: True if file exists, False otherwise
        """
        container_client = self.blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_name)
        return blob_client.exists()