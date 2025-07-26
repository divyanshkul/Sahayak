from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError
import os
from typing import Optional, Union
from pathlib import Path
import mimetypes
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class GCPStorageUploader:
    """Utility class for uploading files to Google Cloud Storage bucket."""
    
    def __init__(self, bucket_name: str, credentials_path: Optional[str] = None):
        """
        Initialize GCP Storage uploader.
        
        Args:
            bucket_name: Name of the GCS bucket
            credentials_path: Path to GCP service account key file (optional if using default credentials)
        """
        self.bucket_name = bucket_name
        
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        try:
            self.client = storage.Client()
            self.bucket = self.client.bucket(bucket_name)
        except Exception as e:
            logger.error(f"Failed to initialize GCP Storage client: {e}")
            raise
    
    def upload_file(
        self, 
        file_path: Union[str, Path], 
        destination_blob_name: Optional[str] = None,
        content_type: Optional[str] = None,
        make_public: bool = False,
        folder: Optional[str] = None
    ) -> Optional[str]:
        """
        Upload a file to GCS bucket.
        
        Args:
            file_path: Local path to the file to upload
            destination_blob_name: Name for the file in GCS (optional, uses filename if not provided)
            content_type: MIME type of the file (optional, auto-detected if not provided)
            make_public: Whether to make the uploaded file publicly accessible
            folder: Folder/prefix to upload the file under
        
        Returns:
            Public URL of the uploaded file if make_public=True, otherwise blob name
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None
            
            # Generate destination blob name if not provided
            if destination_blob_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                destination_blob_name = f"{timestamp}_{file_path.name}"
            
            # Add folder prefix if specified
            if folder:
                destination_blob_name = f"{folder.strip('/')}/{destination_blob_name}"
            
            # Auto-detect content type if not provided
            if content_type is None:
                content_type, _ = mimetypes.guess_type(str(file_path))
                if content_type is None:
                    content_type = 'application/octet-stream'
            
            # Create blob and upload
            blob = self.bucket.blob(destination_blob_name)
            
            with open(file_path, 'rb') as file_data:
                blob.upload_from_file(file_data, content_type=content_type)
            
            logger.info(f"File {file_path} uploaded to {destination_blob_name}")
            
            # Make public if requested
            if make_public:
                try:
                    blob.make_public()
                    public_url = blob.public_url
                    logger.info(f"File made public: {public_url}")
                    return public_url
                except GoogleCloudError as e:
                    if "uniform bucket-level access" in str(e).lower():
                        logger.warning(f"Cannot make file public due to uniform bucket-level access. File uploaded privately.")
                        logger.info(f"File accessible at: gs://{self.bucket_name}/{destination_blob_name}")
                        return f"gs://{self.bucket_name}/{destination_blob_name}"
                    else:
                        raise
            
            return destination_blob_name
            
        except GoogleCloudError as e:
            logger.error(f"GCP error uploading file {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error uploading file {file_path}: {e}")
            return None
    
    def upload_from_memory(
        self,
        file_data: bytes,
        destination_blob_name: str,
        content_type: str = 'application/octet-stream',
        make_public: bool = False,
        folder: Optional[str] = None
    ) -> Optional[str]:
        """
        Upload file data from memory to GCS bucket.
        
        Args:
            file_data: File data as bytes
            destination_blob_name: Name for the file in GCS
            content_type: MIME type of the file
            make_public: Whether to make the uploaded file publicly accessible
            folder: Folder/prefix to upload the file under
        
        Returns:
            Public URL of the uploaded file if make_public=True, otherwise blob name
        """
        try:
            # Add folder prefix if specified
            if folder:
                destination_blob_name = f"{folder.strip('/')}/{destination_blob_name}"
            
            # Create blob and upload
            blob = self.bucket.blob(destination_blob_name)
            blob.upload_from_string(file_data, content_type=content_type)
            
            logger.info(f"File uploaded to {destination_blob_name} from memory")
            
            # Make public if requested
            if make_public:
                try:
                    blob.make_public()
                    public_url = blob.public_url
                    logger.info(f"File made public: {public_url}")
                    return public_url
                except GoogleCloudError as e:
                    if "uniform bucket-level access" in str(e).lower():
                        logger.warning(f"Cannot make file public due to uniform bucket-level access. File uploaded privately.")
                        logger.info(f"File accessible at: gs://{self.bucket_name}/{destination_blob_name}")
                        return f"gs://{self.bucket_name}/{destination_blob_name}"
                    else:
                        raise
            
            return destination_blob_name
            
        except GoogleCloudError as e:
            logger.error(f"GCP error uploading file from memory: {e}")
            return None
        except Exception as e:
            logger.error(f"Error uploading file from memory: {e}")
            return None
    
    def delete_file(self, blob_name: str) -> bool:
        """
        Delete a file from GCS bucket.
        
        Args:
            blob_name: Name of the blob to delete
        
        Returns:
            True if successful, False otherwise
        """
        try:
            blob = self.bucket.blob(blob_name)
            blob.delete()
            logger.info(f"File {blob_name} deleted successfully")
            return True
        except GoogleCloudError as e:
            logger.error(f"GCP error deleting file {blob_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error deleting file {blob_name}: {e}")
            return False
    
    def file_exists(self, blob_name: str) -> bool:
        """
        Check if a file exists in the GCS bucket.
        
        Args:
            blob_name: Name of the blob to check
        
        Returns:
            True if file exists, False otherwise
        """
        try:
            blob = self.bucket.blob(blob_name)
            return blob.exists()
        except Exception as e:
            logger.error(f"Error checking if file exists {blob_name}: {e}")
            return False
    
    def get_public_url(self, blob_name: str) -> Optional[str]:
        """
        Get the public URL for a blob (if it's public).
        
        Args:
            blob_name: Name of the blob
        
        Returns:
            Public URL if the blob is public, None otherwise
        """
        try:
            blob = self.bucket.blob(blob_name)
            if blob.exists():
                return blob.public_url
            return None
        except Exception as e:
            logger.error(f"Error getting public URL for {blob_name}: {e}")
            return None
    
    def list_files(self, prefix: Optional[str] = None, max_results: int = 100) -> list:
        """
        List files in the bucket.
        
        Args:
            prefix: Prefix to filter files (optional)
            max_results: Maximum number of results to return
        
        Returns:
            List of blob names
        """
        try:
            blobs = self.client.list_blobs(
                self.bucket_name, 
                prefix=prefix, 
                max_results=max_results
            )
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []


# Convenience function for quick uploads
def quick_upload(
    file_path: Union[str, Path],
    bucket_name: str,
    credentials_path: Optional[str] = None,
    make_public: bool = False,
    folder: Optional[str] = None
) -> Optional[str]:
    """
    Quick utility function to upload a file to GCS.
    
    Args:
        file_path: Local path to the file to upload
        bucket_name: Name of the GCS bucket
        credentials_path: Path to GCP service account key file (optional)
        make_public: Whether to make the uploaded file publicly accessible
        folder: Folder/prefix to upload the file under
    
    Returns:
        Public URL if make_public=True, otherwise blob name
    """
    uploader = GCPStorageUploader(bucket_name, credentials_path)
    return uploader.upload_file(file_path, make_public=make_public, folder=folder)