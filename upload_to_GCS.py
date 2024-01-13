from google.cloud import storage
def upload_to_gcs(bucket_name, source_file, destination_blob_name):
    """Uploads a file to the GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file)

    print(f'File {source_file} uploaded to {destination_blob_name} in {bucket_name} bucket.')

upload_to_gcs("tipsy-tickle-monster", "data.pickle", "data.pickle")