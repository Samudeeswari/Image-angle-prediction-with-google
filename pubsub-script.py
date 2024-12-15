import os
import json
from google.cloud import storage, bigquery, pubsub_v1
from datetime import datetime

# GCP Configuration
PROJECT_ID = "prosthetic-movement-project"
BUCKET_NAME = "prosthetic-data-storage"
BIGQUERY_DATASET = "movement_dataset"
BIGQUERY_TABLE = "angle_predictions"
PUBSUB_TOPIC = "data_upload_notifications"

# Local directories for images and angles
LOCAL_IMAGE_DIR = "./prosthetic_images"  # Directory containing images
LOCAL_ANGLE_FILE = "./prosthetic_angles.txt"  # File with angle data

def upload_images_to_gcs(bucket_name, local_image_dir):
    """
    Uploads images from the local directory to Google Cloud Storage.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    print("Uploading images to GCS...")
    for file_name in os.listdir(local_image_dir):
        if file_name.lower().endswith(('.jpg', '.png')):
            blob = bucket.blob(f"movement_images/{file_name}")
            blob.upload_from_filename(os.path.join(local_image_dir, file_name))
            print(f"Uploaded: {file_name} to gs://{bucket_name}/movement_images/{file_name}")

    print("Image upload completed!")

def upload_angles_to_bigquery(project_id, dataset_name, table_name, angle_file):
    """
    Uploads angle data from a local text file to BigQuery.
    """
    bigquery_client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{dataset_name}.{table_name}"

    # Read angles from the file
    rows_to_insert = []
    with open(angle_file, 'r') as f:
        for line in f.readlines():
            line = line.strip().replace(";", "")
            if ',' in line:
                x_angle, y_angle = map(float, line.split(','))
                rows_to_insert.append({
                    "x_angle": x_angle,
                    "y_angle": y_angle,
                    "upload_timestamp": datetime.utcnow().isoformat()
                })

    # Define schema if the table doesn't exist
    table = bigquery.Table(table_id, schema=[
        bigquery.SchemaField("x_angle", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("y_angle", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("upload_timestamp", "TIMESTAMP", mode="REQUIRED"),
    ])
    bigquery_client.create_table(table, exists_ok=True)

    # Insert rows into BigQuery
    errors = bigquery_client.insert_rows_json(table_id, rows_to_insert)
    if errors:
        print("Error uploading to BigQuery:", errors)
    else:
        print(f"Uploaded {len(rows_to_insert)} rows to BigQuery table {table_id}.")

def publish_messages_to_pubsub(topic_name, messages):
    """
    Publishes messages to a Pub/Sub topic.
    """
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, topic_name)

    print("Publishing messages to Pub/Sub...")
    for message in messages:
        # Encode message as JSON and publish
        message_data = json.dumps(message).encode("utf-8")
        future = publisher.publish(topic_path, message_data)
        print(f"Published message ID: {future.result()}")

    print("All messages published to Pub/Sub.")

def main():
    # Step 1: Upload images to GCS
    upload_images_to_gcs(BUCKET_NAME, LOCAL_IMAGE_DIR)

    # Step 2: Upload angles to BigQuery
    upload_angles_to_bigquery(PROJECT_ID, BIGQUERY_DATASET, BIGQUERY_TABLE, LOCAL_ANGLE_FILE)

    # Step 3: Send notifications via Pub/Sub
    messages = [
        {"type": "upload", "bucket": BUCKET_NAME, "path": "movement_images/", "status": "success"},
        {"type": "upload", "dataset": BIGQUERY_DATASET, "table": BIGQUERY_TABLE, "status": "success"}
    ]
    publish_messages_to_pubsub(PUBSUB_TOPIC, messages)

if __name__ == "__main__":
    main()
