import json
import boto3
import os
import tempfile
import logging
from pathlib import Path
import subprocess
import sys

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')

def lambda_handler(event, context):
    """
    Lambda handler for BirdNET-Analyzer
    Expected event format:
    {
        "input_bucket": "your-input-bucket",
        "input_key": "path/to/audio/file.wav",
        "output_bucket": "your-output-bucket",
        "output_prefix": "results/",
        "latitude": 42.5,
        "longitude": -76.45,
        "date": "2024-05-15",
        "min_conf": 0.25,
        "sensitivity": 1.0
    }
    """
    try:
        # Parse input parameters
        input_bucket = event['input_bucket']
        input_key = event['input_key']
        output_bucket = event['output_bucket']
        output_prefix = event.get('output_prefix', 'results/')
        
        # Optional parameters
        latitude = event.get('latitude')
        longitude = event.get('longitude')
        date = event.get('date')
        min_conf = event.get('min_conf', 0.25)
        sensitivity = event.get('sensitivity', 1.0)
        
        # Create temporary directories
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input_dir = Path(temp_dir) / "input"
            temp_output_dir = Path(temp_dir) / "output"
            temp_input_dir.mkdir()
            temp_output_dir.mkdir()
            
            # Download audio file from S3
            input_file_path = temp_input_dir / Path(input_key).name
            logger.info(f"Downloading {input_bucket}/{input_key} to {input_file_path}")
            s3_client.download_file(input_bucket, input_key, str(input_file_path))
            
            # Prepare BirdNET-Analyzer command
            cmd = [
                sys.executable, "-m", "birdnet_analyzer.analyze",
                "--i", str(temp_input_dir),
                "--o", str(temp_output_dir),
                "--min_conf", str(min_conf),
                "--sensitivity", str(sensitivity)
            ]
            
            # Add optional parameters
            if latitude is not None and longitude is not None:
                cmd.extend(["--lat", str(latitude), "--lon", str(longitude)])
            if date:
                cmd.extend(["--date", date])
            
            # Run BirdNET-Analyzer
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd="/var/task")
            
            if result.returncode != 0:
                logger.error(f"BirdNET-Analyzer failed: {result.stderr}")
                return {
                    'statusCode': 500,
                    'body': json.dumps({
                        'error': 'Analysis failed',
                        'details': result.stderr
                    })
                }
            
            logger.info(f"Analysis completed: {result.stdout}")
            
            # Upload results to S3
            uploaded_files = []
            for output_file in temp_output_dir.rglob("*"):
                if output_file.is_file():
                    # Create S3 key for output file
                    relative_path = output_file.relative_to(temp_output_dir)
                    s3_key = f"{output_prefix}{relative_path}"
                    
                    # Upload to S3
                    logger.info(f"Uploading {output_file} to {output_bucket}/{s3_key}")
                    s3_client.upload_file(str(output_file), output_bucket, s3_key)
                    uploaded_files.append(s3_key)
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Analysis completed successfully',
                    'input_file': f"{input_bucket}/{input_key}",
                    'output_files': uploaded_files,
                    'output_bucket': output_bucket
                })
            }
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Internal server error',
                'details': str(e)
            })
        }