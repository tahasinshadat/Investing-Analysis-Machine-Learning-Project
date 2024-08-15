import boto3

# Create a session with your credentials
session = boto3.Session(
    aws_access_key_id='123',
    aws_secret_access_key='456',
    region_name='us-east-1'  # Specify the region
)

# Create a client for AWS Bedrock
bedrock = session.client(service_name="bedrock")

# Use the model identifier you provided
model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

# Make the API call to list foundation models
response = bedrock.list_foundation_models(byProvider="anthropic")

# Print model IDs from the response
for summary in response.get("modelSummaries", []):
    print(summary.get("modelId"))
