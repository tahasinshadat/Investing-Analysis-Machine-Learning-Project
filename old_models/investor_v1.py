import boto3
import botocore
import os
import json
import yfinance as yf
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_session_token = os.getenv('AWS_SESSION_TOKEN')

# Initialize Bedrock client
session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    aws_session_token=aws_session_token
)

bedrock_client = session.client(service_name='bedrock-runtime', region_name='us-east-1')
model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

def get_stock_data(symbol):
    """
    Fetches stock data for the given symbol.
    """
    stock = yf.Ticker(symbol)  # Fetch the stock data using yfinance
    info = stock.info  # Get current stock information
    history = stock.history(period="1y")  # Get historical data
    
    # Prepare the response
    response = {
        "current_price": info.get("currentPrice", "N/A"),
        "previous_close": info.get("regularMarketPreviousClose", "N/A"),
        "market_cap": info.get("marketCap", "N/A"),
        "pe_ratio": info.get("forwardEps", "N/A"),
        "historical_data": history.to_dict()  # Convert historical data to dictionary
    }

    # print("Stock Data for", stock_symbol, ":", response)
    
    return response

def get_investing_advice(stock_data, stock_symbol):
    """
    Gets investing advice about the stock from Claude using the Bedrock client.
    """
    # Construct the prompt with stock data
    prompt = (
        f"Here is the stock data for {stock_symbol}:\n"
        f"Current Price: {stock_data['current_price']}\n"
        f"Previous Close: {stock_data['previous_close']}\n"
        f"Market Cap: {stock_data['market_cap']}\n"
        f"P/E Ratio: {stock_data['pe_ratio']}\n"
        f"Historical Data: {stock_data['historical_data']}\n\n"
        f"Based on this data, what investment advice can you provide for {stock_symbol}?"
    )

    # Call the Bedrock client to get a response from Claude
    try:
        response = bedrock_client.invoke_model(
            modelId=model_id,  # Use the correct parameter name
            body=json.dumps({
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 150,
                "anthropic_version": "bedrock-2023-05-31"  # Ensure correct version is used (I just put random dates until it worked LOL)
            }),
            contentType='application/json'  # Use the correct parameter name
        )
        # Extract and return the model's response content
        response_body = json.loads(response['body'].read().decode('utf-8'))
        print(response_body)
        advice = response_body.get('content', [{}])[0].get('text', 'No advice available')
        return advice
    
    except botocore.exceptions.ClientError as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
stock_symbol = "PSA"
stock_data = get_stock_data(stock_symbol)
advice = get_investing_advice(stock_data, stock_symbol)
print("Investment Advice:", advice)
