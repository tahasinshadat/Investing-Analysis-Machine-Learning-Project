ANTHROPIC_API_KEY = "YOUR_API_KEY"  # Replace with your Anthropic API key

import requests
import json
import boto3
import botocore
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd

# Define your temporary credentials
aws_access_key_id = ''
aws_secret_access_key = ''
aws_session_token = ''

# Create a session using the temporary credentials
session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    aws_session_token=aws_session_token
)

region_name = 'us-east-1'  # Replace with your actual region
bedrock_client = session.client('bedrock', region_name=region_name)

model_id = 'anthropic.claude-3-5-sonnet-20240620-v1:0'

# Function to fetch data from Forbes link
def fetch_forbes_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Adjust the scraping logic based on the actual structure of the page
    stocks = []
    for item in soup.find_all('div', class_='some-class'):  # Update selector based on actual page structure
        name = item.find('h3').text  # Assuming stock names are in <h3> tags
        details = item.find('p').text  # Assuming stock details are in <p> tags
        stocks.append({'Name': name, 'Details': details})
    
    return stocks

# Fetch historical data for PSA from Yahoo Finance
def fetch_psa_data():
    psa = yf.Ticker("PSA")
    historical_data = psa.history(period="1y")  # Adjust period as needed
    return historical_data

def get_investor_advice(model_id, stock_data):
    
    return None


forbes_url = 'https://www.forbes.com/advisor/investing/best-real-estate-stocks/'
forbes_data = fetch_forbes_data(forbes_url)

psa_data = fetch_psa_data()
psa_data_json = psa_data.to_json()



advice = get_investor_advice(model_id, psa_data_json)
print("Forbes Data:", forbes_data)
print("PSA Historical Data:")
print(psa_data)
print("Investor Advice:", advice)
