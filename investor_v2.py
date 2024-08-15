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

region_name = 'us-east-1'
kb_id = 'KDW4G5S26S'
bedrock_agent_runtime_client = boto3.client("bedrock-agent-runtime", region_name=region_name)
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
model_arn = f'arn:aws:bedrock:us-east-1::foundation-model/{model_id}'

bedrock_client = session.client(service_name='bedrock-agent-runtime', region_name='us-east-1')

# prompt = "Summarize PSA's 10Q"
# if metadata_filter is None:
# try:
#     response = bedrock_client.retrieve_and_generate(
#         input={
#             'text': prompt
#         },
#         retrieveAndGenerateConfiguration={
#             'type': 'KNOWLEDGE_BASE',
#             'knowledgeBaseConfiguration': {
#                 'knowledgeBaseId': kb_id,
#                 'modelArn': model_arn,
#                 'retrievalConfiguration': {
#                     'vectorSearchConfiguration': {
#                         'numberOfResults': 3
#                     }
#                 },
#             }
#         }
#     )
#     print(response)

# except botocore.exceptions.ClientError as e:
#     print(f"An error occurred: {e}")
        



# Calls the Bedrock client to get a response from Claude.
def invoke_claude(prompt, tokens=200):
    try:
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps({
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": tokens,
                "anthropic_version": "bedrock-2023-05-31"
            }),
            contentType='application/json'
        )
        response_body = json.loads(response['body'].read().decode('utf-8'))
        advice = response_body.get('content', [{}])[0].get('text', 'No advice available')
        return advice
    
    except botocore.exceptions.ClientError as e:
        print(f"An error occurred: {e}")
        return None

# Gets Stock Data from Yahoo Finance
def get_stock_data(symbol):
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

    return response

# Gets sentiment analysis about the stock
def get_sentiment_analysis(stock_data, stock_symbol):
    prompt = (
        f"Here is the stock data for {stock_symbol}:\n"
        f"Current Price: {stock_data['current_price']}\n"
        f"Previous Close: {stock_data['previous_close']}\n"
        f"Market Cap: {stock_data['market_cap']}\n"
        f"P/E Ratio: {stock_data['pe_ratio']}\n\n"
        f"Based on this data and any available news, analyze the sentiment around {stock_symbol} and summarize."
    )
    return invoke_claude(prompt)

# Gets industry analysis about the stock
def get_industry_analysis(stock_data, stock_symbol):
    stock = yf.Ticker(stock_symbol)
    industry = stock.info.get('industry', 'N/A')
    sector = stock.info.get('sector', 'N/A')
    
    prompt = (
        f"Analyze the {industry} industry and {sector} sector for {stock_symbol}. "
        f"Consider trends, growth prospects, regulatory changes, and the competitive landscape."
    )
    return invoke_claude(prompt)

# Gets analyst ratings about the stock
def get_analyst_ratings(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    recommendations = stock.recommendations
    
    if recommendations is None or recommendations.empty:
        return "No analyst ratings available."
    
    latest_rating = recommendations.iloc[-1]
    
    # Extract relevant fields from the latest rating
    period = latest_rating.get('period', 'N/A')
    strongBuy = latest_rating.get('strongBuy', 'N/A')
    buy = latest_rating.get('buy', 'N/A')
    hold = latest_rating.get('hold', 'N/A')
    sell = latest_rating.get('sell', 'N/A')
    strongSell = latest_rating.get('strongSell', 'N/A')
    
    prompt = (
        f"Latest analyst rating for {stock_symbol}:\n"
        f"Period: {period}\n"
        f"Strong Buy: {strongBuy}\n"
        f"Buy: {buy}\n"
        f"Hold: {hold}\n"
        f"Sell: {sell}\n"
        f"Strong Sell: {strongSell}\n\n"
        f"Summarize the impact of these ratings on the investment outlook for {stock_symbol}."
    )
    
    return invoke_claude(prompt)

# Provides an investment analysis based on various factors
def get_final_analysis(stock_data, sentiment_analysis, industry_analysis, analyst_ratings, stock_symbol):
    prompt = (
        f"Ticker: {stock_symbol}\n\n"
        f"Sentiment Analysis:\n{sentiment_analysis}\n\n"
        f"Industry Analysis:\n{industry_analysis}\n\n"
        f"Analyst Ratings:\n{analyst_ratings}\n\n"
        f"Stock Data:\n"
        f"Current Price: {stock_data['current_price']}\n"
        f"Previous Close: {stock_data['previous_close']}\n"
        f"Market Cap: {stock_data['market_cap']}\n"
        f"P/E Ratio: {stock_data['pe_ratio']}\n\n"
        f"Based on the provided data and analyses, provide a comprehensive investment recommendation for {stock_symbol}. "
        f"Consider the company's financial strength, growth prospects, competitive position, and potential risks. "
        f"Provide a clear and concise recommendation on whether to buy, hold, or sell the stock, along with supporting rationale."
    )
    return invoke_claude(prompt, tokens=300)

# Gives investment advice for a particular stock given the stocks historical performance within the past year
def get_investing_advice_for(stock_symbol, analysis_walkthrough=False):

    stock_data = get_stock_data(stock_symbol)

    sentiment_analysis = get_sentiment_analysis(stock_data, stock_symbol)
    industry_analysis = get_industry_analysis(stock_data, stock_symbol)
    analyst_ratings = get_analyst_ratings(stock_symbol)

    # print("Sentiment Analysis:", sentiment_analysis)
    # print("Industry Analysis:", industry_analysis)
    # print("Analyst Ratings:", analyst_ratings)

    final_analysis = get_final_analysis(stock_data, sentiment_analysis, industry_analysis, analyst_ratings, stock_symbol)

    if analysis_walkthrough:
        return f"Investment Walkthrough: \n\n Sentiment Analysis: \n {sentiment_analysis} \n\n Industry Analysis: \n {industry_analysis} \n\n Analyst Ratings: \n{analyst_ratings} \n\n Final Analysis: \n{final_analysis}"
    return final_analysis

test_ticker = "PSA"
# print(get_investing_advice_for(test_ticker, True))
# print(get_analyst_ratings(test_ticker))