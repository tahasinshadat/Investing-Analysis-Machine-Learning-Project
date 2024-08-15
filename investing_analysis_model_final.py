#%% import packages
import boto3
import botocore
import os
import json
import base64
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
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

knowledge_base_id = 'KDW4G5S26S'

model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
model_arn = f'arn:aws:bedrock:us-east-1::foundation-model/{model_id}'

bedrock_client = session.client(service_name='bedrock-runtime', region_name='us-east-1')
bedrock_agent_runtime_client = session.client("bedrock-agent-runtime", region_name='us-east-1')


"""
CLAUDE INVOKING FUNCTIONS
"""
# Calls the Bedrock client to get a response from Claude with multimodal input
def invoke_multimodal_claude(prompt, image_path, tokens=1024):
    # Convert image to base64
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the request payload
    request_payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": encoded_image
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    }
    
    try:
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_payload),
            contentType='application/json'
        )
        response_body = json.loads(response['body'].read().decode('utf-8'))
        # Extract the text response from Claude
        advice = response_body.get('content', [{}])[0].get('text', 'No advice available')
        return advice
    
    except botocore.exceptions.ClientError as e:
        print(f"An error occurred: {e}")
        return None


# Calls the Bedrock client to get a response from Claude using RAG with unstructured Data 
def invoke_claude_with_RAG(prompt):

    # Parses Response from said function to make it more readable
    def parse_claude_response(response):
        parsed_response = ""
        
        # Extract and format the main response text
        if 'output' in response and 'text' in response['output']:
            parsed_response += response['output']['text'] + "\n\n"
        
        # Extract and format citations and references
        if 'citations' in response:
            for citation in response['citations']:
                if 'generatedResponsePart' in citation and 'textResponsePart' in citation['generatedResponsePart']:
                    citation_text = citation['generatedResponsePart']['textResponsePart']['text']
                    parsed_response += f"Excerpt: {citation_text}\n"
                    
                    if 'retrievedReferences' in citation:
                        for reference in citation['retrievedReferences']:
                            if 'content' in reference and 'text' in reference['content']:
                                reference_text = reference['content']['text']
                                s3_uri = reference['location']['s3Location']['uri']
                                parsed_response += f"Reference: {reference_text}\nS3 URI: {s3_uri}\n\n"
        
        return parsed_response
    
    # if metadata_filter is None:
    try:
        response = bedrock_agent_runtime_client.retrieve_and_generate(
            input={
                'text': prompt
            },
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': knowledge_base_id,
                    'modelArn': model_arn,
                    'retrievalConfiguration': {
                        'vectorSearchConfiguration': {
                            'numberOfResults': 3
                        }
                    },
                }
            }
        )

        return parse_claude_response(response)

    except botocore.exceptions.ClientError as e:
        print(f"An error occurred: {e}")
        return None

# Calls the Bedrock client to get a response from Claude
def invoke_claude(prompt, tokens=250):
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



"""
STRUCTURED ANALYSIS + STRUCTURED DATA FUNCTIONS
"""
# Gets Stock Data from Yahoo Finance
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)  # Fetch the stock data using yfinance
    info = stock.info  # Get current stock information
    history = stock.history(period="1y", timeout=60)  # Get historical data
    
    # Prepare the response
    response = {
        "current_price": info.get("currentPrice", "N/A"),
        "previous_close": info.get("regularMarketPreviousClose", "N/A"),
        "market_cap": info.get("marketCap", "N/A"),
        "pe_ratio": info.get("forwardEps", "N/A"),
        "historical_data": history # keep as dataframe for matplotlib charting
    }

    return response

# Plots stock data
def plot_stock_data(history, ticker):
    plt.figure(figsize=(10, 6))
    plt.plot(history.index, history['Close'], label="Close Price")
    plt.title(f"{ticker} Stock Price - Last 1 Year")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"charts/{ticker}_stock_history.png", format="png", dpi=300) # Save the chart to a file that can be fed into Claude
    # plt.show()

# Gets sentiment analysis about the stock
def get_sentiment_analysis(stock_data, ticker):
    prompt = (
        f"Here is the stock data for {ticker}:\n"
        f"Current Price: {stock_data['current_price']}\n"
        f"Previous Close: {stock_data['previous_close']}\n"
        f"Market Cap: {stock_data['market_cap']}\n"
        f"P/E Ratio: {stock_data['pe_ratio']}\n\n"
        f"Based on this data and any available news, analyze the sentiment around {ticker} and summarize."
    )
    return invoke_claude(prompt)

# Gets industry analysis about the stock
def get_industry_analysis(stock_data, ticker):
    stock = yf.Ticker(ticker)
    industry = stock.info.get('industry', 'N/A')
    sector = stock.info.get('sector', 'N/A')
    
    prompt = (
        f"Analyze the {industry} industry and {sector} sector for {ticker}. "
        f"Consider trends, growth prospects, regulatory changes, and the competitive landscape."
    )
    return invoke_claude(prompt)

# Gets analyst ratings about the stock
def get_analyst_ratings(ticker):
    stock = yf.Ticker(ticker)
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
        f"Latest analyst rating for {ticker}:\n"
        f"Period: {period}\n"
        f"Strong Buy: {strongBuy}\n"
        f"Buy: {buy}\n"
        f"Hold: {hold}\n"
        f"Sell: {sell}\n"
        f"Strong Sell: {strongSell}\n\n"
        f"Summarize the impact of these ratings on the investment outlook for {ticker}."
    )
    
    return invoke_claude(prompt)



"""
UNSTRUCTURED ANALYSIS FUNCTIONS
"""
# Analyze 10-Q reports using unstructured data
def get_10Q_analysis(ticker):
    prompt = f"Summarize and analyze the latest 10-Q reports for {ticker}. Focus on key financial metrics, risks, and opportunities."
    return invoke_claude_with_RAG(prompt)

# Analyze quarterly valuation reports using unstructured data
def get_quarterly_valuation_analysis(ticker):
    prompt = f"Analyze the quarterly valuation reports for {ticker}. Highlight trends in revenue, earnings, and growth prospects."
    return invoke_claude_with_RAG(prompt)

# Any other relevant unstructured analysis functions
def get_market_trends_analysis(ticker):
    prompt = f"Provide an in-depth analysis of any recent news, management discussions, and market trends for {ticker}."
    return invoke_claude_with_RAG(prompt)


"""
Own Mathwmatical analysis on stock history
"""
def perform_mathematical_analysis(history):

    if not isinstance(history, pd.DataFrame) or 'Close' not in history.columns:
        raise ValueError("Invalid history DataFrame")

    # Moving Averages
    history['MA_20'] = history['Close'].rolling(window=20).mean()
    history['MA_50'] = history['Close'].rolling(window=50).mean()

    # Volatility
    history['Daily_Return'] = history['Close'].pct_change() # Get daily returns
    volatility = history['Daily_Return'].std() * np.sqrt(252)  # Annualized volatility

    # Linear Regression for Trend Lines
    history = history.dropna()  # Remove NaN values generated by rolling window
    X = np.arange(len(history)).reshape(-1, 1)  # Days as feature
    y = history['Close'].values  # Prices as target
    
    model = LinearRegression()
    model.fit(X, y)
    model.predict(X)
    trend_slope = model.coef_[0]
    
    analysis_results = {
        "moving_averages": {
            "20-day": history['MA_20'].iloc[-1],  # Latest 20-day MA
            "50-day": history['MA_50'].iloc[-1]   # Latest 50-day MA
        },
        "volatility": volatility,
        "trend_line_slope": trend_slope,
        "latest_close": history['Close'].iloc[-1]
    }
    # print(analysis_results)
    return analysis_results



"""
SYNTHESIZE FUNCTIONS
"""
# Synthesizes structured and unstructured data for final analysis
def synthesize_data_and_analyses(structured_data, unstructured_data):
    stock_data, sentiment_analysis, industry_analysis, analyst_ratings = structured_data # Unpack structured data
    q10_analysis, quarterly_valuation_analysis, market_trends_analysis = unstructured_data # Unpack unstructured data
    
    # Combine insights from both structured and unstructured sources
    synthesized_insights = (
        f"Sentiment Analysis:\n{sentiment_analysis}\n\n"
        f"Industry Analysis:\n{industry_analysis}\n\n"
        f"Analyst Ratings:\n{analyst_ratings}\n\n"
        f"Stock Data:\n"
        f"Current Price: {stock_data['current_price']}\n"
        f"Previous Close: {stock_data['previous_close']}\n"
        f"Market Cap: {stock_data['market_cap']}\n"
        f"P/E Ratio: {stock_data['pe_ratio']}\n\n"
        f"10-Q Analysis:\n{q10_analysis}\n\n"
        f"Quarterly Valuation Analysis:\n{quarterly_valuation_analysis}\n\n"
        f"Market Trends Analysis:\n{market_trends_analysis}\n\n"
    )
    
    return synthesized_insights



"""

Provides Investment Analysis based on Various Factors:
    - Sentiment Analysis
        - News
        - Current Price
        - Previous Close
        - Market Cap
        - P/E Ratio
    - Industry Analysis
        - Growth Prospects
        - Regulatory Changes
        - Competitive Landscape
    - Analyst Ratings
        - Analyst Recommendations and Actions, whether they:
            - Strong Buy
            - Buy
            - Hold
            - Sell
            - Strong Sell
    - Q10 Analysis from specific company
    - Quarterly Valuation Analysis from specific company
    - Market Trends Analysis
    - Mathematical Analysis 
        - Linear Regression on trend lines
        - Daily Returns
        - Volatility
        - Moving Averages

"""
# Creates final output prompt
def get_final_analysis(ticker, synthesized_insights, math_analysis):
    prompt = (
        f"Ticker: {ticker}\n\n"
        f"Synthesized Insights:\n{synthesized_insights}\n\n"
        f"Mathematical Analysis:\n"
        f"Latest Close Price: {math_analysis['latest_close']}\n"
        f"20-day Moving Average: {math_analysis['moving_averages']['20-day']}\n"
        f"50-day Moving Average: {math_analysis['moving_averages']['50-day']}\n"
        f"Volatility (Annualized): {math_analysis['volatility']}\n"
        f"Trend Line Slope: {math_analysis['trend_line_slope']}\n\n"
        f"Based on the provided data, analyses, and stock history visualization provide a comprehensive investment recommendation for {ticker}. "
        f"Consider the company's financial strength, growth prospects, competitive position, and potential risks. "
        f"Provide a clear and concise recommendation on whether to buy, hold, or sell the stock, along with supporting rationale, including citations and/or references."
    )
    return invoke_multimodal_claude(prompt, image_path=f"charts/{ticker}_stock_history.png", tokens=1024)

# Ties all analyses, calculations, & informaion to be inputted into final output
def get_investing_advice_for(ticker, analysis_walkthrough=False):
    # Structured Data Analysis
    stock_data = get_stock_data(ticker)

    # Generate the stock plot image using the "histroical_data" data frame
    plot_stock_data(stock_data['historical_data'], ticker)

    # Calculate moving averages, volatility, daily returns, and trends
    mathematical_analysis = perform_mathematical_analysis(stock_data['historical_data'])

    stock_data['historical_data'] = stock_data['historical_data'].to_dict() # Convert 'historical_data' to a dictionary

    sentiment_analysis = get_sentiment_analysis(stock_data, ticker)
    industry_analysis = get_industry_analysis(stock_data, ticker)
    analyst_ratings = get_analyst_ratings(ticker)

    # Unstructured Data Analysis
    q10_analysis = get_10Q_analysis(ticker)
    quarterly_valuation_analysis = get_quarterly_valuation_analysis(ticker)
    market_trends_analysis = get_market_trends_analysis(ticker)

    # Generate the final analysis
    final_analysis = get_final_analysis(
        ticker, 
        synthesize_data_and_analyses(
            structured_data=(stock_data, sentiment_analysis, industry_analysis, analyst_ratings), 
            unstructured_data=(q10_analysis, quarterly_valuation_analysis, market_trends_analysis)
        ),
        mathematical_analysis
    )

    if analysis_walkthrough:
        section_break = "-" * 100
        return (
            f"Investment Walkthrough: \n\n"
            f"Sentiment Analysis: \n{sentiment_analysis} \n\n"
            f"{section_break}\n\n"
            f"Industry Analysis: \n{industry_analysis} \n\n"
            f"{section_break}\n\n"
            f"Analyst Ratings: \n{analyst_ratings} \n\n"
            f"{section_break}\n\n"
            f"10-Q Analysis: \n{q10_analysis} \n\n"
            f"{section_break}\n\n"
            f"Quarterly Valuation Analysis: \n{quarterly_valuation_analysis} \n\n"
            f"{section_break}\n\n"
            f"Market Trends Analysis: \n{market_trends_analysis} \n\n"
            f"{section_break}\n\n"
            f"Final Analysis: \n{final_analysis}"
        )
    
    return final_analysis


# test_ticker = "PSA"
# print(get_investing_advice_for(test_ticker, True))

# testing
# plot_stock_data(get_stock_data(test_ticker)['historical_data'], test_ticker)
# print(invoke_multimodal_claude("given the image, explain what it is", "PSA_stock_history.png"))