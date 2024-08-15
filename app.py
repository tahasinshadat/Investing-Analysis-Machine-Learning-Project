import streamlit as st
import investing_analysis_model_final as investing_model
import time
import random

def run_app():
    st.set_page_config(page_title="Investment Analysis Tool", layout="wide")
    
    st.title("📈 Investment Analysis Tool")
    st.markdown("Welcome to the Investment Analysis Tool. Please enter a stock ticker to begin your analysis.")

    # Sidebar with additional resources and current knowledge base
    st.sidebar.title("📚 Resources")
    st.sidebar.write("## Current Knowledge Base")
    knowledge_base_stocks = ["PSA", "WELL", "EQIX", "SPG", "PLD"] # investing_model.knowledge_bases.keys()
    for stock in knowledge_base_stocks:
        st.sidebar.write(f"- {stock}")

    st.sidebar.write("## Additional Resources")
    st.sidebar.write("[Investing Basics](https://www.investopedia.com/investing-4427788)")
    st.sidebar.write("[Stock Market News](https://www.cnbc.com/markets/)")
    st.sidebar.write("[Investment Strategies](https://www.fidelity.com/learning-center/investment-products/mutual-funds/investment-strategies)")

    # File upload section 
    st.sidebar.write("## Upload Ticker Files for Knowledge Base")
    uploaded_file = st.sidebar.file_uploader("Upload 10Q, Quarterly Valuations, etc.", type=["pdf", "txt", "csv", "xlsx"])
    if uploaded_file:
        st.sidebar.write(f"Uploaded: {uploaded_file.name}")
        # Upload to s3
    ticker_input = st.sidebar.text_input("Ticker for Uploaded Files")


    ticker = st.text_input("Enter the stock ticker:", "")

    if st.button("Get Investing Advice"):
        if ticker:

            stock_data = investing_model.get_stock_data(ticker)
            investing_model.plot_stock_data(stock_data['historical_data'], ticker)
            st.image(f"charts/{ticker}_stock_history.png", caption=f"{ticker} 1-Year Stock History", use_column_width=True)

            loading_placeholder = st.empty()
            loading_messages = [
                "🔍 Analyzing news outlets...",
                "📊 Importing historical stock data...",
                "💬 Performing sentiment analysis...",
                "🏭 Analyzing industry trends...",
                "📑 Extracting insights from Q10 documents...",
                "💹 Conducting quarterly valuation analysis...",
                "📊 Measuring stock volatility...",
                ""
            ]

            for i in range(len(loading_messages)):
                loading_placeholder.text(loading_messages[i % len(loading_messages)])
                time.sleep(random.randint(25, 50) / 100)
            
            with st.spinner('📈 Generating final investment advice...'):
                mathematical_analysis = investing_model.perform_mathematical_analysis(stock_data['historical_data'])
                stock_data['historical_data'] = stock_data['historical_data'].to_dict()
                sentiment_analysis = investing_model.get_sentiment_analysis(stock_data, ticker)
                industry_analysis = investing_model.get_industry_analysis(stock_data, ticker)
                analyst_ratings = investing_model.get_analyst_ratings(ticker)
                q10_analysis = investing_model.get_10Q_analysis(ticker)
                quarterly_valuation_analysis = investing_model.get_quarterly_valuation_analysis(ticker)
                market_trends_analysis = investing_model.get_market_trends_analysis(ticker)

                final_analysis = investing_model.get_final_analysis(
                    ticker, 
                    investing_model.synthesize_data_and_analyses(
                        structured_data=(stock_data, sentiment_analysis, industry_analysis, analyst_ratings), 
                        unstructured_data=(q10_analysis, quarterly_valuation_analysis, market_trends_analysis)
                    ),
                    mathematical_analysis
                )

                st.subheader("💡 Holistic Investment Advice")
                st.write(final_analysis)
                
                st.session_state['final_analysis'] = final_analysis
                st.session_state['stock_data'] = stock_data

    if 'final_analysis' in st.session_state:
        st.subheader("🔍 Further Analysis Options")
        tabs = st.tabs(["Sentiment", "Industry", "Analyst Ratings", "Q10", "Quarterly Valuation", "Market Trends", "Mathematical"])
        
        with tabs[0]:
            st.write("### Sentiment Analysis")
            st.write(sentiment_analysis)
        with tabs[1]:
            st.write("### Industry Analysis")
            st.write(industry_analysis)
        with tabs[2]:
            st.write("### Analyst Ratings Analysis")
            st.write(analyst_ratings)
        with tabs[3]:
            st.write("### Q10 Analysis")
            st.write(q10_analysis)
        with tabs[4]:
            st.write("### Quarterly Valuation Analysis")
            st.write(quarterly_valuation_analysis)
        with tabs[5]:
            st.write("### Market Trends Analysis")
            st.write(market_trends_analysis)
        with tabs[6]:
            st.write("### Mathematical Analysis")
            st.write(mathematical_analysis)

run_app()
