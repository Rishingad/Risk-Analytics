import yfinance as yf
import pandas as pd
import time
import os
from datetime import datetime

ASSETS = ["^GSPC", "^IXIC", "AAPL", "MSFT", "JPM"]
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"
MAX_RETRIES = 3
SLEEP_TIME = 2

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(ROOT_DIR, "data")
LOG_FILE = os.path.join(ROOT_DIR, "download_log.txt")


os.makedirs(DATA_DIR, exist_ok=True)

def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {message}\n")

# ==============================
# DOWNLOAD FUNCTION
# ==============================
def download_ticker(ticker):
    attempts = 0
    
    while attempts < MAX_RETRIES:
        try:
            df = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                auto_adjust=True,   # prices already adjusted; "Adj Close" col removed
                progress=False
            )
            
            if df.empty:
                raise ValueError("Empty dataset returned")
            
            print(f"✅ {ticker} downloaded ({len(df)} rows)")
            log_message(f"{ticker} SUCCESS ({len(df)} rows)")
            
            # auto_adjust=True → adjusted prices are under "Close", not "Adj Close"
            return df["Close"].squeeze()
        
        except Exception as e:
            attempts += 1
            print(f"⚠️ Retry {attempts} for {ticker} | Error: {e}")
            log_message(f"{ticker} FAIL attempt {attempts}: {e}")
            time.sleep(SLEEP_TIME)
    
    print(f"❌ Failed to download {ticker} after {MAX_RETRIES} attempts")
    log_message(f"{ticker} FINAL FAILURE")
    return None

# ==============================
# MAIN PIPELINE
# ==============================
def run_pipeline():
    all_series = {}
    
    print("🚀 Starting data download...\n")
    
    for ticker in ASSETS:
        series = download_ticker(ticker)
        
        if series is not None:
            all_series[ticker] = series
        
        time.sleep(1)  # avoid rate limiting
    
    if not all_series:
        raise RuntimeError("No data downloaded. Check logs.")
    
    # Combine all tickers
    prices = pd.DataFrame(all_series)
    
    # Drop rows with missing values
    prices = prices.dropna()
    
    # Save final dataset
    output_path = os.path.join(DATA_DIR, "prices.csv")
    prices.to_csv(output_path)
    
    print("\n📁 Data saved to:", output_path)
    print("📊 Final shape:", prices.shape)
    
    log_message(f"FINAL DATASET SHAPE: {prices.shape}")
    
    return prices

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    df = run_pipeline()
    print("\n🎉 Pipeline completed successfully!")
