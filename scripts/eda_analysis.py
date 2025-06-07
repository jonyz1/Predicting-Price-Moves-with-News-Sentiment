import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

def load_data():
    """Load the news data from CSV file"""
    try:
        df = pd.read_csv('./raw_analyst_ratings.csv')
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def analyze_text_lengths(df):
    """Analyze text lengths in headlines and content"""
    # Calculate headline lengths
    df['headline_length'] = df['headline'].str.len()

    # Basic statistics for headline lengths
    headline_stats = df['headline_length'].describe()

    # Create histogram of headline lengths
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='headline_length', bins=50)
    plt.title('Distribution of Headline Lengths')
    plt.xlabel('Headline Length (characters)')
    plt.ylabel('Count')
    plt.savefig('notebooks/headline_length_distribution.png')
    plt.close()

    return headline_stats

def analyze_publishers(df):
    """Analyze publisher distribution"""
    # Count articles per publisher
    publisher_counts = df['publisher'].value_counts()

    # Plot top 10 publishers
    plt.figure(figsize=(12, 6))
    publisher_counts.head(10).plot(kind='bar')
    plt.title('Top 10 Publishers by Article Count')
    plt.xlabel('Publisher')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('notebooks/top_publishers.png')
    plt.close()

    return publisher_counts

def analyze_publication_dates(df):
    """Analyze publication date patterns"""
    # Convert publication date to datetime
    df['publication_date'] = pd.to_datetime(df['date'])

    # Extract date components
    df['year'] = df['publication_date'].dt.year
    df['month'] = df['publication_date'].dt.month
    df['day_of_week'] = df['publication_date'].dt.day_name()

    # Articles per year
    yearly_counts = df['year'].value_counts().sort_index()

    # Articles per day of week
    daily_counts = df['day_of_week'].value_counts()

    # Plot yearly trend
    plt.figure(figsize=(12, 6))
    yearly_counts.plot(kind='line', marker='o')
    plt.title('Number of Articles Published per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Articles')
    plt.grid(True)
    plt.savefig('notebooks/yearly_trend.png')
    plt.close()

    # Plot daily distribution
    plt.figure(figsize=(10, 6))
    daily_counts.plot(kind='bar')
    plt.title('Number of Articles Published by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('notebooks/daily_distribution.png')
    plt.close()

    return yearly_counts, daily_counts

def main():
    # Load data
    df = load_data()
    if df is None:
        return

    # Perform analyses
    print("\n=== Headline Length Statistics ===")
    headline_stats = analyze_text_lengths(df)
    print(headline_stats)

    print("\n=== Publisher Analysis ===")
    publisher_counts = analyze_publishers(df)
    print("\nTop 10 Publishers:")
    print(publisher_counts.head(10))

    print("\n=== Publication Date Analysis ===")
    yearly_counts, daily_counts = analyze_publication_dates(df)
    print("\nArticles per Year:")
    print(yearly_counts)
    print("\nArticles per Day of Week:")
    print(daily_counts)

if __name__ == "__main__":
    main()
