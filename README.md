# Predicting Price Moves with News Sentiment

This project analyzes the correlation between financial news sentiment and stock market movements to enhance predictive analytics capabilities for Nova Financial Solutions.

## Project Overview

The project focuses on two main objectives:
1. **Sentiment Analysis**: Performing sentiment analysis on financial news headlines to quantify the tone and sentiment expressed.
2. **Correlation Analysis**: Establishing statistical correlations between news sentiment and corresponding stock price movements.

## Project Structure

```
├── .vscode/              # VS Code settings
├── .github/             # GitHub workflows and CI/CD
├── src/                 # Source code
├── notebooks/           # Jupyter notebooks for analysis
├── tests/              # Unit tests
└── scripts/            # Utility scripts
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data

The project uses the Financial News and Stock Price Integration Dataset (FNSPID), which includes:
- Article headlines
- Publication dates
- Stock symbols
- Article URLs
- Publisher information
