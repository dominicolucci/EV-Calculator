# 🏀 NBA Sports Betting EV Prediction Platform

This project is a collection of tools designed to scrape NBA player game data, perform expected value (EV) analysis, and apply machine learning models (like Logistic Regression) to predict betting outcomes based on player performance.

## Project Purpose

- Scrape NBA player game logs from Basketball-Reference and NBA APIs.
- Analyze player performance under specific conditions (home/away games, back-to-back games, opponent strength).
- Train machine learning models to predict outcomes (e.g., over/under stat lines) and calculate expected value (EV).
- Assist in identifying profitable sports betting opportunities based on player data.

## 📂 Files and Their Functions

| File | Purpose/Description |
|:---|:---|
| **arbscanner.py** | Scrapes player game logs and calculates simple expected value (EV) metrics based on historical performance against betting lines. |
| **b2b_measurer.py** | Analyzes how players perform specifically during back-to-back games, which often impacts player stat lines. |
| **depthchart.py** | Scrapes recent NBA injury reports to account for player availability, focusing on both short-term and long-term injuries. |
| **EV_projector.py** | Full pipeline combining player data scraping, EV calculation, and a basic logistic regression model for predicting performance relative to betting lines. |
| **logistic_regression.py** | Trains a logistic regression model based on historical game data with optional features like home/away splits and opponent ratings. |
| **logistic_regression2.py** | An enhanced logistic regression model variant, including advanced feature handling (opponent offensive and defensive ratings). |
| **nba_api_scraper.py** | Alternative data scraper using the `nba_api` Python package to pull player game logs directly from NBA data sources instead of Basketball-Reference. |
| **NBA_Player_Map.py** | Static map between player names and their Basketball-Reference ID codes, allowing for easier scraping without needing repeated lookups. |
| **playernames.py** | Dynamic scraper that pulls the list of all current NBA players from Basketball-Reference per-game stats page. |

## 📦 Requirements

- Python 3.8+
- Libraries:
  - `requests`
  - `beautifulsoup4`
  - `pandas`
  - `scikit-learn`
  - `nba_api`
  - `numpy`

## ⚙️ Setup Instructions

1. Install required libraries:
   ```bash
   pip install -r requirements.txt
