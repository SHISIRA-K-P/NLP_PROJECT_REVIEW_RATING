# Automated Review Rating System

## Project Description

Automated Review Rating System

This project uses a dataset of fine-food reviews collected from Amazon, containing approximately 500,000 reviews spanning more than 10 years, up to October 2012. The dataset includes review text, ratings, and product and user information. The goal of this project is to analyze the review text and automatically predict the corresponding rating.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn

## Project Structure

- data - Dataset files
- notebooks - Jupyter notebooks
- models - Machine learning models
- app -
- frontend - Frontend application

## Setup

Create virtual environment:

python -m venv venv

Activate:

venv\Scripts\Activate

Install dependencies:

pip install -r requirements.txt


## dataset
https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews?select=Reviews.csv


Dataset → Basic Cleaning → Duplicate/Conflict Handling → Rating Distribution → Advanced NLP Preprocessing → Lemmatization → Short/Long Review Filtering → Final Dataset → ML Rating Prediction.