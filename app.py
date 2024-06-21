from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
import json
import requests
from bs4 import BeautifulSoup
from plotly.utils import PlotlyJSONEncoder
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-scores', methods=['POST'])
def get_scores():
    try:
        data = request.get_json()
        profile_url = data['profileUrl']
        rankings_url = f"{profile_url}/rankings/"

        response = requests.get(rankings_url)
        response.raise_for_status()  # Raise HTTPError for bad responses

        soup = BeautifulSoup(response.content, 'html.parser')

        # Scrape the user's name
        user_name_tag = soup.select_one('h1.c-toolbar__title.u-flex.u-flex-align-items-center span.u-mr-')
        if not user_name_tag:
            raise ValueError("User name not found")
        user_name = user_name_tag.get_text(strip=True)

        seasons = []
        scores_array = []

        for item in soup.select('.c-card__item'):
            season = item.select_one('h2.c-title').get_text(strip=True)
            if season:
                scores_for_season = [
                    float(a.get_text(strip=True))
                    for a in item.select('.o-grid--cols-3\\@md.o-grid--gap-10 .c-link-boxes h3.c-link-boxes__header.u-flex.u-flex-justify-between a')
                    if a.get_text(strip=True).replace('.', '', 1).isdigit()
                ]
                if scores_for_season:
                    average_score = sum(scores_for_season) / len(scores_for_season)
                    seasons.append(season)
                    scores_array.append(average_score)

        if not seasons or not scores_array:
            raise ValueError("No valid data found")

        return jsonify({'seasons': seasons, 'scoresArray': scores_array, 'userName': user_name})
    except Exception as e:
        print(e)
        return jsonify({'error': 'Failed to fetch scores. Please enter a valid URL.'}), 400

@app.route('/process-data', methods=['POST'])
def process_data():
    try:
        data = request.get_json()
        seasons = data['seasons']
        scores = data['scoresArray']
        user_name = data['userName']

        # Convert seasons to numerical values for fitting the model
        years = np.array([int(season.split('-')[1]) for season in seasons]).reshape(-1, 1)

        # Polynomial Regression
        poly = PolynomialFeatures(degree=2)  # Adjust the degree if needed
        X_poly = poly.fit_transform(years)
        model = LinearRegression()
        model.fit(X_poly, scores)

        # Predict the next two seasons' scores based on the current year
        current_year = datetime.now().year
        future_years = np.array([current_year + 1, current_year + 2]).reshape(-1, 1)
        future_years_poly = poly.transform(future_years)
        predictions = model.predict(future_years_poly)

        # Prepare data for plotting
        all_years = np.concatenate([years, future_years])
        all_scores = np.concatenate([scores, predictions])

        # Generate a smooth curve for the polynomial fit
        year_range = np.linspace(years.min(), future_years.max(), 300).reshape(-1, 1)
        year_range_poly = poly.transform(year_range)
        predicted_range = model.predict(year_range_poly)

        # Create the plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[int(season.split('-')[1]) for season in seasons], y=scores, mode='markers', name='Actual scores'))
        fig.add_trace(go.Scatter(x=year_range.flatten(), y=predicted_range, mode='lines', name='Polynomial fit'))
        fig.add_trace(go.Scatter(x=future_years.flatten(), y=predictions, mode='markers+text', text=[f'{pred:.2f}' for pred in predictions], textposition='top center', name='Predicted scores'))

        fig.update_layout(
            title=f"{user_name}'s SwimCloud Points Over Seasons",
            xaxis_title='Year',
            yaxis_title='Points',
            margin=dict(t=40, b=40)
        )

        graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)

        return jsonify(graph_json)
    except Exception as e:
        print(e)
        return jsonify({'error': 'Failed to process data.'}), 400

if __name__ == '__main__':
    app.run(port=5000)

    