# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 11:26:21 2024

@author: sm22alb
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_samples
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt



def read_data(file_path):
    """Read data from a CSV file and return a cleaned DataFrame."""
    data = pd.read_csv(file_path, skiprows=4)
    return data

def read_food_exports(file_path):
    """Read food exports data from a CSV file and return a cleaned DataFrame."""
    food_exports = pd.read_csv(file_path, skiprows=4)
    return food_exports
def preprocess_data(data, country_name):
    """Preprocess data, filter relevant columns, handle missing values."""
    filtered_data = data[data['Country Name'] == country_name]
    selected_data = filtered_data[['Country Name'] + [str(year) for year in range(1980, 2022)]]
    selected_data = selected_data.dropna()
    return selected_data

def preprocess_food_exports(food_exports, country_name):
    """Preprocess food exports, filter relevant columns, handle missing values."""
    filtered_data = food_exports[food_exports['Country Name'] == country_name]
    selected_data = filtered_data[['Country Name'] + [str(year) for year in range(1980, 2022)]]
    selected_data = selected_data.dropna()
    return selected_data

def transpose_arable_land(data):
    """Transpose the DataFrame and reset the index."""
    transposed_data = data.transpose().reset_index()
    transposed_data['Country Name'] = data['Country Name'].values[0]
    transposed_data.columns = ['Year', 'arable_land', 'Country Name'] + list(transposed_data.iloc[0, 3:])
    transposed_data = transposed_data[1:].reset_index(drop=True)
    return transposed_data

def transpose_food_exports(data):
    """Transpose the DataFrame and reset the index."""
    transposed_data = data.transpose().reset_index()
    transposed_data['Country Name'] = data['Country Name'].values[0]
    transposed_data.columns = ['Year', 'food_exports', 'Country Name'] + list(transposed_data.iloc[0, 3:])
    transposed_data = transposed_data[1:].reset_index(drop=True)
    return transposed_data

def merge_transposed_data(transposed_arable_land, transposed_food_exports):
    """Merge transposed arable land and transposed food exports based on 'Country Name' and 'Year'."""
    merged_data = pd.merge(transposed_arable_land, transposed_food_exports, on=['Country Name', 'Year'])
    return merged_data

def print_silhouette_scores(data, max_clusters=10):
    """Print silhouette scores for different numbers of clusters."""
    features = data[['arable_land', 'food_exports']]

    for n_clusters in range(2, max_clusters + 1):
        km = KMeans(n_clusters=n_clusters)
        data['Cluster'] = km.fit_predict(features)

        # Calculate average silhouette score
        silhouette_avg = silhouette_samples(features, data['Cluster']).mean()
        print(f"For n_clusters = {n_clusters}, the average silhouette score is: {silhouette_avg}")

def apply_kmeans(data, n_clusters):
    """Apply K-Means clustering and scale relevant columns."""
    km = KMeans(n_clusters=n_clusters)
   
    # Include only 'Arable land' and 'Food exports' for clustering
    features = data[['arable_land', 'food_exports']]
    data['Cluster'] = km.fit_predict(features)

    # Scale the relevant columns
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)
   
    # Print the scaled cluster centers
    scaled_centers = scaler.transform(km.cluster_centers_)
    print("Scaled Cluster Centers:")
    for i, center in enumerate(scaled_centers):
        print(f"Cluster {i+1}: {center}")

    # Update the relevant columns with scaled values
    data[['arable_land', 'food_exports']] = scaled_data

    return data


def separate_clusters(data):
    """Separate clusters based on the 'Cluster' column."""
    clusters = [data[data['Cluster'] == i] for i in range(data['Cluster'].nunique())]
    return clusters

def plot_clusters(data, clusters):
    """Plot clusters with different colors and center values."""
    colors = ['#00FF00', 'red', 'blue']
    plt.figure(figsize=(8, 6))
    for i, cluster in enumerate(clusters):
        plt.scatter(cluster['food_exports'], cluster['arable_land'], color=colors[i], label=f'Cluster {i+1}')
   
    # Plot center values
    centers = data.groupby('Cluster').mean()[['food_exports', 'arable_land']]
    plt.scatter(centers['food_exports'], centers['arable_land'], marker='d', color='black', alpha=0.8, label='Cluster Centers')
   
    plt.xlabel('Food Exports')
    plt.ylabel('Arable Land')
    plt.legend()
    plt.show()
    
def plot_line_graph(data, variable_name, title, x_interval=1):
    plt.figure(figsize=(10, 6))
    plt.plot(data['Year'], data[variable_name])

    plt.xlabel('Year')
    plt.ylabel(variable_name)
    plt.title(title)

    # Set x-axis interval
    plt.xticks(data['Year'][::x_interval])

    plt.show()

def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    t = t - 1990
    f = n0 * np.exp(g * t)
    return f
    

def plot_line(data, variable_name, title, x_interval=3):
    # Convert the 'Year' column to numeric
    data['Year'] = pd.to_numeric(data['Year'])

    plt.figure(figsize=(10, 6))
    plt.plot(data['Year'], data[variable_name], label=variable_name)

    # Fit exponential curve with initial parameter guesses
    param_guess = [data[variable_name].min(), 0.01]  # Adjust initial guesses as needed
    param, covar = opt.curve_fit(exponential, data['Year'], data[variable_name], p0=param_guess, maxfev=10000)

    # Calculate uncertainty range (confidence interval) up to the end of the forecast
    year_forecast = np.linspace(data['Year'].min(), 2030, 100)
    param_std_dev = np.sqrt(np.diag(covar))
    lower_bound = exponential(year_forecast, *(param - param_std_dev))
    upper_bound = exponential(year_forecast, *(param + param_std_dev))

    # Plot the uncertainty range with a shaded region
    plt.fill_between(year_forecast, lower_bound, upper_bound, color='yellow', alpha=0.3)

    # Plot the exponential fit with a different color and no legend
    plt.plot(data['Year'], exponential(data['Year'], *param), color='blue', label='_nolegend_')

    # Create array for forecasting until 2030
    year_forecast = np.linspace(data['Year'].min(), 2030, 100)
    forecast_exp = exponential(year_forecast, *param)

    # Plot the exponential forecast until 2030
    plt.plot(year_forecast, forecast_exp, label="Exponential Forecast")

    plt.xlabel("Year")
    plt.ylabel(variable_name)
    plt.title(title)
   
    # Set x-axis interval
    plt.xticks(np.arange(data['Year'].min(), 2031, x_interval))  # Ensure x-axis range includes 2030

    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    # Example usage
    file_path = 'arable_land.csv'
    food_exports_file_path = 'food_exports.csv'  # Replace with the actual file path
    country_name = 'European Union'
   
    data = read_data(file_path)
    selected_data = preprocess_data(data, country_name)

    # Read and preprocess food exports data
    food_exports_data = read_food_exports(food_exports_file_path)  # Corrected variable name
    food_exports_selected_data = preprocess_food_exports(food_exports_data, country_name)
    transposed_arable_land = transpose_arable_land(selected_data)
    transposed_food_exports = transpose_food_exports(food_exports_selected_data)
    merged_data = merge_transposed_data(transposed_arable_land, transposed_food_exports)
    print_silhouette_scores(merged_data, max_clusters=10)

   
    n_clusters = 3
    clustered_data = apply_kmeans(merged_data,n_clusters)
    clusters = separate_clusters(clustered_data)
    plot_clusters(clustered_data, clusters)
    
    plot_line(transposed_food_exports, 'food_exports', 'Food Exports Forecast Over Years', x_interval=3)
    plot_line(transposed_arable_land, 'arable_land', 'ARABLE lAND OVER YEARS', x_interval=3)
    
    plot_line_graph(transposed_arable_land, 'arable_land', 'ARABLE lAND OVER YEARS', x_interval=2)
    plot_line_graph(transposed_food_exports, 'food_exports', 'FOOD EXPORTS OVER YEARS', x_interval=2)

