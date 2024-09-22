# Flu Prediction Using Deep Learning

This project aims to predict flu cases based on various factors using a deep learning model that combines Graph Convolutional Networks (GCN) and Recurrent Neural Networks (RNN).

## Project Structure

- `generate_flu_data.py`: Contains functions to generate a synthetic dataset of flu cases.
- `flu_prediction_model.py`: Implements the advanced deep learning model for flu prediction.
- `flu_data.csv`: The generated dataset used for training and testing the model.

## Dataset Generation

The dataset is generated with the following features:
- **Temperature**: Average temperature in degrees Celsius.
- **Humidity**: Average humidity percentage.
- **Precipitation**: Average precipitation in mm.
- **Vaccination Rate**: Percentage of vaccinated individuals.
- **Population Density**: Number of individuals per square kilometer.
- **Social Distancing Measures**: Boolean indicating if social distancing measures are in place.
- **Previous Flu Cases**: Number of flu cases in the previous year.
- **Age Distribution**: Distribution of the population across different age groups.

The target variable is **Flu Cases**, calculated based on a weighted combination of the above features.

## Model Architecture

The model utilizes:
- **Graph Convolutional Networks (GCN)**: To capture relationships between different features.
- **Long Short-Term Memory (LSTM)**: To account for temporal dependencies in the data.

## Research Papers

This model draws on concepts from the following research papers:

1. [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
2. [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
3. [A Comprehensive Review on Deep Learning in Medical Imaging](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7349309/)

## Getting Started

To run the project, you need to have Python 3.x installed along with the required libraries. You can install the necessary libraries using pip:

```bash
pip install -r requirements.txt
```

