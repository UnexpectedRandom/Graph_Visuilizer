# Flu Prediction Using Deep Learning

This project aims to predict flu cases based on various factors using a deep learning model that combines Graph Convolutional Networks (GCN) and Recurrent Neural Networks (RNN).

## Project Structure

- `generate_flu_data.py`: Contains functions to generate a synthetic dataset of flu cases.
- `flu_prediction_model.py`: Once your run the `generate_flu_data.py` it implements the advanced deep learning model for flu prediction.
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

After installing the dependencies, run the data generation script to create the dataset:
python generate_flu_data.py

Then, you can train the model using the generated dataset in flu_data.csv.


### MIT License

For the license file, you can create a `LICENSE` file with the following content:

```plaintext
MIT License

Copyright (c) 2024 UnexpectedRandom

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

1. The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

2. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
```

