
# Credit Scoring Model Using Random Forest

This project focuses on developing a credit scoring model to predict the creditworthiness of individuals based on historical financial data. The dataset used for this model was originally intended for credit card fraud detection, but it has been adapted here to assess creditworthiness.

## Project Overview

In this project, we leverage a Random Forest classifier to predict whether a transaction is indicative of good or bad credit. While the dataset primarily contains anonymized credit card transaction data, it provides valuable insights into financial behaviors that can be useful in credit scoring.

### Key Features:
- **Dataset:** Contains 30 anonymized features (V1 to V28), transaction `Amount`, and a `Class` label (indicating fraud in the original context).
- **Model:** Random Forest Classifier.
- **Evaluation:** Includes advanced visualizations such as Precision-Recall curves, confusion matrices, and a heatmap of the classification report.
- **Accuracy:** The model's accuracy is presented as both a numerical score and a percentage for better interpretability.

## Steps Involved

1. **Data Preprocessing:**
   - Handling missing values.
   - Data normalization and scaling.
   - Splitting data into training and test sets.

2. **Model Training:**
   - Building and training the Random Forest model using the training data.
   - Tuning model parameters to improve performance.

3. **Model Evaluation:**
   - Generating predictions on the test set.
   - Evaluating the model using accuracy scores, confusion matrix, and advanced visualizations.
   - Plotting a heatmap of the classification report to visualize Precision, Recall, F1-Score, and Support.

4. **Visualization:**
   - Plotting the confusion matrix for a visual summary of the model's performance.
   - Creating a Precision-Recall curve to evaluate the trade-offs between precision and recall.
   - Generating a heatmap from the classification report to visualize key performance metrics.

## Results

- **Accuracy:** The model achieved an accuracy of `X.XX%` on the test data.
- **Visualization:** Detailed plots are included to provide insights into the model's performance across different metrics.

## Repository Structure

- `credit_scoring_model.ipynb`: The Jupyter notebook containing the entire workflow from data preprocessing to model evaluation.
- `data/`: Folder containing the dataset used for this project.
- `plots/`: Folder containing the visualizations generated during the model evaluation phase.
- `README.md`: This file, providing an overview and instructions for replicating the project.

## Dataset Link
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download


## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-scoring-model.git
   ```
2. Navigate to the project directory:
   ```bash
   cd credit-scoring-model
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Open the Jupyter notebook:
   ```bash
   jupyter notebook credit_scoring_model.ipynb
   ```
5. Run the cells sequentially to reproduce the results.

## Contributing

Feel free to fork the repository and submit pull requests. Contributions that improve the model, add new visualizations, or provide better interpretations are always welcome!

