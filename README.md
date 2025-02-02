# Anime Ratings Prediction Project

This project leverages machine learning techniques to predict anime ratings based on various features extracted from an anime dataset. The workflow encompasses data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and evaluation using linear regression. The project showcases both statistical insights and practical ML model application, making it appealing to data scientists, machine learning engineers, and technical recruiters.

## Table of Contents

- [Dataset Description](#dataset-description)
- [Technical Stack](#technical-stack)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model Development](#model-development)
- [Model Evaluation](#model-evaluation)
- [Feature Selection](#feature-selection)
- [Results](#results)
- [Future Work](#future-work)

## Dataset Description

The dataset, `anime_data.csv`, consists of **12,101** entries and **44** features, encompassing metadata about anime titles, including:

- **Categorical Features**: `mediaType`, `sznOfRelease`, `studio_primary`, etc.
- **Numerical Features**: `eps` (episodes), `duration` (in minutes), `watched`, `votes`, etc.
- **Binary Tags**: Genre and thematic tags like `tag_Comedy`, `tag_Action`, `tag_Fantasy`, etc.
- **Target Variable**: `rating` (floating-point values ranging from **0.844** to **4.702**).

**Key Insights:**

- 4,468 missing descriptions and 4,636 missing durations.
- Ratings distribution is skewed towards higher ratings.

## Technical Stack

**Languages:**  
- Python

**Libraries:**  
- `pandas`, `numpy` for data manipulation  
- `matplotlib`, `seaborn` for data visualization  
- `scikit-learn` for ML modeling and evaluation  
- `mlxtend` for sequential feature selection

## Data Preprocessing

### Handling Missing Data

- Dropped features with extensive missing values (`description`, `years_running`).
- Removed rows with missing values, reducing the dataset to **7,465** entries.

### Encoding Categorical Variables

- Utilized `pd.get_dummies()` for one-hot encoding categorical columns like `mediaType`, `sznOfRelease`, and `studio_primary`.

### Feature Reduction

- Dropped irrelevant features (`title`, `description`) to focus on quantifiable predictors.

## Exploratory Data Analysis

### Univariate Analysis

- **Continuous Variables**: Visualized using histograms and boxplots to detect outliers and understand distributions.
  - *Example*: The average rating for anime with durations over 110 minutes is **3.76**, indicating longer anime tend to be better-rated.

- **Categorical Variables**: Count plots and percentage distributions were used.
  - *Example*: **Spring** is the most common release season for highly-rated anime.

### Correlation Analysis

- A heatmap was generated to understand feature relationships.
![Image](https://github.com/user-attachments/assets/4eae2746-d093-4e32-8fc8-e1cbeb1956ec)
- Observed strong correlations between `watched`, `votes`, and `rating`, indicating popularity influences ratings.

## Feature Engineering

- **Binary Tag Features**: Incorporated genre-specific tags (`tag_Comedy`, `tag_Action`, etc.) to enhance model granularity.

- **Derived Features**:
  - Created binary indicators for missing values in categorical data.
  - Combined features like `watched` and `votes` for popularity metrics.

## Model Development

### Model Choice

- **Linear Regression** was selected as the baseline model due to its interpretability and efficiency for continuous targets.

### Data Splitting

- **Training Set**: 70% (**5,225** entries)
- **Test Set**: 30% (**2,240** entries)

### Model Training

- The model was trained on the processed feature set with one-hot encoded variables.

## Model Evaluation

### Metrics Used

- **R-squared (R²)**: Measures the proportion of variance explained by the model.
- **Root Mean Squared Error (RMSE)**: Penalizes larger errors.
- **Mean Absolute Error (MAE)**: Provides an average magnitude of errors.

| Metric  | Training Data | Test Data |
|---------|---------------|-----------|
| **RMSE**  | 0.577         | 0.569     |
| **MAE**   | 0.467         | 0.464     |
| **R²**    | 0.521         | 0.515     |

**Interpretation:** The model explains approximately **51.5%** of the variance in ratings on unseen data, indicating a reasonable baseline performance.

## Feature Selection

To optimize performance and reduce model complexity, **Sequential Forward Selection (SFS)** from `mlxtend` was used.

### Approach

- Iteratively added features that maximized the **R²** score.
- Cross-validated with **5-fold** splits for robust evaluation.

## Results

- The optimal subset of **47 features** achieved an **R²** of **0.509**.

### Key Features Included:

- `watched`, `votes`, `duration`, `mediaType_TV`, and `studio_primary_Production I.G`.

### Visualization

- A performance curve was plotted to visualize gains in **R²** as features were added, showing diminishing returns after ~47 features.

### Influential Features

- **Popularity metrics** (`watched`, `votes`) and **studio reputation** (e.g., *Production I.G*, *Kyoto Animation*) significantly influence ratings.
- **Media type** and **release season** also contribute, with TV series and spring releases performing better.

### Genre Impact

- Anime tagged with **Drama**, **Adventure**, and **Fantasy** genres tend to receive higher ratings.

## Future Work

### Model Improvement

- Explore advanced models like **Random Forest Regression** or **Gradient Boosting** for better performance.
- Implement regularization techniques (**Ridge**, **Lasso**) to handle multicollinearity.

### Natural Language Processing (NLP)

- Incorporate text features from `description` using **TF-IDF** or **word embeddings** to capture narrative quality.

### Handling Missing Data

- Employ imputation techniques (e.g., **KNN Imputer**) to retain more data and improve model generalization.

### Model Deployment

- Develop a web application to recommend highly-rated anime using the trained model.

---

This project demonstrates the power of data-driven insights in understanding factors influencing anime ratings and provides a solid foundation for more sophisticated machine learning approaches.
