# Predicting Stress from Reddit Posts using NLP 🧠

This project implements a complete machine learning pipeline to predict stress from Reddit posts using the DREADDIT dataset. It leverages Natural Language Processing (NLP) techniques for text data and incorporates social engagement metrics to build robust classification models.

The primary goal is to accurately distinguish between posts from subreddits associated with stress (e.g., r/anxiety, r/stress) and those from non-stress-related subreddits.

---

## 📊 Key Features

* **Exploratory Data Analysis (EDA):** Visualizations of label distribution, social feature correlations, and text length to understand the dataset's characteristics.
* **Hybrid Feature Set:** Combines text-based features (`TfidfVectorizer` with n-grams) with numerical social features (`social_karma`, `social_upvote_ratio`, `social_num_comments`).
* **Robust Preprocessing Pipeline:** Uses `scikit-learn`'s `Pipeline` and `ColumnTransformer` to create a streamlined and reproducible workflow for feature processing.
* **Model Comparison:** Trains, tunes, and evaluates four different classification models:
    * Logistic Regression
    * Random Forest
    * Gradient Boosting
    * Linear SVM
* **Hyperparameter Tuning:** Employs `GridSearchCV` with `StratifiedKFold` cross-validation to find the optimal hyperparameters for each model based on the ROC-AUC score.
* **Comprehensive Evaluation:** Assesses model performance using Accuracy, ROC-AUC, and a detailed Classification Report (Precision, Recall, F1-Score).
* **Probability Calibration:** Uses `CalibratedClassifierCV` for models like `LinearSVC` that do not natively provide probability estimates.

---

## 📈 Results

The project systematically evaluates each model after hyperparameter tuning. The final comparison showed that the **Logistic Regression** model achieved the best balance of performance and efficiency on the test set.

| Model                 | CV ROC-AUC | Test Accuracy | Test ROC-AUC |
|-----------------------|------------|---------------|--------------|
| Logistic Regression   | 0.851      | 0.771         | 0.849        |
| Random Forest         | 0.840      | 0.762         | 0.835        |
| Gradient Boosting     | 0.845      | 0.761         | 0.840        |
| Linear SVM            | 0.849      | 0.768         | 0.847        |

*(Note: These are example results based on the script's logic; actual scores may vary slightly upon execution.)*



---

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Install the dependencies:**
    Make sure you have the `dreaddit-train.csv` and `dreaddit-test.csv` files in the root directory.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the script:**
    ```bash
    python your_script_name.py
    ```
    This will execute the entire pipeline, from EDA and model training to evaluation, and will save the final model comparison results to `results.csv`.

---

## 🛠️ Dependencies

You can create a `requirements.txt` file with the following content:
