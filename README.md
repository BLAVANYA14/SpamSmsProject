# 📩 Spam SMS Detection using Machine Learning
🚀 Live Demo
👉 Click here to try the app:
https://blavanya14-spamsmsproject-spam-app-o3gmn9.streamlit.app/
## 📌 Project Overview

This project is a **Spam SMS Detection system** built using **Machine Learning and Natural Language Processing (NLP)**. The goal is to automatically classify SMS messages as **Spam** or **Ham (Not Spam)** based on their content.

The model uses **TF-IDF vectorization** for feature extraction and a **Multinomial Naïve Bayes classifier**, which is well-suited for text classification problems.

---

## 🛠️ Technologies & Libraries Used

* **Python**
* **Pandas & NumPy** – Data handling and numerical operations
* **NLTK** – Text preprocessing (stopwords removal, lemmatization)
* **Scikit-learn** – ML model, TF-IDF, train-test split, evaluation
* **Joblib** – Model and vectorizer persistence

---

## 📂 Dataset Description

* Dataset file: `spamraw.csv`
* Columns:

  * `text` → SMS message content
  * `type` → Label (`ham` or `spam`)

### Label Encoding

* `ham` → 0
* `spam` → 1

---

## 🧹 Text Preprocessing Steps

Each SMS message undergoes the following preprocessing steps:

1. Convert text to lowercase
2. Remove numbers
3. Remove punctuation
4. Remove extra spaces
5. Remove English stopwords
6. Apply **WordNet Lemmatization**

This cleaning process improves model accuracy by reducing noise in the text data.

---

## 🔍 Feature Extraction

* **TF-IDF Vectorizer** is used to convert text into numerical form
* Maximum features limited to **5000** for efficiency

---

## 🤖 Machine Learning Model

* **Algorithm**: Multinomial Naïve Bayes
* **Train-Test Split**: 80% training, 20% testing
* **Reason for choosing NB**:

  * Performs well on text classification
  * Fast and memory efficient
  * Handles high-dimensional sparse data effectively

---

## 📊 Model Evaluation

The model is evaluated using:

* **Accuracy Score**
* **Classification Report** (Precision, Recall, F1-score)

These metrics help assess how well the model distinguishes between spam and ham messages.

---

## 💾 Model Persistence

After training:

* Trained model saved as: `spam_sms_model.pkl`
* TF-IDF vectorizer saved as: `tfidf_vectorizer.pkl`

This allows reuse of the model without retraining.

---

## 🔮 SMS Prediction Functionality

The project includes a prediction function that:

1. Loads the saved model and vectorizer
2. Cleans the input SMS
3. Converts it into TF-IDF features
4. Predicts whether the SMS is **Spam** or **Ham**

### Example Predictions

* "Congratulations! You have won a free iPhone" → **Spam**
* "Hey, how are you?" → **Ham (Not Spam)**

---

## ▶️ How to Run the Project

1. Install required libraries:

   ```bash
   pip install pandas numpy scikit-learn nltk joblib
   ```
2. Place `spamraw.csv` in the project directory
3. Run the script:

   ```bash
   python app2.py
   ```
4. View accuracy, classification report, and predictions in the console

---

## 🚀 Future Improvements

* Use advanced models like **Logistic Regression, SVM, or LSTM**
* Add a **Flask / Streamlit web interface**
* Perform hyperparameter tuning
* Handle class imbalance more effectively

---

## 📌 Conclusion

This project demonstrates a complete **end-to-end ML pipeline** for spam SMS detection, including preprocessing, feature extraction, model training, evaluation, and deployment-ready prediction functionality. It is a strong foundation for real-world text classification applications.

---

✨ *Built with Machine Learning & NLP*
