# Spam-Ham Classification using Word2Vec and Random Forest

## Aim

The goal of this project is to classify text messages as either "spam" or "ham" (non-spam) using a combination of Word2Vec word embeddings and a Random Forest classifier. The process involves preprocessing text data, transforming it into word vectors using Word2Vec, and then using these vectors to train a machine learning model for classification.

## Methodology

### 1. Data Collection
The dataset used in this project is a CSV file containing labeled messages ("spam" or "ham") and the corresponding message text.

### 2. Data Preprocessing
- **Text cleaning**: Each message is cleaned by removing non-alphabetic characters and converting all text to lowercase.
- **Lemmatization**: Words are lemmatized using NLTK's `WordNetLemmatizer` to convert them to their base form.
- **Tokenization**: The cleaned text is tokenized into words.

### 3. Word2Vec Embedding
Word2Vec is used to create vector representations for words:
- **Pre-trained embeddings**: The project uses the pre-trained 'word2vec-google-news-300' model from Gensim to obtain word vectors for common words.
- **Training Word2Vec model**: A custom Word2Vec model is trained from the corpus of messages to generate word embeddings specific to the dataset.

### 4. Document Vector Representation
- **Average Word2Vec**: Since Word2Vec generates a vector for each word, we aggregate these word vectors by calculating the average for each document (message) to get a single vector representation for each message.

### 5. Model Training
- **Random Forest Classifier**: A Random Forest model is trained on the processed document vectors (features) and the corresponding labels (spam or ham).

### 6. Evaluation
The model's performance is evaluated using accuracy, confusion matrix, and classification report, which includes precision, recall, and F1-score.

## Libraries Used

- **gensim**: For training and using Word2Vec models.
- **nltk**: For text preprocessing tasks such as tokenization, lemmatization, and stopword removal.
- **sklearn**: For machine learning tasks such as model training, splitting the dataset, and evaluation.
- **pandas**: For data manipulation and reading the dataset.
- **tqdm**: For showing progress during computations.

## Evaluation Metric

- **Accuracy**: Measures the proportion of correct predictions.
- **Confusion Matrix**: Provides insight into the true positives, false positives, true negatives, and false negatives.
- **Classification Report**: Includes precision, recall, and F1-score, which provide a more detailed evaluation of the model performance, especially for imbalanced datasets.

## Results

The Random Forest classifier achieved an accuracy of approximately 98%, which suggests that the model performs well in classifying messages as spam or ham. The classification report showed high precision and recall for both classes, indicating a well-balanced model.

```plaintext
Confusion Matrix:
[[974   3]
 [ 15  88]]

Classification Report:
              precision    recall  f1-score   support

         ham       0.98      1.00      0.99       977
        spam       0.97      0.85      0.91       103

    accuracy                           0.98      1080
   macro avg       0.97      0.92      0.95      1080
weighted avg       0.98      0.98      0.98      1080


## Conclusion
This project demonstrates the effectiveness of Word2Vec embeddings for text classification tasks. By leveraging pre-trained embeddings and training a Random Forest classifier, the model can achieve high accuracy in distinguishing between spam and ham messages. This approach can be extended to other text classification problems as well.

## Future Work
**Hyperparameter tuning:** The Random Forest model could benefit from hyperparameter tuning to further improve accuracy.

**Deep learning models:** Exploring more advanced models like LSTM or Transformer-based architectures could potentially lead to better performance.

**Data augmentation:** Increasing the dataset size by including more diverse messages could improve the model's robustness.

**Real-time classification:** Implementing the model in a real-time spam detection system could enhance its practical usability in email or messaging platforms.
