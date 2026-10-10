import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

#Loading preprocessed data and vectorizer from disk
print("Loading preprocessed data and vectorizer from disk...")
X = joblib.load("X_matrix.pkl")
y = joblib.load("Y_labels.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

#Spliting the data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nLoaded Successfully. Training data on {X_train.shape[0]} articles, testing on {X_test.shape[0]} articles.")

# Initializing and training the Multinomial Naive Bayes classifier
print("\nTraining the Multinomial Naive Bayes classifier...")
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluating the model on the test set
print("\nEvaluating the model on the test set...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nTraining Complete! Model Accuracy: {accuracy*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Saving the trained model to disk for future use
joblib.dump(model, "text_classification_model.pkl")
print("\nTrained model saved to 'text_classification_model.pkl'.")