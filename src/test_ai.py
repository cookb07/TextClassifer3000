import joblib

#Loading trained model and vectorizer from disk
print("Loading trained model and vectorizer from disk for testing...")
model = joblib.load("text_classification_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Writing brand new sentences for testing the model
test_sentences = [
    "The new software update uses neural networks to speed up processing in cloud environments.",
    "Napoleon's army marched across Europe during the brutal winter campaigns of the early 19th century.",
    "Biologists discovered a new species of bacteria mutating its DNA structure inside deep ocean volcanic vents."
]

print("\nRunning predictions on completely new tests...")
print("-" * 60)

for sentence in test_sentences:
    # Preprocess the sentence using the same vectorizer
    sentence_vectorized = vectorizer.transform([sentence])
    
    # Predict the category using the trained model
    predicted_category = model.predict(sentence_vectorized)[0]
    
    print(f"Sentence: {sentence}")
    print(f"Predicted Category: {predicted_category}")
    print("-" * 60)