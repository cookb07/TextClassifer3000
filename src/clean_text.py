#Function to clean text data by removing punctuation and converting to lowercase
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    punctuation = [".", ",", "!", "?", ";", ":", "-", "_", "(", ")", "[", "]", "{", "}", "'", '"', "/", "\\", "@", "#", "$", "%", "^", "&", "*", "+", "=", "<", ">", "|"]
    for char in punctuation:
        text = text.replace(char, "")
    return text