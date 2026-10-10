import pandas as pd
import wikipediaapi

# Initalize wikipedia API with a user agen, required by wikipedia
wiki = wikipediaapi.Wikipedia(
    user_agent="Article_Classification_Data_Collector_Bot",
    language ="en",
    extract_format=wikipediaapi.ExtractFormat.WIKI)

# Data categories to be classified
categories_to_track = {
    "Technology": ["Artificial Intelligence", "Quantum Computing", "Cloud Computing"],
    "History": ["Industrial Revolution", "World War II", "American Revolution"],
    "Science": ["Space exploration", "Biology", "Chemistry"]
}

data = []

print("Beginning wikipedia data pipeline...")

# loop to fetch page data on categories
for label, topics in categories_to_track.items():
    for topic in topics:
        page = wiki.page(topic)

        if page.exists():
            text_content = page.summary
            data.append({
                "test": text_content,
                "category": label,
                "source_topic": topic
            })
            print(f"Page for topic '{topic}' exists. Data collected.")
        else:
            print(f"Page for topic '{topic}' does not exist.")

#convert dictionary to clean pandas dataframe
df = pd.DataFrame(data)

# Save the dataframe to a CSV file
df.to_csv("wikipedia_dataset.csv", index=False)
print("Data collection complete. Dataset saved to 'wikipedia_dataset.csv'.")