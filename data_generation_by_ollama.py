import requests
import json
import pandas as pd
from tqdm import tqdm

def generate_review(prompt, model="deepseek:latest"):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, data=json.dumps(data), headers=headers)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"[ERROR] Request failed: {e}")
        return ""

def process_review_dataset(input_csv, output_csv, model_name="gemma:latest"):
    df = pd.read_csv(input_csv)

    required_cols = {"Age", "Title", "Rating"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input CSV must contain these columns: {required_cols}")

    generated_reviews = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating Reviews"):
        age = row["Age"]
        title = row["Title"]
        rating = row["Rating"]

        prompt = (
            """
            Generate a realistic review where a {Age}-year-old user shares their experience using the product '{Title}', rated {Rating} stars. Focus on durability and value.
            
            Examples:
            Review 1:
            "I liked the color, the silhouette, and the fabric of this dress. But the ruching just looked bunchy and ruined the whole thing. I was so disappointed, I really waned to like this dress. Runs a little small; I would need to size up to make it workappropriate."
            Review 2:
            "I was a little bit afraid about the size of this dress but I just received it and I'm in love! The size fits perfectly! The only thing is: I'm 5'2 so I find a bit long, but this is not a real problem! Thank you, ModCloth, now, I'm not afraid anymore about order more pieces of clothe!"
            Review 3:
            "I like this dress, but OMG is it short. I had to move it to the weekend clothes section of my closet, so I don't flash my coworkers. Maybe it works better for ladies with a little less booty, but if you have a round derriere, this is NSFW."
            Review 4:
            "I'm not familiar with dresses nor am I familiar with dresses over $25, but I saw this dress on polyvore and I knew I had to get it. Although the price had me waver, I'm incredibly satisfied with the product."
            Review 5:
            "The dress itself was great quality. I had to return it though.  Something was off in the chest area. I could quite figure it out, but it just didn't flatter me. It made me feel frumpy."
            
            Now, generate a similar review where a {Age}-year-old user shares their experience using the product '{Title}', rated {Rating} stars. Focus on durability and value.
            """
        )

        review = generate_review(prompt, model=model_name)
        generated_reviews.append(review)

    df["Generated_Review"] = generated_reviews
    df.to_csv(output_csv, index=False)
    print(f"[✅] Reviews saved to: {output_csv}")

if __name__ == "__main__":
    print("[INFO] Starting product review generation using Ollama model...")
    input_file = "product_metadata.csv"  # Must have Age, Title, Rating columns
    output_file = "generated_product_reviews.csv"
    model = "gemma:latest"  # Or llama3, openchat, etc.
    process_review_dataset(input_file, output_file, model_name=model)
