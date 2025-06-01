import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# Load data
df = pd.read_csv("review.csv")  # Should contain 'Age', 'Title', 'Rating'

# Load model and tokenizer
model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
model.eval()

# Few-shot prefix with 5 human examples
few_shot_prefix = """
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

# Function to format the prompt
def construct_prompt(age, title, rating):
    return few_shot_prefix.format(Age=age, Title=title, Rating=rating)

# Review generation
generated_reviews = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating Reviews"):
    prompt = construct_prompt(row['Age'], row['Title'], row['Rating'])
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )
    
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract only the generated review (last part)
    generated = text.split("Review 5:")[1].split("Now, generate")[0].strip()
    final_review = text.split("Now, generate")[-1].strip()
    generated_reviews.append(final_review)

# Save to CSV
df['Generated_Review'] = generated_reviews
df.to_csv("fewshot_generated_reviews.csv", index=False)
print("Review generation complete and saved to 'fewshot_generated_reviews.csv'")
