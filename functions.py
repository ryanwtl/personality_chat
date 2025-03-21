from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import emoji
import pandas

XLMR_PATH = r"C:\Users\User\TARUMT\project\Personality\model\model_roberta\big5\roberta8_emoji"
XLMR_NAME = "xlm-roberta-base"
models = {}
tokenizers = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_emojis(text):
    return emoji.demojize(str(text))

def load_model_tokenizer():
    model_name = ""
    model_path = "theweekday/xlmRoBERTa-"

    for trait in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
        path = f"{model_path}{trait}"
        print(path)
        model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1)
        tokenizer = AutoTokenizer.from_pretrained(path)
        tokenizers[trait] = tokenizer
        models[trait] = model 
    return models, tokenizers

def personality_analysis_sentence(sentence, models, tokenizers, max_length=512):
    analysis = {}
    components = ['value', 'score']
    for trait, model in models.items():
        model.to(device)
        encodings = tokenizers[trait]([sentence], truncation=True, padding=True, max_length=max_length, return_tensors='pt').to(device)
        model.eval()
        with torch.no_grad():
            output = {}
            score = torch.sigmoid(model(**encodings).logits).item()
            binary_value = 'y' if score > 0.61 else 'n'
            output[components[0]] = binary_value
            output[components[1]] = f"{score:.4f}"
            analysis[trait] = output

    return analysis

def validate_personality_with_llm(text, scores, client):
    messages = [
        {"role": "system", "content": "You are an expert in personality analysis."},
        {"role": "user", "content": f"Here is a text: '{text}'. The following are personality scores: {scores}. Validate these scores based on the text, by giving a score and a brief explanation. then return in a JSON format."}
    ]

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    result = ""
    for chunk in completion:
        result += chunk.choices[0].delta.content or ""
    
    return result

def property_recommend(user, scores, client):
    messages = [
        {"role": "system", "content": "You are to recommend a property based on the personality traits"},
        {"role": "user", "content": f"""Here user identity: '{user}'. The following are user's personality scores: {scores}. based on the personality traits, give suggestions on the type of property in terms of :
1. Location
2. Rent & Payment Terms
3. Room & Facility Quality
4. Amenities & Services
5. Maintenance & Support
6. Community Atmosphere
7. Reputation & Reviews
8. Safety & Security
9. Contractual Terms & Conditions
10. Customer Service
11. Flexibility & Exit Options
12. Overall Experience
         
keep it brief and return in a JSON format."""}
    ]

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    result = ""
    for chunk in completion:
        result += chunk.choices[0].delta.content or ""
    
    return result

def analysis_result_output2(analysis):
    # Create a list of dictionaries to store the results for each trait
    results = []
    
    for trait, details in analysis.items():
        value = 'Yes' if details['value'] == 'y' else 'No'
        score = float(details['score'])
        
        # Append the data to the results list
        results.append({
            "Trait": trait.capitalize(),
            "Value (Yes/No)": value,
            "Score": f"{score:.4f}"
        })
    
    # Convert the results to a DataFrame
    df = pandas.DataFrame(results)

    return df
