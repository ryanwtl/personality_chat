from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import emoji

models = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_emojis(text):
    return emoji.demojize(str(text))

def load_model_tokenizer(token):
    model_name = ""
    model_path = "theweekday/personality_traits_"

    tokenizer = RobertaTokenizer.from_pretrained(model_path, token=token)

    for trait in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
        path = f"{model_path}{trait}"
        model = RobertaForSequenceClassification.from_pretrained(path, token=token, num_labels=1)
        model.to(device)
        models[trait] = model 
    return models, tokenizer

def personality_analysis_sentence(sentence, models, tokenizer, max_length=512):
    analysis = {}
    components = ['value', 'score']
    for trait, model in models.items():
        model.to(device)
        encodings = tokenizer([sentence], truncation=True, padding=True, max_length=max_length, return_tensors='pt').to(device)
        model.eval()
        with torch.no_grad():
            output = {}
            score = torch.sigmoid(model(**encodings).logits).item()
            binary_value = 'y' if score > 0.61 else 'n'
            output[components[0]] = binary_value
            output[components[1]] = f"{score:.4f}"
            analysis[trait] = output

    return analysis
