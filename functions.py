from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import emoji

XLMR_PATH = "\roberta8_emoji"
XLMR_NAME = "xlm-roberta-base"
models = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_emojis(text):
    return emoji.demojize(str(text))

def load_model_tokenizer():
    model_name = XLMR_NAME
    model_path = XLMR_PATH

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for trait in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
        if trait == 'extraversion':
            path = r"C:\Users\User\TARUMT\project\Personality\model\model_roberta\big5\roberta8.1_emoji_lowercase"
        else:
            path = model_path
        print(f"Using Device : {device}")
        file_name = path + f"/{trait}_roberta_model.pt"
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 1)
        model.to(device)
        model.load_state_dict(torch.load(file_name))
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
