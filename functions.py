from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import emoji
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from groq import Groq


XLMR_PATH = r"C:\Users\User\TARUMT\project\Personality\model\model_roberta\big5\roberta8_emoji"
XLMR_NAME = "xlm-roberta-base"
models = {}
tokenizers = {}

properties = [
        {
            "name": "The Ooak, Mont Kiara",
            "location": "Mont Kiara, Kuala Lumpur",
            "min_price": 600000,
            "max_price": 1700000,
            "min_rental_price": 2200,
            "max_rental_price": 7000,
            "aligned_personality_trait": "extraversion",
            "image": "https://sg1-cdn.pgimgs.com/projectnet-project/133519/ZPPHO.113546608.R800X800.jpg/400x300",
            "floor_plans": {
                "type C1": {
                "bedrooms": 1,
                "bathrooms": 1,
                "size(sqft)": "710"
                },
                "type C2": {
                "bedrooms": 1,
                "bathrooms": 1,
                "size(sqft)": "833-853"
                },
                "type C3": {
                "bedrooms": 1,
                "bathrooms": 1,
                "size(sqft)": "788"
                },
                "type C4": {
                "bedrooms": 1,
                "bathrooms": 1,
                "size(sqft)": "716-736"
                },
                "type B2": {
                "bedrooms": 2,
                "bathrooms": 2,
                "size(sqft)": "1166-1236"
                },
                "type B3": {
                "bedrooms": 2,
                "bathrooms": 2,
                "size(sqft)": "1103"
                },
                "type A1b": {
                "bedrooms": 1,
                "bathrooms": 1,
                "size(sqft)": "696"
                },
                "type A2b": {
                "bedrooms": 1,
                "bathrooms": 1,
                "size(sqft)": "696"
                },
                "type A1a": {
                "bedrooms": 1,
                "bathrooms": 1,
                "size(sqft)": "816"
                },
                "type A2a": {
                "bedrooms": 1,
                "bathrooms": 1,
                "size(sqft)": "816"
                },
                "type B1": {
                "bedrooms": 2,
                "bathrooms": 2,
                "size(sqft)": "1016"
                },
            },
            "nearby": {
                "Transport": {
                "MRT": False,
                "Bus Stop": True
                },
                "Family Living": {
                "Hospital/Clinic": True,
                "School": True,
                "Post Box": True,
                "ATM": True,
                "Supermarket": True
                },
                "Lifestyle": {
                "Park/Gym": True,
                "Restaurant/Pubs": True,
                "Movie Theater": True,
                "Malls": True,
                "Place of Worship": True
                }
            }
            
        },
        {
            "name": "Rica Residence, Sentul",
            "location": "Sentul, Kuala Lumpur",
            "min_price": 440000,
            "max_price": 795000,
            "min_rental_price": 700,
            "max_rental_price": 2800,
            "aligned_personality_trait": "conscientiousness",
            "image": "https://sg1-cdn.pgimgs.com/projectnet-project/4050/ZPPHO.130758924.R800X800.jpg/400x300",
            "floor_plans": {
                "1_Bedroom": {
                "bedrooms": 1,
                "bathrooms": 1,
                "size(sqft)": "657-800"
                },
                "2_Bedroom": {
                "bedrooms": 2,
                "bathrooms": 2,
                "size(sqft)": "1000-1100"
                },
                "3_Bedroom": {
                "bedrooms": 3,
                "bathrooms": 2,
                "size(sqft)": "1200-1238"
                },
                "Dual_Key_Units": {
                "bedrooms": "1+1",
                "bathrooms": "2",
                "size(sqft)": "1200"
                }
            },
            "nearby": {
                "Transport": {
                "MRT": False,
                "Bus Stop": True
                },
                "Family Living": {
                "Hospital/Clinic": True,
                "School": True,
                "Post Box": True,
                "ATM": True,
                "Supermarket": True
                },
                "Lifestyle": {
                "Park/Gym": True,
                "Restaurant/Pubs": True,
                "Movie Theater": True,
                "Malls": True,
                "Place of Worship": True
                }
            },
        },
        {
            "name": "M Vertica, Cheras",
            "location": "Cheras, Kuala Lumpur",
            "min_price": 345000,
            "max_price": 768000,
            "min_rental_price": 550,
            "max_rental_price": 5000,
            "aligned_personality_trait": "openness",
            "image": "https://sg1-cdn.pgimgs.com/projectnet-project/4818/ZPPHO.113018780.R800X800.jpg/400x300",
            "floor_plans": {
                "Type A": {
                "bedrooms": 3,
                "bathrooms": 2,
                "size(sqft)": "850"
                },
                "Type B": {
                "bedrooms": 3,
                "bathrooms": 2,
                "size(sqft)": "1000"
                }
            },
            "nearby": {
                "Transport": {
                "MRT": True,
                "Bus Stop": True
                },
                "Family Living": {
                "Hospital/Clinic": True,
                "School": True,
                "Post Box": True,
                "ATM": True,
                "Supermarket": True
                },
                "Lifestyle": {
                "Park/Gym": True,
                "Restaurant/Pubs": True,
                "Movie Theater": False,
                "Malls": True,
                "Place of Worship": True
                }
            }
        },
        {
            "name": "The PARC3, Taman Pertama",
            "location": "Taman Pertama, Kuala Lumpur",
            "min_price": 382000,
            "max_price": 1200000,
            "min_rental_price": 250,
            "max_rental_price": 4300,
            "aligned_personality_trait": ["agreeableness","neuroticism"],
            "image": "https://sg1-cdn.pgimgs.com/projectnet-project/45504/ZPPHO.112169464.R800X800.jpg/400x300",
            "floor_plans": {
                "Type A1": {
                "bedrooms": 3,
                "bathrooms": 2,
                "size(sqft)": "592"
                },
                "Type A3": {
                "bedrooms": 3,
                "bathrooms": 2,
                "size(sqft)": "977"
                },
                "Type B1": {
                "bedrooms": 3,
                "bathrooms": 2,
                "size(sqft)": "931"
                },
                "Type B2": {
                "bedrooms": 3,
                "bathrooms": 2,
                "size(sqft)": "977"
                },
                "Type C1": {
                "bedrooms": 4,
                "bathrooms": 2,
                "size(sqft)": "1278"
                },
                "Type C2": {
                "bedrooms": 4,
                "bathrooms": 2,
                "size(sqft)": "1278"
                },
                "Type D1": {
                "bedrooms": 2,
                "bathrooms": 2,
                "size(sqft)": "749"
                },
                "Type D2": {
                "bedrooms": 2,
                "bathrooms": 2,
                "size(sqft)": "749"
                },
                "Type E1": {
                "bedrooms": 1,
                "bathrooms": 1,
                "size(sqft)": "592"
                },
                "Type E2": {
                "bedrooms": 1,
                "bathrooms": 1,
                "size(sqft)": "592"
                }
            },
            "nearby": {
                "Transport": {
                    "MRT": True,
                    "Bus Stop": True
                },
                "Family Living": {
                    "Hospital/Clinic": True,
                    "School": True,
                    "Post Box": True,
                    "ATM": True,
                    "Supermarket": True
                },
                "Lifestyle": {
                    "Park/Gym": True,
                    "Restaurant/Pubs": True,
                    "Movie Theater": True,
                    "Malls": True,
                    "Place of Worship": True
                }
            }
        }
    ]

def allocate_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = allocate_device()

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

def get_llama_response(user_input):
    client = Groq()
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful business assistant."},
            {"role": "user", "content": user_input},
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.5,
        top_p=1,
        stop=None,
        stream=False,
    )
    return chat_completion.choices[0].message.content

def personality_analysis_sentence(sentence, models, tokenizers, threshold, max_length=512):
    analysis = {}
    components = ['value', 'score']

    print(f"\nusing_device: {device}")

    sentence = convert_emojis(sentence)
    sentence = sentence.replace("\n", " ").replace("\r", " ").strip()

    print(f"\nanalysis_sentence:\n{sentence}")
    for trait, model in models.items():
        model.to(device)
        encodings = tokenizers[trait]([sentence], truncation=True, padding=True, max_length=max_length, return_tensors='pt').to(device)
        model.eval()
        with torch.no_grad():
            output = {}
            score = torch.sigmoid(model(**encodings).logits).item()
            binary_value = 'y' if score > threshold else 'n'
            output[components[0]] = binary_value
            output[components[1]] = f"{score:.4f}"
            analysis[trait] = output

    return analysis

def property_prediction(results, threshold, properties=properties):
    """
    Streamlit function to display recommended properties based on personality trait scores.
    
    :param results: Dictionary containing personality traits and their scores.
    :param properties: List of property dictionaries with aligned personality traits.
    """
    st.title("🏡 Personality-Based Property Recommendations")
    st.markdown("Find the best property that aligns with your personality traits! 🏙️✨")
    
    if not results:
        st.warning("⚠️ No personality data provided!")
        return
    
    selected_properties = []
    
    for prop in properties:
        trait = prop.get("aligned_personality_trait", "")
        if (isinstance(trait,list)):
            for t in trait:
                t = t.lower()
        else:
            t = trait.lower()
        if t in results and float(results[t]['score']) > threshold:
            print(f"trait:{trait},score:{results[t]['score']}")
            selected_properties.append(prop)

    if selected_properties:
        with st.expander("🏠 Recommended Properties Based on Your Personality Traits"):
            display_radar_chart(results, threshold)
            for prop in selected_properties:
                st.subheader(f"🏠 {prop['name']}")
                st.text(f"📍 Location: {prop.get('location', 'Not Available')}")
                st.text(f"💰 Price Range: RM{prop.get('min_price', 'N/A')} - RM{prop.get('max_price', 'N/A')}")
                st.text(f"🏢 Rental Price: RM{prop.get('min_rental_price', 'N/A')} - RM{prop.get('max_rental_price', 'N/A')}")
                
                if "image" in prop:
                    st.image(prop["image"], use_container_width=True, caption=prop['name'])
                
                st.markdown("### 🏠 Floor Plans")
                for plan, details in prop.get("floor_plans", {}).items():
                    st.markdown(f"- **{plan}**: {details['bedrooms']} Bedrooms, {details['bathrooms']} Bathrooms, {details['size(sqft)']} sqft")
                
                st.markdown("### 🌍 Nearby Amenities")
                for category, amenities in prop.get("nearby", {}).items():
                    st.markdown(f"**{category}**: {', '.join([key for key, value in amenities.items() if value])}")
                
                st.markdown("### 🏡 Why This Property?")
                explanations = {
                    "openness": "People high in Openness are typically open to new experiences, creative, curious, and enjoy variety. They are more likely to be drawn to unique, innovative, or eclectic environments.\n\n**Choice:** M Vertica, Cheras: This property has diverse and extensive facilities like a maze garden, reflexology trail, herbs & farming garden, and various sports courts. People high in Openness may appreciate these novel, creative spaces and the variety of lifestyle options.",
                    "conscientiousness": "People high in Conscientiousness are organized, reliable, and prefer structure. They may appreciate practicality, cleanliness, and a focus on long-term goals or family-oriented living.\n\n**Choice:** Rica Residence, Sentul: The 24-hour security, swimming pool, and gym would attract those who value stability and routine. The well-maintained, structured amenities would likely appeal to conscientious individuals.",
                    "extraversion": "Extraverted people tend to be social, outgoing, and energetic. They prefer environments that encourage socializing, entertainment, and active engagement.\n\n**Choice:** The Ooak Serviced Apartments, Mont Kiara: This property offers lounges, swimming pools, and close proximity to restaurants, parks, and gyms, all of which provide social and leisure activities for an extraverted person.",
                    "agreeableness": "People high in Agreeableness tend to be cooperative, kind, and compassionate. They value peaceful, harmonious environments with a sense of community.\n\n**Choice:** The PARC3, Taman Pertama: With a variety of facilities like badminton halls and barbecue areas, this could appeal to people who enjoy quiet, communal activities with family and friends.",
                    "neuroticism": "People high in Neuroticism may have a higher sensitivity to stress and are likely to prefer environments that offer security, stability, and low levels of risk or unpredictability.\n\n**Choice:** The PARC3, Taman Pertama: The 24-hour security and organized parking structure are features that neurotic individuals would appreciate, as they offer a sense of security and predictability."
                }

                traits = prop.get("aligned_personality_trait", "")
                if(isinstance(traits,list)):
                    for trait in traits:
                        aligned_trait = trait.lower()
                else:
                    aligned_trait = traits.lower()

                print(f"aligned_trait:{aligned_trait}")
                if aligned_trait in explanations:
                    st.markdown(explanations[aligned_trait])
    else:
        st.info("😕 No matching properties based on personality traits.")

def create_radar_chart(sample_results,threshold):
    labels = []
    values = []
    for trait, result in sample_results.items():
        labels.append(trait.capitalize())
        values.append(float(result['score']))
    
    fig = go.Figure()
    
    # Add the main personality scores plot
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Close the radar chart loop
        theta=labels + [labels[0]],
        fill='toself',
        name='Personality Scores',
        line=dict(color='blue'),
        marker=dict(size=8, color='blue')
    ))
    
    # Add a shaded region below threshold
    fig.add_trace(go.Scatterpolar(
        r=[threshold] * (len(labels) + 1),
        theta=labels + [labels[0]],
        fill='toself',
        name=f'Threshold ({threshold})',
        line=dict(color='red', dash='dash'),
        fillcolor='rgba(255, 0, 0, 0.2)',
        marker=dict(size=6, color='red', symbol='circle')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True
    )
    return fig

chart_counter = 0

def assign_id():
    """Generates a unique ID for each chart."""
    global chart_counter
    chart_counter += 1
    return f"fig. {chart_counter}"
    

def display_radar_chart(sample_results,threshold):
    """Displays the radar chart using Streamlit's chart display function."""

    key = assign_id()

    print(f"\nradar_chart_id: {key}")
    
    fig = create_radar_chart(sample_results,threshold)
    st.plotly_chart(fig,key=key)

# Others Functions   

def validate_personality_with_llm(text, scores, client):
    messages = [
        {"role": "system", "content": "You are an expert in personality analysis."},
        {"role": "user", "content": f"Here is a text: '{text}'. The following are personality analysis: {scores}. Validate these analysis based on the text, by giving a score to indicate the level of trustworthy of the analysis(1 to 10) and a brief explanation. then return in a JSON format."}
    ]

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=1,
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
    df = pd.DataFrame(results)

    return df
