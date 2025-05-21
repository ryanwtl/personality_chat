# 🧑🏻 Personality-Based Property Recommendation App

This is a Streamlit-based interactive web application that recommends properties based on the user's personality traits. It uses advanced NLP models via the HuggingFace and Groq APIs to analyze input text and provide personality-aligned property suggestions. The app is built to visualize user profiles, property traits, and matching logic for intuitive exploration.

---

## 📁 Project Structure

```bash
├── app.py                  # Streamlit app layout and UI
├── functions.py            # Core functions for prediction, matching, and visualization
├── requirements.txt        # Python dependencies
├── results/
│   ├──                     # All the results related to model performance
```

---

## 🚀 Features

- Upload and analyze user personality text
- Predict Big Five traits (OCEAN model) using transformers model 
- Get responses from LLaMA 3.3B via Groq API
- Display personalized property matches
- Visualize personality alignment through radar charts
- Highlight personality-property compatibility

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/property-personality-app.git
cd property-personality-app
```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies

```bash
# if u have setup gpu then install torch first before install the requirements.txt
pip install pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# else simply just install the requirements.txt
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

This app requires the Groq API keys to access the LLaMA-3.3-70B model.

Create a `.env` file or export these in your shell:

```bash
export GROQ_API_KEY=your_groq_token
```

> Replace `your_groq_token` with valid API keys.

---

## ▶️ Running the App

Launch the Streamlit app using:

```bash
streamlit run app/app.py
```

This will open a local URL (usually http://localhost:8501/) in your browser to interact with the app.

---

## 📊 Results Display

The app will automatically load and display:

- **Tables**: performance_length.csv to show the performance of model in different length while performance_language.csv shows the performance of the model in English and Malay
- **Radar Charts**: 
  - `results\Personality Traits among Tenants from Belive.png`: Big Five personality traits visualization of the dataset
  - `results\Number of Tenants vs. Co-occurring Traits.png`: Matching property traits visualization

Make sure these files are present in the `results/` folder before running the app.

---

## 📌 Dependencies

Some key libraries include:

- `streamlit`
- `pandas`
- `plotly`
- `groq`
- `transformers`
- `python-dotenv`

All dependencies are listed in `requirements.txt`.

---

## ✨ Future Enhancements

- Integrate dynamic plotting (e.g., Plotly or Altair)
- Add filtering/sorting to property recommendations
- Expand to include more user input methods (audio, forms, etc.)
- Save session data or export recommendations

---

## 📬 Contact

For questions, issues, or contributions, feel free to open an issue or reach out via GitHub.
