import streamlit as st
import joblib
import requests
import os
import re

# =============================
# Load ML Components
# =============================
emergency_model = joblib.load("emergency_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# =============================
# HuggingFace Setup
# =============================
HF_API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"

HF_HEADERS = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
}

def hf_zero_shot(text, labels):
    payload = {
        "inputs": text,
        "parameters": {"candidate_labels": labels}
    }

    response = requests.post(
        HF_API_URL,
        headers=HF_HEADERS,
        json=payload,
        timeout=30
    )

    if response.status_code != 200:
        raise Exception(
            f"Request failed: {response.status_code}\n{response.text}"
        )

    result = response.json()
    # wrap dict in list if needed
    if isinstance(result, dict):
        result = [result]

    return result

incident_labels = [
    "fire",
    "flood",
    "earthquake",
    "hurricane",
    "explosion",
    "wildfire",
    "building collapse",
    "transport accident"
]
# =============================
# Text Cleaning
# =============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# =============================
# Severity Agent
# =============================
class SeverityAgent:
    def assess(self, text):
        high_keywords = ["urgent", "help", "critical", "injured", "dead"]
        if any(word in text for word in high_keywords):
            return "High"
        return "Medium"

severity_agent = SeverityAgent()

# =============================
# Dispatch Agent
# =============================
class DispatchAgent:
    def route(self, incident_type):
        routing_map = {
            "fire": "Fire Department",
            "flood": "Disaster Response Team",
            "earthquake": "Disaster Response Team",
            "hurricane": "Disaster Response Team",
            "explosion": "Police & Fire Department",
            "wildfire": "Fire Department",
            "building collapse": "EMS & Fire Department",
            "transport accident": "EMS & Police"
        }
        return routing_map.get(incident_type, "General Emergency Unit")

dispatch_agent = DispatchAgent()

# =============================
# Main Pipeline
# =============================
def respondrAI_pipeline(text):

    # Step 0: Clean the text
    cleaned = clean_text(text)

    # Step 1: Emergency Detection
    vec = vectorizer.transform([cleaned])
    is_emergency = emergency_model.predict(vec)[0]

    if is_emergency == 0:
        return {
            "emergency": False,
            "message": "No emergency detected."
        }

    # Step 2: Incident Classification via HF Zero-Shot
    hf_result = hf_zero_shot(cleaned, incident_labels)


    incident_type = hf_result[0]['label']

    # Step 3: Severity Assessment
    priority = severity_agent.assess(cleaned)

    # Step 4: Dispatch Routing
    unit = dispatch_agent.route(incident_type)

    return {
        "emergency": True,
        "incident_type": incident_type,
        "priority": priority,
        "dispatch_to": unit
    }

# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="RespondrAI", page_icon="🚨")

st.title("🚨 RespondrAI")
st.markdown("Agentic AI for Emergency Incident Classification & Dispatch")

user_input = st.text_area("Enter a tweet or emergency report:")

if st.button("Analyze Incident"):


    if user_input.strip() == "":
        st.warning("Please enter text first.")
    else:
        result = respondrAI_pipeline(user_input)

        if not result["emergency"]:
            st.success("✅ No emergency detected.")
        else:
            st.error("🚨 Emergency Detected!")
            st.write("**Incident Type:**", result["incident_type"])
            st.write("**Priority Level:**", result["priority"])
            st.write("**Dispatch To:**", result["dispatch_to"])
