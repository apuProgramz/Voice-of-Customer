# ============================================
# 0) Setup
# ============================================
# !pip install -q transformers sentencepiece torch torchvision torchaudio nltk pandas matplotlib rapidfuzz

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk

import torch
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
from rapidfuzz import fuzz

nltk.download('vader_lexicon', quiet=True)

plt.rcParams.update({"figure.figsize": (11, 6), "axes.grid": False})

# ============================================
# 1) Load CSV
# ============================================
CSV_PATH_DEFAULT = "D:/MY FILES/Akasa data/data/feedback.csv"
ALT_CSV = "/Users/apurvaghare/My Files/Akasa Data/data/feedback.csv"

csv_path = CSV_PATH_DEFAULT if os.path.exists(CSV_PATH_DEFAULT) else ALT_CSV
df = pd.read_csv(csv_path)
df = df.head(2000).copy() 

def infer_text_column(df: pd.DataFrame) -> str:
    for cand in ["feedback", "review", "reviews", "comment", "comments", "text", "message", "content"]:
        if cand in df.columns:
            return cand
    obj_cols = [c for c in df.columns if df[c].dtype == 'object']
    return obj_cols[0]

text_col = infer_text_column(df)
df[text_col] = df[text_col].astype(str).fillna("")
print(f"Using text column: {text_col}")
df.head(3)

# ============================================
# 2) Tags + Synonym Rules
# ============================================
TAG_KEYWORDS = {
    "Add_on_fees": ["extra charges", "hidden fees", "add on fee", "surcharge"],
    "Airport_rules": ["security check", "airport rule", "boarding gate", "customs"],
    "Amount_of_leg_space": ["legroom", "seat space", "leg space", "cramped"],
    "Auto_assigning_of_seats": ["auto seat", "random seat", "automatic seat assignment"],
    "Availability_of_add_ons": ["extra luggage option", "add on available", "seat selection option"],
    "Aviation_quality": ["airline standard", "aviation quality", "flight quality"],
    "charging_port": ["usb port", "charging point", "power socket"],
    "Cleanliness_of_aircraft": ["dirty plane", "clean aircraft", "hygiene", "sanitization"],
    "Communication_and_updates": ["status update", "delay info", "announcements missing", "communication"],
    "Crew_uniform": ["crew dress", "uniform", "attire of staff"],
    "Excess_baggage_rules": ["extra baggage", "baggage charges", "luggage fee"],
    "Flight_boarding_process": ["boarding gate", "queue", "boarding delay"],
    "In_flight_announcements": ["announcement", "captain said", "crew announcement"],
    "In_flight_service": ["cabin service", "staff service", "hospitality"],
    "In_flight_temperature": ["too cold", "air conditioning", "temperature in cabin", "too hot"],
    "Luggage_belt_wait_period": ["baggage belt", "luggage wait", "carousel"],
    "Price_of_ticket": ["ticket price", "fare", "expensive", "cheap ticket"],
    "Professionalism_of_flight_crew": ["rude crew", "friendly staff", "professional crew"],
    "Professionalism_of_ground_staff": ["ground staff rude", "check in staff", "helpdesk"],
    "Punctuality_of_flight": ["on time", "delay", "late flight"],
    "Quality_of_food": ["bad food", "tasty food", "meal quality"],
    "Seat_quality": ["comfortable seat", "seat broken", "recline"],
    "Variety_of_food": ["food choice", "menu", "options in food"],
    "Web_check_in_process": ["online check in", "web check in", "app check in"]
}
TAGS = list(TAG_KEYWORDS.keys())


# ============================================
# 3) Rule-based Tagging + Optional Zero-shot Fallback
# ============================================

from rapidfuzz import fuzz
from tqdm import tqdm

# --- RULE-BASED TAGGING ---
def rule_based_tags(text: str) -> list:
    tags = []
    txt = text.lower()
    for tag, kws in TAG_KEYWORDS.items():
        for kw in kws:
            if fuzz.partial_ratio(kw.lower(), txt) > 80:  # fuzzy match
                tags.append(tag)
                break
    return list(set(tags))

# --- SWITCH: use zero-shot fallback or not ---
USE_ZERO_SHOT = False   # <<< change to True if you want zero-shot fallback

if USE_ZERO_SHOT:
    from transformers import pipeline

    zshot = pipeline("zero-shot-classification",
                     model="valhalla/distilbart-mnli-12-1",
                     device=-1)
    ZSCORE_THRESHOLD = 0.35

    def assign_tags_zero_shot(texts, labels, threshold=0.35, batch_size=16):
        assigned = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Zero-shot tagging"):
            chunk = texts[i:i+batch_size]
            res = zshot(chunk, candidate_labels=labels, multi_label=True)
            if isinstance(res, dict):
                res = [res]
            for r in res:
                labs = [lab for lab, score in zip(r["labels"], r["scores"]) if score >= threshold]
                assigned.append(labs)
        return assigned
else:
    assign_tags_zero_shot = None  # no-op

# --- FINAL ASSIGNMENT ---
all_texts = df[text_col].tolist()
final_tags = []

if USE_ZERO_SHOT:
    zs_tags = assign_tags_zero_shot(all_texts, TAGS, threshold=0.35)
else:
    zs_tags = [[] for _ in all_texts]  # empty if no zero-shot

for text, z in zip(all_texts, zs_tags):
    r = rule_based_tags(text)
    if r:
        final_tags.append(r)   # prefer rule-based
    else:
        final_tags.append(z)   # fallback (if enabled)

df["__tags__"] = final_tags
df.head(10)

# ============================================
# 4) Sentiment Analysis
# ============================================
sia = SentimentIntensityAnalyzer()

def label_sentiment(text: str) -> str:
    s = sia.polarity_scores(text)
    if s["compound"] > 0.05:
        return "positive"
    elif s["compound"] < -0.05:
        return "negative"
    else:
        return "neutral"

df["__sentiment__"] = df[text_col].apply(label_sentiment)
df[["__tags__", "__sentiment__"]].head(10)

# ============================================
# 5) Aggregation
# ============================================
def summarize_counts(df, tags):
    rows = []
    for tag in tags:
        sub = df[df["__tags__"].apply(lambda L: tag in L)]
        total = len(sub)
        pos = int((sub["__sentiment__"] == "positive").sum())
        neu = int((sub["__sentiment__"] == "neutral").sum())
        neg = int((sub["__sentiment__"] == "negative").sum())
        rows.append({
            "tag": tag,
            "total": total,
            "positive": pos,
            "neutral": neu,
            "negative": neg
        })
    return pd.DataFrame(rows).sort_values("total", ascending=False).reset_index(drop=True)

summary = summarize_counts(df, TAGS)
summary.to_csv("tag_sentiment_summary.csv", index=False)
summary

# ============================================
# 6) Plots
# ============================================
def plot_tag_frequency(summary, top_n=None, fname="tag_frequency.png"):
    data = summary if not top_n else summary.head(top_n)
    plt.figure()
    plt.bar(data["tag"], data["total"])
    plt.xticks(rotation=90, ha="right")
    plt.title("Tag Frequency")
    plt.xlabel("Tags")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.show()

def plot_tag_sentiment_stacked(summary, top_n=None, fname="tag_sentiment_stacked.png"):
    data = summary if not top_n else summary.head(top_n)
    neg = data["negative"].values
    neu = data["neutral"].values
    pos = data["positive"].values

    plt.figure()
    plt.bar(data["tag"], neg)
    plt.bar(data["tag"], neu, bottom=neg)
    plt.bar(data["tag"], pos, bottom=neg+neu)
    plt.xticks(rotation=90, ha="right")
    plt.title("Sentiment Breakdown per Tag")
    plt.xlabel("Tags")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.show()

plot_tag_frequency(summary)
plot_tag_sentiment_stacked(summary)

print("Outputs: tag_sentiment_summary.csv, tag_frequency.png, tag_sentiment_stacked.png")