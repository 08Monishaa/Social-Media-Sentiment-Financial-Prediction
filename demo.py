"""
demo.py
End-to-End Demo Script: Social Media Sentiment & Financial Trend Prediction
Loads pre-trained artifacts from the `models/` directory, classifies sample social
media/news texts, extracts trending topics, predicts next-day financial trends
using both the baseline and sentiment-enhanced models, and prints the evaluation report.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd


def load_artifacts(models_dir: str = "models"):
    """Load all serialized models and evaluation metrics from the models directory."""
    required_files = [
        "sentiment_pipeline.joblib",
        "topic_modeler.joblib",
        "baseline_price_model.joblib",
        "sentiment_enhanced_predictor.joblib",
        "evaluation_results.json",
    ]
    for fname in required_files:
        fpath = os.path.join(models_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Missing '{fpath}'. Please extract models.zip into your project root first."
            )

    sentiment_pipe = joblib.load(os.path.join(models_dir, "sentiment_pipeline.joblib"))
    topic_modeler = joblib.load(os.path.join(models_dir, "topic_modeler.joblib"))
    baseline_bundle = joblib.load(os.path.join(models_dir, "baseline_price_model.joblib"))
    enhanced_bundle = joblib.load(os.path.join(models_dir, "sentiment_enhanced_predictor.joblib"))

    with open(os.path.join(models_dir, "evaluation_results.json"), "r", encoding="utf-8") as f:
        eval_report = json.load(f)

    return sentiment_pipe, topic_modeler, baseline_bundle, enhanced_bundle, eval_report


def analyze_social_posts(posts, sentiment_pipe, topic_modeler):
    """Classify sentiment and extract dominant topics for incoming social media / news posts."""
    label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
    preds = sentiment_pipe.predict(posts)
    probs = sentiment_pipe.predict_proba(posts)
    topic_distributions = topic_modeler.transform(posts)

    feature_names = topic_modeler.named_steps["tfidf"].get_feature_names_out()
    lda_components = topic_modeler.named_steps["lda"].components_

    records = []
    for idx, text in enumerate(posts):
        dominant_topic = int(np.argmax(topic_distributions[idx]))
        topic_score = float(np.max(topic_distributions[idx]))
        top_keywords = [
            feature_names[i] for i in lda_components[dominant_topic].argsort()[:-4:-1]
        ]
        records.append({
            "post": text,
            "sentiment_label": label_map[int(preds[idx])],
            "sentiment_score": int(preds[idx]),
            "confidence": round(float(np.max(probs[idx])), 4),
            "dominant_topic": f"Topic {dominant_topic + 1} ({', '.join(top_keywords)})",
            "topic_score": round(topic_score, 4),
        })

    return pd.DataFrame(records)


def predict_financial_trend(market_snapshot: pd.DataFrame, baseline_bundle: dict, enhanced_bundle: dict):
    """Compare baseline (prices-only) vs. sentiment-enhanced predictions on current market data."""
    base_features = baseline_bundle["features"]
    enh_features = enhanced_bundle["features"]

    base_dir = baseline_bundle["direction_classifier"].predict(market_snapshot[base_features])[0]
    base_prob = baseline_bundle["direction_classifier"].predict_proba(market_snapshot[base_features])[0][1]
    base_ret = baseline_bundle["return_regressor"].predict(market_snapshot[base_features])[0]

    enh_dir = enhanced_bundle["direction_classifier"].predict(market_snapshot[enh_features])[0]
    enh_prob = enhanced_bundle["direction_classifier"].predict_proba(market_snapshot[enh_features])[0][1]
    enh_ret = enhanced_bundle["return_regressor"].predict(market_snapshot[enh_features])[0]

    return {
        "Baseline (Historical Prices Only)": {
            "predicted_trend": "UP (Bullish)" if base_dir == 1 else "DOWN (Bearish)",
            "upward_probability": round(float(base_prob), 4),
            "predicted_next_day_return": f"{base_ret * 100:+.3f}%",
        },
        "Sentiment-Enhanced Model": {
            "predicted_trend": "UP (Bullish)" if enh_dir == 1 else "DOWN (Bearish)",
            "upward_probability": round(float(enh_prob), 4),
            "predicted_next_day_return": f"{enh_ret * 100:+.3f}%",
        },
    }


def main():
    print("=" * 72)
    print("SOCIAL MEDIA SENTIMENT & FINANCIAL TREND PREDICTION - DEMO")
    print("=" * 72)

    # 1. Load models and evaluation results
    sentiment_pipe, topic_modeler, baseline_bundle, enhanced_bundle, eval_report = load_artifacts("models")

    # 2. Sample incoming social media posts & financial news headlines
    sample_posts = [
        "Company reports record quarterly earnings beating Wall Street expectations #growth",
        "Strong revenue growth driven by high demand for new AI cloud products #market",
        "Analysts upgrade stock to buy citing robust profit margins and expansion",
        "Company schedules annual shareholder meeting for next month #news",
        "Supply chain disruptions raise mild concerns over short-term shipping costs",
    ]

    print("\n[1] SENTIMENT & TRENDING TOPIC ANALYSIS ON SAMPLE POSTS")
    print("-" * 72)
    sentiment_df = analyze_social_posts(sample_posts, sentiment_pipe, topic_modeler)
    for _, row in sentiment_df.iterrows():
        print(f"Post      : {row['post']}")
        print(f"Sentiment : {row['sentiment_label']} (score={row['sentiment_score']:+d}, conf={row['confidence']:.2f}) | {row['dominant_topic']}")
        print("-" * 72)

    # Aggregate daily sentiment features to combine with market data
    agg_sentiment_mean = float(sentiment_df["sentiment_score"].mean())
    agg_sentiment_std = float(sentiment_df["sentiment_score"].std(ddof=0))
    agg_post_volume = len(sample_posts) * 65  # Scaled daily post count
    agg_topic_score = float(sentiment_df["topic_score"].mean())

    # 3. Combine aggregated sentiment features with current historical market indicators
    current_market_data = pd.DataFrame([{
        "close": 164.50,
        "return_1d": 0.0125,
        "return_5d": 0.0310,
        "volatility_5d": 0.0142,
        "sentiment_mean": agg_sentiment_mean,
        "sentiment_std": agg_sentiment_std,
        "post_volume": agg_post_volume,
        "topic_score": agg_topic_score,
    }])

    print("\n[2] NEXT-DAY FINANCIAL TREND PREDICTION (BASELINE vs. SENTIMENT-ENHANCED)")
    print("-" * 72)
    print(f"Market Snapshot -> Close: ${current_market_data['close'].iloc[0]:.2f} | "
          f"1D Return: {current_market_data['return_1d'].iloc[0]*100:+.2f}% | "
          f"Aggregated Sentiment: {agg_sentiment_mean:+.2f}")
    predictions = predict_financial_trend(current_market_data, baseline_bundle, enhanced_bundle)
    for model_name, res in predictions.items():
        print(f"\n{model_name}:")
        for k, v in res.items():
            print(f"  - {k}: {v}")

    # 4. Display stored Time-Based Split Evaluation Comparison
    print("\n[3] HISTORICAL TIME-BASED TEST SET COMPARISON")
    print("-" * 72)
    split_info = eval_report["split_strategy"]
    print(f"Split Method : {split_info['method']}")
    print(f"Train Period : {split_info['train_period']} ({split_info['train_samples']} days)")
    print(f"Test Period  : {split_info['test_period']} ({split_info['test_samples']} days)\n")

    comp_df = pd.DataFrame(eval_report["model_comparison"]).drop(index="features_used")
    print(comp_df.to_string())
    print("=" * 72)


if __name__ == "__main__":
    main()
