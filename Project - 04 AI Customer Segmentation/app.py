"""
Flask Web Application for Project - 04: AI Customer Segmentation
Spotify Behavioral Customer Intelligence & LLM Strategy System
Serves the executive dashboard, REST APIs for interactive visualizations,
live strategy generation, user lookup, campaign simulator, and PDF report downloads.
"""

import os
import sys
import json
import pandas as pd
from flask import Flask, render_template, jsonify, request, send_file

# Add project root to path
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from data.data_generator import generate_spotify_telemetry_data
from models.segmentation_engine import SpotifySegmentationEngine
from models.llm_intelligence import LLMCustomerIntelligence
from models.campaign_simulator import CampaignSimulator

app = Flask(__name__, static_folder="static", template_folder="templates")

# Global instances initialized on startup
DATA_DIR = os.path.join(base_dir, "data")
CSV_PATH = os.path.join(DATA_DIR, "spotify_customer_telemetry.csv")
REPORT_PATH = os.path.join(base_dir, "report", "Spotify_AI_Customer_Segmentation_Report.pdf")

# Ensure data exists
if not os.path.exists(CSV_PATH):
    print("Generating Spotify customer telemetry data...")
    df_raw, _ = generate_spotify_telemetry_data(DATA_DIR, n_users=20000)
else:
    df_raw = pd.read_csv(CSV_PATH)

print("Initializing Spotify AI Behavioral Segmentation Engine...")
segmentation_engine = SpotifySegmentationEngine(df_raw).fit()
llm_intelligence = LLMCustomerIntelligence()
campaign_simulator = CampaignSimulator()
print("All AI models and customer intelligence pipelines loaded successfully!")

@app.route("/")
def index():
    """Serves the main interactive Spotify AI Customer Intelligence Dashboard."""
    summary = {
        "total_users": len(segmentation_engine.df),
        "n_clusters": segmentation_engine.n_clusters,
        "silhouette_score": segmentation_engine.cluster_metrics["silhouette_score"],
        "calinski_harabasz": segmentation_engine.cluster_metrics["calinski_harabasz_score"],
        "davies_bouldin": segmentation_engine.cluster_metrics["davies_bouldin_score"],
        "avg_ltv": round(float(segmentation_engine.df["lifetime_value_ltv"].mean()), 2),
        "total_monthly_revenue": round(float(segmentation_engine.df["monthly_spend"].sum()), 2),
        "clusters": segmentation_engine.segment_profiles
    }
    return render_template("index.html", summary=summary)

@app.route("/api/summary", methods=["GET"])
def get_summary():
    """Returns cluster profiles, global KPIs, and evaluation metrics."""
    return jsonify({
        "total_users": len(segmentation_engine.df),
        "n_clusters": segmentation_engine.n_clusters,
        "cluster_metrics": segmentation_engine.cluster_metrics,
        "k_evaluation": segmentation_engine.k_evaluation,
        "segment_profiles": segmentation_engine.segment_profiles
    })

@app.route("/api/pca-scatter", methods=["GET"])
def get_pca_scatter():
    """Returns sampled points with PCA coordinates and telemetry for scatter visualization."""
    limit = request.args.get("limit", default=1500, type=int)
    scatter_data = segmentation_engine.get_pca_scatter_sample(max_points=limit)
    return jsonify({"points": scatter_data})

@app.route("/api/radar-data", methods=["GET"])
def get_radar_data():
    """Returns normalized 8-dimensional radar chart metrics across all segments."""
    data = segmentation_engine.get_radar_data()
    return jsonify(data)

@app.route("/api/persona/<int:segment_id>", methods=["GET"])
def get_persona_details(segment_id):
    """Returns deep psychological persona and consumption DNA for a segment."""
    persona = llm_intelligence.get_persona(segment_id)
    return jsonify(persona)

@app.route("/api/generate-strategy", methods=["POST"])
def generate_strategy():
    """Generates real-time LLM marketing strategy, multi-channel copy, and attribution forecast."""
    data = request.get_json() or {}
    segment_id = int(data.get("segment_id", 0))
    objective = data.get("objective", "Convert Free to Premium")
    tone = data.get("tone", "Witty & Punchy")
    channel = data.get("channel", "Omni-Channel")
    
    result = llm_intelligence.generate_custom_strategy(
        segment_id=segment_id,
        business_objective=objective,
        campaign_tone=tone,
        channel_focus=channel
    )
    return jsonify(result)

@app.route("/api/simulate-campaign", methods=["POST"])
def simulate_campaign():
    """Simulates marketing budget performance comparing Demographics vs Behavioral AI."""
    data = request.get_json() or {}
    budget = float(data.get("budget", 100000.0))
    objective = data.get("objective", "Premium Acquisition")
    
    simulation = campaign_simulator.run_simulation(total_budget=budget, objective=objective)
    return jsonify(simulation)

@app.route("/api/user/<user_id>", methods=["GET"])
def lookup_user(user_id):
    """Looks up specific customer telemetry and calculates GMM soft cluster affinity breakdown."""
    user_data = segmentation_engine.find_user_by_id(user_id)
    if user_data is None:
        return jsonify({"error": f"User ID '{user_id}' not found in database."}), 404
    return jsonify(user_data)

@app.route("/api/download-report", methods=["GET"])
def download_report():
    """Serves the generated publication-grade PDF report."""
    if os.path.exists(REPORT_PATH):
        return send_file(
            REPORT_PATH,
            as_attachment=True,
            download_name="Spotify_AI_Customer_Segmentation_Report.pdf",
            mimetype="application/pdf"
        )
    else:
        return jsonify({"error": "Report PDF has not been generated yet. Please run generator."}), 404

@app.route("/api/export-csv", methods=["GET"])
def export_csv():
    """Exports segmented telemetry data as CSV download."""
    if os.path.exists(CSV_PATH):
        return send_file(
            CSV_PATH,
            as_attachment=True,
            download_name="spotify_behavioral_segments_telemetry.csv",
            mimetype="text/csv"
        )
    return jsonify({"error": "CSV file not found."}), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    print(f"\n[+] Launching Spotify Customer Intelligence Dashboard at http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
