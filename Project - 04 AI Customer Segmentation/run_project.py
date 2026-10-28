"""
Master Orchestrator for Project - 04: AI Customer Segmentation
Spotify Behavioral Customer Intelligence & LLM Strategy System
Executes telemetry generation, unsupervised behavioral clustering, LLM persona synthesis,
compiles the publication-grade executive PDF deliverable, and launches the interactive web dashboard.
"""

import os
import sys
import argparse

# Force UTF-8 stdout if supported
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import pandas as pd
from data.data_generator import generate_spotify_telemetry_data
from models.segmentation_engine import SpotifySegmentationEngine
from models.llm_intelligence import LLMCustomerIntelligence
from models.campaign_simulator import CampaignSimulator
from report.generate_pdf_report import generate_pdf

def main():
    parser = argparse.ArgumentParser(description="Master runner for Project - 04: AI Customer Segmentation")
    parser.add_argument("--no-server", action="store_true", help="Generate models, analytics, and PDF without launching web server.")
    parser.add_argument("--port", type=int, default=5003, help="Port to run Flask web dashboard on (default: 5003).")
    args = parser.parse_args()

    print("=" * 90)
    print(" PROJECT - 04: AI CUSTOMER SEGMENTATION")
    print(" Spotify Behavioral Customer Intelligence & LLM Differentiated Marketing Strategy System")
    print("=" * 90)

    data_dir = os.path.join(base_dir, "data")
    csv_path = os.path.join(data_dir, "spotify_customer_telemetry.csv")
    report_path = os.path.join(base_dir, "report", "Spotify_AI_Customer_Segmentation_Report.pdf")

    # Step 1: Telemetry Data Ingestion / Generation
    print("\n[Step 1/5] Ingesting Spotify customer interaction and transaction telemetry...")
    if not os.path.exists(csv_path):
        df, _ = generate_spotify_telemetry_data(data_dir, n_users=20000)
    else:
        df = pd.read_csv(csv_path)
    print(f"  [+] Ingested {len(df):,} subscriber records across 18 behavioral & acoustic dimensions.")
    print(f"  [+] Active Premium Penetration: {(df['monthly_spend'] > 0).mean() * 100:.1f}% | Avg Realized LTV: ${df['lifetime_value_ltv'].mean():.2f}")

    # Step 2: Unsupervised Machine Learning Clustering (K-Means++ & GMM)
    print("\n[Step 2/5] Running Unsupervised Behavioral Clustering & PCA Latent Space Decomposition...")
    engine = SpotifySegmentationEngine(df).fit()
    print(f"  [+] Unsupervised K-Means++ Converged: Silhouette Score = {engine.cluster_metrics['silhouette_score']:.4f}")
    print(f"  [+] Calinski-Harabasz Score: {engine.cluster_metrics['calinski_harabasz_score']} | Davies-Bouldin: {engine.cluster_metrics['davies_bouldin_score']}")
    print(f"  [+] PCA 3D Decomposition Total Explained Variance: {engine.cluster_metrics['total_explained_variance_3d'] * 100:.1f}%")
    
    print("\n  [Discovered Behavioral Archetypes]:")
    for s_id in range(engine.n_clusters):
        prof = engine.segment_profiles[s_id]
        print(f"    * Cluster {s_id}: {prof['name']:<48} | Share: {prof['share_pct']:>5.1f}% | LTV: ${prof['avg_ltv']:>6.2f} | Churn Risk: {prof['avg_churn_risk']*100:>4.1f}%")

    # Step 3: LLM Customer Intelligence & Persona Synthesis
    print("\n[Step 3/5] Chaining LLM Prompts to Synthesize Psychological Personas & Differentiated Strategies...")
    llm = LLMCustomerIntelligence()
    for s_id in range(engine.n_clusters):
        persona = llm.get_persona(s_id)
        short = persona["persona_name"].split("(")[0].strip()
        tagline = persona["tagline"]
        print(f"    * {short:<24}: \"{tagline}\"")

    # Step 4: Marketing Campaign ROI Simulation
    print("\n[Step 4/5] Executing Controlled Marketing Campaign Simulation ($100K Allocation)...")
    simulator = CampaignSimulator()
    sim = simulator.run_simulation(total_budget=100000.0, objective="Premium Acquisition")
    lift = sim["efficiency_lift"]
    print(f"  [+] Demographic Targeting: {sim['demographic_approach']['total_conversions']:,} conversions | CAC: ${sim['demographic_approach']['blended_cac_usd']:.2f} | Net Rev: ${sim['demographic_approach']['net_12m_revenue_usd']:,.0f}")
    print(f"  [+] Behavioral AI System:  {sim['behavioral_approach']['total_conversions']:,} conversions | CAC: ${sim['behavioral_approach']['blended_cac_usd']:.2f} | Net Rev: ${sim['behavioral_approach']['net_12m_revenue_usd']:,.0f}")
    print(f"  >>> Conversion Lift: +{lift['conversions_lift_pct']}% | CAC Reduction: -{lift['cac_reduction_pct']}% | Incremental Net 12-Mo Revenue: +${lift['net_revenue_lift_usd']:,.2f}")

    # Step 5: Executive Publication-Grade PDF Report
    print("\n[Step 5/5] Compiling Multi-Page Executive PDF Report...")
    generate_pdf(report_path)
    print(f"  [+] Verified PDF Report Artifact: {report_path}")

    print("\n" + "=" * 90)
    print(" PROJECT - 04 EXECUTION COMPLETE")
    print("=" * 90)

    # Server Launch
    if not args.no_server:
        print(f"\n[+] Booting Spotify Customer Intelligence Interactive Dashboard on port {args.port}...")
        print(f"    Access Local Dashboard URL: http://127.0.0.1:{args.port}\n")
        from app import app
        app.run(host="0.0.0.0", port=args.port, debug=False)

if __name__ == "__main__":
    main()
