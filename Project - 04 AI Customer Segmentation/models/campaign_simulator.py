"""
Demographic vs. Behavioral Campaign Simulation Engine
Project - 04: AI Customer Segmentation
Simulates marketing campaign performance comparing Spotify's legacy demographic targeting
versus the proposed LLM-powered behavioral customer intelligence system.
"""

from typing import Dict, Any

class CampaignSimulator:
    """
    Simulates marketing efficiency, conversion lift, Customer Acquisition Cost (CAC),
    and 12-month net revenue comparing Traditional Demographic Segmentation vs
    LLM-powered Behavioral Segmentation.
    """
    
    DEMOGRAPHIC_BENCHMARKS = {
        "18-24 (Gen-Z)": {"budget_share": 0.35, "cpm": 14.50, "ctr": 0.018, "cvr": 0.012, "cac": 46.50, "churn_m3": 0.34, "ltv_12m": 88.00},
        "25-34 (Millennial)": {"budget_share": 0.38, "cpm": 16.00, "ctr": 0.016, "cvr": 0.011, "cac": 49.20, "churn_m3": 0.30, "ltv_12m": 112.00},
        "35-50 (Gen-X)": {"budget_share": 0.18, "cpm": 15.20, "ctr": 0.014, "cvr": 0.009, "cac": 54.80, "churn_m3": 0.26, "ltv_12m": 125.00},
        "51+ (Boomer)": {"budget_share": 0.09, "cpm": 13.80, "ctr": 0.011, "cvr": 0.007, "cac": 62.10, "churn_m3": 0.38, "ltv_12m": 72.00}
    }
    
    BEHAVIORAL_OPTIMIZED = {
        0: {"name": "Sonic Loyalist", "budget_share": 0.15, "cpm": 17.50, "ctr": 0.038, "cvr": 0.041, "cac": 18.20, "churn_m3": 0.06, "ltv_12m": 165.00},
        1: {"name": "Workout Hitseeker", "budget_share": 0.28, "cpm": 14.80, "ctr": 0.046, "cvr": 0.036, "cac": 16.80, "churn_m3": 0.18, "ltv_12m": 124.00},
        2: {"name": "Spoken-Word Explorer", "budget_share": 0.18, "cpm": 18.20, "ctr": 0.034, "cvr": 0.038, "cac": 21.40, "churn_m3": 0.09, "ltv_12m": 182.00},
        3: {"name": "Discovery Nomad", "budget_share": 0.22, "cpm": 13.50, "ctr": 0.052, "cvr": 0.039, "cac": 15.60, "churn_m3": 0.16, "ltv_12m": 138.00},
        4: {"name": "Value Maximizer", "budget_share": 0.10, "cpm": 16.00, "ctr": 0.028, "cvr": 0.032, "cac": 24.10, "churn_m3": 0.08, "ltv_12m": 198.00},
        5: {"name": "Disengaged Drifter", "budget_share": 0.07, "cpm": 9.50, "ctr": 0.024, "cvr": 0.022, "cac": 28.50, "churn_m3": 0.28, "ltv_12m": 58.00}
    }

    def __init__(self):
        pass

    def run_simulation(self, total_budget: float = 100000.0, objective: str = "Premium Acquisition") -> Dict[str, Any]:
        """Runs side-by-side financial simulation for a given marketing budget."""
        
        # 1. Demographic Simulation
        demo_results = []
        demo_total_conversions = 0
        demo_total_impressions = 0
        demo_total_clicks = 0
        demo_total_12m_revenue = 0.0
        
        for seg_name, p in self.DEMOGRAPHIC_BENCHMARKS.items():
            seg_budget = total_budget * p["budget_share"]
            impressions = int((seg_budget / p["cpm"]) * 1000)
            clicks = int(impressions * p["ctr"])
            conversions = int(clicks * p["cvr"])
            net_12m = conversions * (1.0 - p["churn_m3"]) * p["ltv_12m"]
            cac = seg_budget / conversions if conversions > 0 else 0
            
            demo_total_impressions += impressions
            demo_total_clicks += clicks
            demo_total_conversions += conversions
            demo_total_12m_revenue += net_12m
            
            demo_results.append({
                "segment": seg_name,
                "spend": round(seg_budget, 2),
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
                "cac": round(cac, 2),
                "net_12m_revenue": round(net_12m, 2)
            })

        demo_blended_cac = round(total_budget / demo_total_conversions, 2) if demo_total_conversions > 0 else 0
        demo_blended_ctr = round((demo_total_clicks / demo_total_impressions) * 100, 2)
        demo_blended_cvr = round((demo_total_conversions / demo_total_clicks) * 100, 2)
        demo_roas = round(demo_total_12m_revenue / total_budget, 2)

        # 2. Behavioral Simulation
        behav_results = []
        behav_total_conversions = 0
        behav_total_impressions = 0
        behav_total_clicks = 0
        behav_total_12m_revenue = 0.0
        
        for seg_id, p in self.BEHAVIORAL_OPTIMIZED.items():
            seg_budget = total_budget * p["budget_share"]
            impressions = int((seg_budget / p["cpm"]) * 1000)
            clicks = int(impressions * p["ctr"])
            conversions = int(clicks * p["cvr"])
            net_12m = conversions * (1.0 - p["churn_m3"]) * p["ltv_12m"]
            cac = seg_budget / conversions if conversions > 0 else 0
            
            behav_total_impressions += impressions
            behav_total_clicks += clicks
            behav_total_conversions += conversions
            behav_total_12m_revenue += net_12m
            
            behav_results.append({
                "segment": p["name"],
                "spend": round(seg_budget, 2),
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
                "cac": round(cac, 2),
                "net_12m_revenue": round(net_12m, 2)
            })

        behav_blended_cac = round(total_budget / behav_total_conversions, 2) if behav_total_conversions > 0 else 0
        behav_blended_ctr = round((behav_total_clicks / behav_total_impressions) * 100, 2)
        behav_blended_cvr = round((behav_total_conversions / behav_total_clicks) * 100, 2)
        behav_roas = round(behav_total_12m_revenue / total_budget, 2)

        # Lift & Improvement Calculations
        conversions_lift_pct = round(((behav_total_conversions - demo_total_conversions) / demo_total_conversions) * 100, 1)
        cac_reduction_pct = round(((demo_blended_cac - behav_blended_cac) / demo_blended_cac) * 100, 1)
        revenue_lift_usd = round(behav_total_12m_revenue - demo_total_12m_revenue, 2)
        revenue_lift_pct = round(((behav_total_12m_revenue - demo_total_12m_revenue) / demo_total_12m_revenue) * 100, 1)

        return {
            "budget": total_budget,
            "objective": objective,
            "demographic_approach": {
                "total_impressions": demo_total_impressions,
                "total_clicks": demo_total_clicks,
                "blended_ctr_pct": demo_blended_ctr,
                "total_conversions": demo_total_conversions,
                "blended_cvr_pct": demo_blended_cvr,
                "blended_cac_usd": demo_blended_cac,
                "net_12m_revenue_usd": round(demo_total_12m_revenue, 2),
                "roas": demo_roas,
                "breakdown": demo_results
            },
            "behavioral_approach": {
                "total_impressions": behav_total_impressions,
                "total_clicks": behav_total_clicks,
                "blended_ctr_pct": behav_blended_ctr,
                "total_conversions": behav_total_conversions,
                "blended_cvr_pct": behav_blended_cvr,
                "blended_cac_usd": behav_blended_cac,
                "net_12m_revenue_usd": round(behav_total_12m_revenue, 2),
                "roas": behav_roas,
                "breakdown": behav_results
            },
            "efficiency_lift": {
                "conversions_lift_pct": conversions_lift_pct,
                "cac_reduction_pct": cac_reduction_pct,
                "net_revenue_lift_usd": revenue_lift_usd,
                "net_revenue_lift_pct": revenue_lift_pct,
                "roas_multiplier": round(behav_roas / demo_roas, 2) if demo_roas > 0 else 0
            }
        }
