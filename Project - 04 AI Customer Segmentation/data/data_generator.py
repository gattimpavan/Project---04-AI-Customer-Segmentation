"""
Synthetic Spotify Customer Telemetry & Transaction Data Generator
Project - 04: AI Customer Segmentation
Generates high-dimensional behavioral interaction and transactional data for 20,000 Spotify users,
mirroring authentic streaming telemetry, acoustic profiles, device habits, and monetization states.
"""

import os
import json
import numpy as np
import pandas as pd

def generate_spotify_telemetry_data(output_dir: str, n_users: int = 20000, seed: int = 42) -> tuple:
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    csv_file = os.path.join(output_dir, "spotify_customer_telemetry.csv")
    meta_file = os.path.join(output_dir, "demographic_baseline.json")
    
    # 1. User IDs
    user_ids = [f"SPOT-{100000 + i}" for i in range(n_users)]
    
    # 2. Traditional Demographic Attributes (to compare against behavioral)
    # Spotify age distribution skew: Gen Z (18-24: 35%), Millennials (25-34: 38%), Gen X (35-50: 18%), Boomers (51+: 9%)
    age_brackets = ["18-24 (Gen-Z)", "25-34 (Millennial)", "35-50 (Gen-X)", "51+ (Boomer)"]
    demographic_segments = np.random.choice(age_brackets, size=n_users, p=[0.35, 0.38, 0.18, 0.09])
    
    ages = []
    for seg in demographic_segments:
        if seg.startswith("18"):
            ages.append(np.random.randint(18, 25))
        elif seg.startswith("25"):
            ages.append(np.random.randint(25, 35))
        elif seg.startswith("35"):
            ages.append(np.random.randint(35, 51))
        else:
            ages.append(np.random.randint(51, 72))
            
    genders = np.random.choice(["Female", "Male", "Non-Binary"], size=n_users, p=[0.48, 0.47, 0.05])
    countries = np.random.choice(
        ["United States", "United Kingdom", "Germany", "Brazil", "India", "Mexico", "France", "Japan"],
        size=n_users,
        p=[0.28, 0.14, 0.12, 0.12, 0.14, 0.08, 0.07, 0.05]
    )

    # 3. Ground-truth Archetypes (latent behavioral generators to inject real multi-modal clusters)
    # The system will perform unsupervised clustering, but we seed realistic underlying behavior:
    # 0: Sonic Loyalist (Deep Focus & Hi-Fi Enthusiast) ~ 18%
    # 1: Commute & Workout Hitseeker ~ 24%
    # 2: Spoken-Word & Podcast Intellectual ~ 16%
    # 3: Social Discovery Nomad (Viral & Trends) ~ 20%
    # 4: Value Maximizer (Shared / Family Streamer) ~ 12%
    # 5: Disengaged Free-Tier Drifter (At-Risk) ~ 10%
    archetype_probs = [0.18, 0.24, 0.16, 0.20, 0.12, 0.10]
    archetypes = np.random.choice(6, size=n_users, p=archetype_probs)
    
    # Pre-allocate arrays
    daily_streaming_mins = np.zeros(n_users)
    monthly_active_days = np.zeros(n_users, dtype=int)
    skip_rate = np.zeros(n_users)
    shuffle_ratio = np.zeros(n_users)
    danceability = np.zeros(n_users)
    energy = np.zeros(n_users)
    valence_mood = np.zeros(n_users)
    acousticness = np.zeros(n_users)
    instrumentalness = np.zeros(n_users)
    podcast_stream_pct = np.zeros(n_users)
    audiobook_stream_pct = np.zeros(n_users)
    discover_weekly_save_rate = np.zeros(n_users)
    user_created_playlists = np.zeros(n_users, dtype=int)
    library_saved_tracks = np.zeros(n_users, dtype=int)
    social_shares_l90d = np.zeros(n_users, dtype=int)
    stream_drop_l30d_pct = np.zeros(n_users)
    offline_downloads_active = np.zeros(n_users, dtype=int)
    subscription_tier = []
    tenure_months = np.zeros(n_users, dtype=int)
    primary_device = []
    peak_daypart = []
    payment_failures_l12m = np.zeros(n_users, dtype=int)
    discount_promo_user = np.zeros(n_users, dtype=int)
    ad_audio_completion_rate = np.zeros(n_users)
    ad_click_rate = np.zeros(n_users)
    
    # Devices & dayparts pools
    device_pool = ["Mobile iOS", "Mobile Android", "Desktop App", "Smart Speaker", "Connected Car", "Smart TV"]
    daypart_pool = ["Morning Commute (6-9 AM)", "Deep Work / Study (9 AM-5 PM)", "Evening Workout (5-8 PM)", "Night Chill (8-11 PM)", "Late Night (11 PM-5 AM)"]
    
    for i in range(n_users):
        arch = archetypes[i]
        
        if arch == 0:  # Sonic Loyalist (Focus & Hi-Fi Powerhouse)
            daily_streaming_mins[i] = np.random.normal(275, 45)
            monthly_active_days[i] = np.random.randint(26, 31)
            skip_rate[i] = np.clip(np.random.normal(0.06, 0.03), 0.01, 0.18)
            shuffle_ratio[i] = np.clip(np.random.normal(0.18, 0.08), 0.02, 0.40)
            danceability[i] = np.clip(np.random.normal(0.42, 0.12), 0.10, 0.70)
            energy[i] = np.clip(np.random.normal(0.48, 0.12), 0.15, 0.75)
            valence_mood[i] = np.clip(np.random.normal(0.45, 0.14), 0.10, 0.80)
            acousticness[i] = np.clip(np.random.normal(0.65, 0.15), 0.20, 0.98)
            instrumentalness[i] = np.clip(np.random.normal(0.72, 0.15), 0.30, 0.99)
            podcast_stream_pct[i] = np.clip(np.random.normal(0.08, 0.05), 0.0, 0.25)
            audiobook_stream_pct[i] = np.clip(np.random.normal(0.04, 0.03), 0.0, 0.15)
            discover_weekly_save_rate[i] = np.clip(np.random.normal(0.55, 0.12), 0.2, 0.95)
            user_created_playlists[i] = np.random.randint(15, 60)
            library_saved_tracks[i] = np.random.randint(1800, 9500)
            social_shares_l90d[i] = np.random.randint(1, 10)
            stream_drop_l30d_pct[i] = np.random.normal(0.02, 0.06)
            offline_downloads_active[i] = 1 if np.random.rand() < 0.88 else 0
            subscription_tier.append("Premium Individual" if np.random.rand() < 0.85 else "Premium Duo")
            tenure_months[i] = np.random.randint(24, 72)
            primary_device.append(np.random.choice(["Desktop App", "Smart Speaker", "Mobile iOS"], p=[0.48, 0.32, 0.20]))
            peak_daypart.append(np.random.choice(["Deep Work / Study (9 AM-5 PM)", "Night Chill (8-11 PM)"], p=[0.70, 0.30]))
            payment_failures_l12m[i] = 0 if np.random.rand() < 0.94 else 1
            discount_promo_user[i] = 0
            ad_audio_completion_rate[i] = 1.0  # Paid
            ad_click_rate[i] = 0.0

        elif arch == 1:  # Commute & Workout Hitseeker
            daily_streaming_mins[i] = np.random.normal(110, 30)
            monthly_active_days[i] = np.random.randint(20, 29)
            skip_rate[i] = np.clip(np.random.normal(0.38, 0.09), 0.15, 0.65)
            shuffle_ratio[i] = np.clip(np.random.normal(0.72, 0.12), 0.40, 0.95)
            danceability[i] = np.clip(np.random.normal(0.78, 0.09), 0.50, 0.98)
            energy[i] = np.clip(np.random.normal(0.82, 0.08), 0.55, 0.99)
            valence_mood[i] = np.clip(np.random.normal(0.68, 0.12), 0.35, 0.95)
            acousticness[i] = np.clip(np.random.normal(0.18, 0.09), 0.02, 0.45)
            instrumentalness[i] = np.clip(np.random.normal(0.08, 0.05), 0.0, 0.25)
            podcast_stream_pct[i] = np.clip(np.random.normal(0.06, 0.04), 0.0, 0.18)
            audiobook_stream_pct[i] = 0.01
            discover_weekly_save_rate[i] = np.clip(np.random.normal(0.22, 0.10), 0.02, 0.50)
            user_created_playlists[i] = np.random.randint(2, 12)
            library_saved_tracks[i] = np.random.randint(250, 1600)
            social_shares_l90d[i] = np.random.randint(2, 15)
            stream_drop_l30d_pct[i] = np.random.normal(0.0, 0.09)
            offline_downloads_active[i] = 1 if np.random.rand() < 0.65 else 0
            subscription_tier.append(np.random.choice(["Premium Individual", "Free (Ad-Supported)", "Premium Student"], p=[0.55, 0.32, 0.13]))
            tenure_months[i] = np.random.randint(6, 36)
            primary_device.append(np.random.choice(["Mobile iOS", "Connected Car", "Mobile Android"], p=[0.50, 0.32, 0.18]))
            peak_daypart.append(np.random.choice(["Morning Commute (6-9 AM)", "Evening Workout (5-8 PM)"], p=[0.48, 0.52]))
            payment_failures_l12m[i] = np.random.choice([0, 1, 2], p=[0.82, 0.14, 0.04])
            discount_promo_user[i] = 1 if np.random.rand() < 0.45 else 0
            ad_audio_completion_rate[i] = np.clip(np.random.normal(0.68, 0.15), 0.3, 0.95)
            ad_click_rate[i] = np.clip(np.random.normal(0.028, 0.012), 0.005, 0.075)

        elif arch == 2:  # Spoken-Word & Podcast Intellectual
            daily_streaming_mins[i] = np.random.normal(190, 40)
            monthly_active_days[i] = np.random.randint(24, 31)
            skip_rate[i] = np.clip(np.random.normal(0.11, 0.04), 0.02, 0.25)
            shuffle_ratio[i] = np.clip(np.random.normal(0.10, 0.05), 0.01, 0.25)
            danceability[i] = np.clip(np.random.normal(0.35, 0.12), 0.10, 0.60)
            energy[i] = np.clip(np.random.normal(0.42, 0.12), 0.15, 0.68)
            valence_mood[i] = np.clip(np.random.normal(0.48, 0.14), 0.15, 0.80)
            acousticness[i] = np.clip(np.random.normal(0.45, 0.15), 0.10, 0.85)
            instrumentalness[i] = np.clip(np.random.normal(0.20, 0.12), 0.01, 0.55)
            podcast_stream_pct[i] = np.clip(np.random.normal(0.56, 0.11), 0.35, 0.85)
            audiobook_stream_pct[i] = np.clip(np.random.normal(0.18, 0.08), 0.03, 0.45)
            discover_weekly_save_rate[i] = np.clip(np.random.normal(0.30, 0.10), 0.05, 0.60)
            user_created_playlists[i] = np.random.randint(4, 20)
            library_saved_tracks[i] = np.random.randint(600, 3200)
            social_shares_l90d[i] = np.random.randint(1, 8)
            stream_drop_l30d_pct[i] = np.random.normal(0.03, 0.07)
            offline_downloads_active[i] = 1 if np.random.rand() < 0.82 else 0
            subscription_tier.append("Premium Individual" if np.random.rand() < 0.78 else "Premium Duo")
            tenure_months[i] = np.random.randint(18, 60)
            primary_device.append(np.random.choice(["Mobile iOS", "Smart Speaker", "Desktop App"], p=[0.55, 0.28, 0.17]))
            peak_daypart.append(np.random.choice(["Morning Commute (6-9 AM)", "Night Chill (8-11 PM)", "Deep Work / Study (9 AM-5 PM)"], p=[0.40, 0.40, 0.20]))
            payment_failures_l12m[i] = 0 if np.random.rand() < 0.95 else 1
            discount_promo_user[i] = 0 if np.random.rand() < 0.85 else 1
            ad_audio_completion_rate[i] = 0.92
            ad_click_rate[i] = 0.012

        elif arch == 3:  # Social Discovery Nomad (Viral & Trends)
            daily_streaming_mins[i] = np.random.normal(160, 35)
            monthly_active_days[i] = np.random.randint(23, 30)
            skip_rate[i] = np.clip(np.random.normal(0.28, 0.07), 0.12, 0.50)
            shuffle_ratio[i] = np.clip(np.random.normal(0.65, 0.14), 0.30, 0.92)
            danceability[i] = np.clip(np.random.normal(0.74, 0.10), 0.45, 0.95)
            energy[i] = np.clip(np.random.normal(0.70, 0.11), 0.40, 0.95)
            valence_mood[i] = np.clip(np.random.normal(0.72, 0.11), 0.40, 0.98)
            acousticness[i] = np.clip(np.random.normal(0.28, 0.12), 0.05, 0.60)
            instrumentalness[i] = np.clip(np.random.normal(0.12, 0.08), 0.0, 0.35)
            podcast_stream_pct[i] = np.clip(np.random.normal(0.14, 0.07), 0.0, 0.30)
            audiobook_stream_pct[i] = 0.02
            discover_weekly_save_rate[i] = np.clip(np.random.normal(0.68, 0.11), 0.35, 0.98)
            user_created_playlists[i] = np.random.randint(18, 55)
            library_saved_tracks[i] = np.random.randint(1400, 5500)
            social_shares_l90d[i] = np.random.randint(12, 45)  # Heavy Jam & IG stories
            stream_drop_l30d_pct[i] = np.random.normal(0.05, 0.08)
            offline_downloads_active[i] = 1 if np.random.rand() < 0.72 else 0
            subscription_tier.append(np.random.choice(["Premium Student", "Premium Individual", "Free (Ad-Supported)"], p=[0.42, 0.38, 0.20]))
            tenure_months[i] = np.random.randint(8, 38)
            primary_device.append(np.random.choice(["Mobile iOS", "Mobile Android"], p=[0.70, 0.30]))
            peak_daypart.append(np.random.choice(["Late Night (11 PM-5 AM)", "Evening Workout (5-8 PM)"], p=[0.58, 0.42]))
            payment_failures_l12m[i] = np.random.choice([0, 1], p=[0.88, 0.12])
            discount_promo_user[i] = 1 if np.random.rand() < 0.60 else 0
            ad_audio_completion_rate[i] = np.clip(np.random.normal(0.74, 0.14), 0.4, 0.95)
            ad_click_rate[i] = np.clip(np.random.normal(0.045, 0.015), 0.01, 0.09)

        elif arch == 4:  # Value Maximizer (Family & Multi-Device Streamer)
            daily_streaming_mins[i] = np.random.normal(210, 45)
            monthly_active_days[i] = np.random.randint(25, 31)
            skip_rate[i] = np.clip(np.random.normal(0.16, 0.05), 0.04, 0.32)
            shuffle_ratio[i] = np.clip(np.random.normal(0.45, 0.12), 0.15, 0.75)
            danceability[i] = np.clip(np.random.normal(0.58, 0.12), 0.25, 0.85)
            energy[i] = np.clip(np.random.normal(0.55, 0.12), 0.20, 0.85)
            valence_mood[i] = np.clip(np.random.normal(0.60, 0.12), 0.25, 0.90)
            acousticness[i] = np.clip(np.random.normal(0.38, 0.14), 0.10, 0.75)
            instrumentalness[i] = np.clip(np.random.normal(0.18, 0.10), 0.02, 0.50)
            podcast_stream_pct[i] = np.clip(np.random.normal(0.18, 0.08), 0.02, 0.38)
            audiobook_stream_pct[i] = np.clip(np.random.normal(0.08, 0.05), 0.0, 0.22)
            discover_weekly_save_rate[i] = np.clip(np.random.normal(0.38, 0.12), 0.10, 0.70)
            user_created_playlists[i] = np.random.randint(10, 35)
            library_saved_tracks[i] = np.random.randint(1200, 4800)
            social_shares_l90d[i] = np.random.randint(3, 14)
            stream_drop_l30d_pct[i] = np.random.normal(0.01, 0.05)
            offline_downloads_active[i] = 1 if np.random.rand() < 0.90 else 0
            subscription_tier.append("Premium Family" if np.random.rand() < 0.82 else "Premium Duo")
            tenure_months[i] = np.random.randint(28, 80)
            primary_device.append(np.random.choice(["Smart Speaker", "Smart TV", "Connected Car"], p=[0.42, 0.30, 0.28]))
            peak_daypart.append(np.random.choice(["Night Chill (8-11 PM)", "Deep Work / Study (9 AM-5 PM)"], p=[0.55, 0.45]))
            payment_failures_l12m[i] = 0 if np.random.rand() < 0.92 else 1
            discount_promo_user[i] = 0
            ad_audio_completion_rate[i] = 1.0
            ad_click_rate[i] = 0.0

        else:  # Disengaged Free-Tier Drifter (At-Risk Churn)
            daily_streaming_mins[i] = np.clip(np.random.normal(32, 14), 6, 75)
            monthly_active_days[i] = np.random.randint(3, 14)
            skip_rate[i] = np.clip(np.random.normal(0.52, 0.11), 0.28, 0.88)
            shuffle_ratio[i] = np.clip(np.random.normal(0.85, 0.10), 0.50, 0.99)
            danceability[i] = np.clip(np.random.normal(0.65, 0.14), 0.30, 0.90)
            energy[i] = np.clip(np.random.normal(0.68, 0.13), 0.35, 0.92)
            valence_mood[i] = np.clip(np.random.normal(0.50, 0.16), 0.15, 0.85)
            acousticness[i] = np.clip(np.random.normal(0.22, 0.12), 0.02, 0.55)
            instrumentalness[i] = np.clip(np.random.normal(0.05, 0.04), 0.0, 0.20)
            podcast_stream_pct[i] = np.clip(np.random.normal(0.04, 0.03), 0.0, 0.15)
            audiobook_stream_pct[i] = 0.0
            discover_weekly_save_rate[i] = np.clip(np.random.normal(0.08, 0.05), 0.0, 0.25)
            user_created_playlists[i] = np.random.randint(0, 4)
            library_saved_tracks[i] = np.random.randint(25, 350)
            social_shares_l90d[i] = 0 if np.random.rand() < 0.80 else 1
            stream_drop_l30d_pct[i] = np.random.normal(-0.35, 0.12)  # Steep drop in listening
            offline_downloads_active[i] = 0
            subscription_tier.append("Free (Ad-Supported)")
            tenure_months[i] = np.random.randint(2, 18)
            primary_device.append(np.random.choice(["Mobile Android", "Mobile iOS"], p=[0.65, 0.35]))
            peak_daypart.append(np.random.choice(["Late Night (11 PM-5 AM)", "Morning Commute (6-9 AM)"], p=[0.55, 0.45]))
            payment_failures_l12m[i] = np.random.choice([0, 1, 2], p=[0.70, 0.22, 0.08])
            discount_promo_user[i] = 1 if np.random.rand() < 0.65 else 0
            ad_audio_completion_rate[i] = np.clip(np.random.normal(0.42, 0.16), 0.10, 0.75)
            ad_click_rate[i] = np.clip(np.random.normal(0.011, 0.007), 0.001, 0.035)

    # 4. Financial & Monetization Calculations
    tier_pricing = {
        "Free (Ad-Supported)": 0.00,
        "Premium Student": 5.99,
        "Premium Individual": 10.99,
        "Premium Duo": 14.99,
        "Premium Family": 16.99
    }
    
    monthly_spend = np.array([tier_pricing[t] for t in subscription_tier])
    # Approximate ad revenue generated by free users: ~$0.0035 per ad * ads listened
    ad_revenue_monthly = np.where(
        monthly_spend == 0,
        np.clip(daily_streaming_mins / 15 * 30 * 0.004 * ad_audio_completion_rate, 0.20, 2.80),
        0.0
    )
    
    # Cumulative LTV
    lifetime_value_ltv = np.round(
        (monthly_spend * tenure_months) + (ad_revenue_monthly * tenure_months),
        2
    )

    # Churn / Disengagement Risk Score (0.0 to 1.0)
    # Modeled from streaming drop, skip rate, active days, and payment failures
    churn_risk_score = np.clip(
        0.35 * (1.0 - (monthly_active_days / 30.0)) +
        0.25 * skip_rate +
        0.25 * np.clip(-stream_drop_l30d_pct, 0.0, 1.0) +
        0.15 * (payment_failures_l12m / 3.0),
        0.01,
        0.98
    )

    # Build DataFrame
    df = pd.DataFrame({
        "user_id": user_ids,
        "demographic_segment": demographic_segments,
        "age": ages,
        "gender": genders,
        "country": countries,
        "subscription_tier": subscription_tier,
        "tenure_months": tenure_months,
        "monthly_spend": np.round(monthly_spend, 2),
        "ad_revenue_monthly": np.round(ad_revenue_monthly, 2),
        "lifetime_value_ltv": lifetime_value_ltv,
        "daily_streaming_minutes": np.round(daily_streaming_mins, 1),
        "monthly_active_days": monthly_active_days,
        "peak_daypart": peak_daypart,
        "primary_device": primary_device,
        "skip_rate": np.round(skip_rate, 4),
        "shuffle_ratio": np.round(shuffle_ratio, 4),
        "danceability": np.round(danceability, 4),
        "energy": np.round(energy, 4),
        "valence_mood": np.round(valence_mood, 4),
        "acousticness": np.round(acousticness, 4),
        "instrumentalness": np.round(instrumentalness, 4),
        "podcast_stream_pct": np.round(podcast_stream_pct, 4),
        "audiobook_stream_pct": np.round(audiobook_stream_pct, 4),
        "discover_weekly_save_rate": np.round(discover_weekly_save_rate, 4),
        "user_created_playlists": user_created_playlists,
        "library_saved_tracks": library_saved_tracks,
        "social_shares_l90d": social_shares_l90d,
        "stream_drop_l30d_pct": np.round(stream_drop_l30d_pct, 4),
        "offline_downloads_active": offline_downloads_active,
        "payment_failures_l12m": payment_failures_l12m,
        "discount_promo_user": discount_promo_user,
        "ad_audio_completion_rate": np.round(ad_audio_completion_rate, 4),
        "ad_click_rate": np.round(ad_click_rate, 4),
        "churn_risk_score": np.round(churn_risk_score, 4),
        "ground_truth_archetype": archetypes
    })

    # Save CSV
    df.to_csv(csv_file, index=False)

    # 5. Build Demographic Baseline Comparison Benchmark
    # Demonstrating the problem with Spotify's traditional demographic targeting:
    # Under demographics, each age cohort has huge intra-group variance and low response rates.
    demo_perf = {}
    for seg in age_brackets:
        sub = df[df["demographic_segment"] == seg]
        demo_perf[seg] = {
            "user_count": int(len(sub)),
            "pct_of_base": round(len(sub) / n_users * 100, 1),
            "avg_daily_streaming_min": round(float(sub["daily_streaming_minutes"].mean()), 1),
            "avg_skip_rate": round(float(sub["skip_rate"].mean()), 3),
            "avg_podcast_pct": round(float(sub["podcast_stream_pct"].mean()), 3),
            "avg_ltv": round(float(sub["lifetime_value_ltv"].mean()), 2),
            "premium_penetration_pct": round(float((sub["monthly_spend"] > 0).mean() * 100), 1),
            "traditional_campaign_ctr": round(float(np.random.uniform(0.014, 0.022)), 4),
            "traditional_campaign_cvr": round(float(np.random.uniform(0.008, 0.014)), 4),
            "estimated_cac_usd": round(float(np.random.uniform(42.0, 58.0)), 2),
            "intra_cluster_variance_score": 0.84  # High variance = blunt targeting
        }
    
    baseline_metadata = {
        "dataset_name": "Spotify Customer Interaction & Transaction Telemetry",
        "total_records": n_users,
        "features_count": len(df.columns),
        "demographic_segments": demo_perf,
        "traditional_demographic_flaws": [
            "Conflates high-skip casuals with deep-focus audiophiles in the same 18-24 bracket.",
            "Misses prime podcast & audiobook monetizable listeners grouped only by age/gender.",
            "Wastes ad budget pushing generic Premium trials to at-risk disengaged users who need habit reactivation first.",
            "Fails to identify multi-device family plan candidates hidden within 25-34 and 35-50 demographics.",
            "Results in low campaign conversion (~1.1% CVR) and inflated Customer Acquisition Cost ($49.50 avg CAC)."
        ]
    }

    with open(meta_file, "w") as f:
        json.dump(baseline_metadata, f, indent=2)

    return df, baseline_metadata

if __name__ == "__main__":
    out = os.path.dirname(os.path.abspath(__file__))
    df, meta = generate_spotify_telemetry_data(out, n_users=20000)
    print(f"Generated {len(df):,} records successfully saved to {out}")
