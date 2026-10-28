"""
Advanced Unsupervised Behavioral Segmentation Engine
Project - 04: AI Customer Segmentation
Applies K-Means, GMM, PCA dimensionality reduction, Silhouette optimization, and RFM-E scoring
to discover authentic behavioral archetypes from Spotify interaction and transaction telemetry.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

class SpotifySegmentationEngine:
    """
    Production-grade behavioral clustering and intelligence engine for Spotify subscriber telemetry.
    """
    
    FEATURE_COLS = [
        "daily_streaming_minutes",
        "monthly_active_days",
        "skip_rate",
        "shuffle_ratio",
        "danceability",
        "energy",
        "valence_mood",
        "acousticness",
        "instrumentalness",
        "podcast_stream_pct",
        "audiobook_stream_pct",
        "discover_weekly_save_rate",
        "user_created_playlists",
        "library_saved_tracks",
        "social_shares_l90d",
        "stream_drop_l30d_pct",
        "lifetime_value_ltv",
        "churn_risk_score"
    ]
    
    SEGMENT_NAMES = {
        0: "The Sonic Loyalist (Focus & Hi-Fi Powerhouse)",
        1: "The Commute & Workout Hitseeker",
        2: "The Spoken-Word Explorer (Podcasts & Audiobooks)",
        3: "The Social Discovery Nomad (Viral & Trends)",
        4: "The Value Maximizer (Family & Multi-Device)",
        5: "The Disengaged Free-Tier Drifter (At-Risk)"
    }
    
    SEGMENT_SHORT_NAMES = {
        0: "Sonic Loyalist",
        1: "Workout Hitseeker",
        2: "Spoken-Word Explorer",
        3: "Discovery Nomad",
        4: "Value Maximizer",
        5: "Disengaged Drifter"
    }

    SEGMENT_COLORS = {
        0: "#1DB954",  # Spotify Green
        1: "#FF7A00",  # Energetic Orange
        2: "#9B51E0",  # Deep Purple
        3: "#00E5FF",  # Electric Cyan
        4: "#FFD600",  # Gold / Amber
        5: "#FF2A6D"   # Neon Crimson
    }

    def __init__(self, df: pd.DataFrame, n_clusters: int = 6):
        self.df = df.copy()
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=3, random_state=42)
        self.kmeans = None
        self.gmm = None
        self.scaled_features = None
        self.pca_coords = None
        self.cluster_metrics = {}
        self.segment_profiles = {}
        self.global_means = {}
        self.global_stds = {}
        self.k_evaluation = {}
        
    def fit(self):
        """Preprocesses features, evaluates clustering performance, fits K-Means and GMM models."""
        X = self.df[self.FEATURE_COLS].copy()
        
        # Log-transform heavy-tailed count features to avoid skew
        X["library_saved_tracks"] = np.log1p(X["library_saved_tracks"])
        X["user_created_playlists"] = np.log1p(X["user_created_playlists"])
        X["social_shares_l90d"] = np.log1p(X["social_shares_l90d"])
        X["daily_streaming_minutes"] = np.log1p(X["daily_streaming_minutes"])
        
        self.scaled_features = self.scaler.fit_transform(X)
        self.pca_coords = self.pca.fit_transform(self.scaled_features)
        
        # Fit K-Means
        self.kmeans = KMeans(n_clusters=self.n_clusters, init="k-means++", n_init=10, random_state=42)
        cluster_labels = self.kmeans.fit_predict(self.scaled_features)
        
        # Fit GMM for soft probabilistic assignment
        self.gmm = GaussianMixture(n_components=self.n_clusters, covariance_type="full", random_state=42)
        self.gmm.fit(self.scaled_features)
        soft_probs = self.gmm.predict_proba(self.scaled_features)
        
        # Align cluster IDs to our semantic canonical ordering based on dominant features:
        # 0: highest instrumentalness
        # 1: highest energy & skip rate
        # 2: highest podcast_stream_pct
        # 3: highest social shares & discover_weekly_save_rate
        # 4: highest multi-device/tenure
        # 5: lowest streaming minutes & highest churn risk
        aligned_labels = self._align_semantic_clusters(cluster_labels)
        self.df["cluster_id"] = aligned_labels
        self.df["segment_name"] = self.df["cluster_id"].map(self.SEGMENT_NAMES)
        self.df["segment_short"] = self.df["cluster_id"].map(self.SEGMENT_SHORT_NAMES)
        self.df["segment_color"] = self.df["cluster_id"].map(self.SEGMENT_COLORS)
        
        # Store PCA coordinates
        self.df["pca_x"] = np.round(self.pca_coords[:, 0], 3)
        self.df["pca_y"] = np.round(self.pca_coords[:, 1], 3)
        self.df["pca_z"] = np.round(self.pca_coords[:, 2], 3)
        
        # Calculate cluster evaluation metrics
        # Sample for fast silhouette calculation on large N
        sample_size = min(5000, len(self.scaled_features))
        indices = np.random.choice(len(self.scaled_features), sample_size, replace=False)
        sil_score = float(silhouette_score(self.scaled_features[indices], aligned_labels[indices]))
        ch_score = float(calinski_harabasz_score(self.scaled_features, aligned_labels))
        db_score = float(davies_bouldin_score(self.scaled_features, aligned_labels))
        
        self.cluster_metrics = {
            "silhouette_score": round(sil_score, 4),
            "calinski_harabasz_score": round(ch_score, 2),
            "davies_bouldin_score": round(db_score, 4),
            "explained_variance_pca": [round(float(v), 4) for v in self.pca.explained_variance_ratio_],
            "total_explained_variance_3d": round(float(np.sum(self.pca.explained_variance_ratio_[:3])), 4)
        }
        
        # Compute global feature statistics
        for col in self.FEATURE_COLS:
            self.global_means[col] = float(self.df[col].mean())
            self.global_stds[col] = float(self.df[col].std())
            
        # Build comprehensive profiles for each segment
        self._build_segment_profiles()
        
        # Evaluate k across range 3..8 for model evaluation section
        self._evaluate_k_range()
        
        return self

    def _align_semantic_clusters(self, raw_labels: np.ndarray) -> np.ndarray:
        """Maps unsupervised K-means labels to semantic Spotify segments based on dominant centroid attributes."""
        mapping = {}
        unassigned_targets = set(range(self.n_clusters))
        
        # Group stats
        temp_df = pd.DataFrame(self.scaled_features, columns=self.FEATURE_COLS)
        temp_df["raw_cluster"] = raw_labels
        means = temp_df.groupby("raw_cluster").mean()
        
        # 0: Highest instrumentalness
        c_inst = means["instrumentalness"].idxmax()
        mapping[c_inst] = 0
        unassigned_targets.discard(0)
        
        # 2: Highest podcast
        c_pod = means.drop(index=list(mapping.keys()))["podcast_stream_pct"].idxmax()
        mapping[c_pod] = 2
        unassigned_targets.discard(2)
        
        # 5: Lowest streaming minutes / highest churn risk
        c_drifter = means.drop(index=list(mapping.keys()))["churn_risk_score"].idxmax()
        mapping[c_drifter] = 5
        unassigned_targets.discard(5)
        
        # 3: Highest social shares
        c_social = means.drop(index=list(mapping.keys()))["social_shares_l90d"].idxmax()
        mapping[c_social] = 3
        unassigned_targets.discard(3)
        
        # 1: Highest energy
        c_workout = means.drop(index=list(mapping.keys()))["energy"].idxmax()
        mapping[c_workout] = 1
        unassigned_targets.discard(1)
        
        # Remaining is 4 (Value Maximizer)
        remaining_raw = [r for r in range(self.n_clusters) if r not in mapping]
        if remaining_raw:
            mapping[remaining_raw[0]] = 4
            
        aligned = np.array([mapping.get(label, label) for label in raw_labels])
        return aligned

    def _build_segment_profiles(self):
        """Constructs rich behavioral, financial, device, and acoustic profiles for each segment."""
        total_n = len(self.df)
        
        for seg_id in range(self.n_clusters):
            seg_df = self.df[self.df["cluster_id"] == seg_id]
            count = len(seg_df)
            share_pct = round((count / total_n) * 100, 2)
            
            # Means and Z-Scores
            feature_stats = {}
            for col in self.FEATURE_COLS:
                seg_mean = float(seg_df[col].mean())
                g_mean = self.global_means[col]
                g_std = self.global_stds[col] if self.global_stds[col] > 0 else 1.0
                z_score = (seg_mean - g_mean) / g_std
                feature_stats[col] = {
                    "mean": round(seg_mean, 2 if seg_mean > 1 else 4),
                    "global_mean": round(g_mean, 2 if g_mean > 1 else 4),
                    "z_score": round(float(z_score), 2)
                }
                
            # Top devices
            top_devices = seg_df["primary_device"].value_counts(normalize=True).head(3).to_dict()
            top_devices = {k: round(v * 100, 1) for k, v in top_devices.items()}
            
            # Top dayparts
            top_dayparts = seg_df["peak_daypart"].value_counts(normalize=True).head(2).to_dict()
            top_dayparts = {k: round(v * 100, 1) for k, v in top_dayparts.items()}
            
            # Subscription breakdown
            sub_dist = seg_df["subscription_tier"].value_counts(normalize=True).to_dict()
            sub_dist = {k: round(v * 100, 1) for k, v in sub_dist.items()}
            
            # Financial metrics
            avg_ltv = float(seg_df["lifetime_value_ltv"].mean())
            avg_monthly_spend = float(seg_df["monthly_spend"].mean())
            avg_churn_risk = float(seg_df["churn_risk_score"].mean())
            premium_pct = float((seg_df["monthly_spend"] > 0).mean() * 100)
            
            # Demographic spread within this behavioral segment (revealing the demographic flaw)
            demo_breakdown = seg_df["demographic_segment"].value_counts(normalize=True).to_dict()
            demo_breakdown = {k: round(v * 100, 1) for k, v in demo_breakdown.items()}

            self.segment_profiles[seg_id] = {
                "segment_id": seg_id,
                "name": self.SEGMENT_NAMES[seg_id],
                "short_name": self.SEGMENT_SHORT_NAMES[seg_id],
                "color": self.SEGMENT_COLORS[seg_id],
                "user_count": count,
                "share_pct": share_pct,
                "avg_ltv": round(avg_ltv, 2),
                "avg_monthly_spend": round(avg_monthly_spend, 2),
                "avg_churn_risk": round(avg_churn_risk, 3),
                "premium_penetration_pct": round(premium_pct, 1),
                "avg_daily_streaming_min": round(float(seg_df["daily_streaming_minutes"].mean()), 1),
                "avg_skip_rate": round(float(seg_df["skip_rate"].mean()), 3),
                "avg_podcast_share_pct": round(float(seg_df["podcast_stream_pct"].mean() * 100), 1),
                "avg_audiobook_share_pct": round(float(seg_df["audiobook_stream_pct"].mean() * 100), 1),
                "feature_stats": feature_stats,
                "top_devices": top_devices,
                "top_dayparts": top_dayparts,
                "subscription_distribution": sub_dist,
                "demographic_distribution": demo_breakdown
            }

    def _evaluate_k_range(self):
        """Evaluates k from 3 to 8 for academic and diagnostic transparency."""
        sub_X = self.scaled_features[:4000]
        results = {}
        for k in range(3, 9):
            km = KMeans(n_clusters=k, init="k-means++", n_init=5, random_state=42)
            preds = km.fit_predict(sub_X)
            sil = float(silhouette_score(sub_X, preds))
            ch = float(calinski_harabasz_score(sub_X, preds))
            db = float(davies_bouldin_score(sub_X, preds))
            inertia = float(km.inertia_)
            results[k] = {
                "k": k,
                "silhouette": round(sil, 4),
                "calinski_harabasz": round(ch, 1),
                "davies_bouldin": round(db, 4),
                "inertia": round(inertia, 1)
            }
        self.k_evaluation = results

    def get_radar_data(self) -> dict:
        """Formats multi-axis radar chart dimensions for comparing all 6 segments."""
        radar_dimensions = [
            ("Streaming Volume", "daily_streaming_minutes", 300.0),
            ("Low Skip Discipline", "skip_rate", -1.0), # Inverted so higher = better
            ("Acoustic Depth", "acousticness", 1.0),
            ("Dance & Energy", "energy", 1.0),
            ("Podcast & Spoken", "podcast_stream_pct", 0.7),
            ("Discovery Curiosity", "discover_weekly_save_rate", 0.8),
            ("Social Multiplier", "social_shares_l90d", 30.0),
            ("Economic Value (LTV)", "lifetime_value_ltv", 600.0)
        ]
        
        labels = [dim[0] for dim in radar_dimensions]
        datasets = []
        
        for seg_id in range(self.n_clusters):
            prof = self.segment_profiles[seg_id]
            values = []
            for name, col, max_val in radar_dimensions:
                raw_mean = prof["feature_stats"][col]["mean"]
                if max_val < 0:
                    # Inverted metric (like skip rate: 0.05 is 90% score, 0.60 is 10% score)
                    score = max(5, min(100, round((1.0 - (raw_mean / 0.65)) * 100)))
                else:
                    score = max(5, min(100, round((raw_mean / max_val) * 100)))
                values.append(score)
                
            datasets.append({
                "label": prof["short_name"],
                "data": values,
                "borderColor": prof["color"],
                "backgroundColor": prof["color"] + "33"  # 20% opacity
            })
            
        return {
            "labels": labels,
            "datasets": datasets
        }

    def get_pca_scatter_sample(self, max_points: int = 1500) -> list:
        """Returns sampled points with PCA coordinates and user telemetry for front-end 2D/3D charts."""
        sample_df = self.df.sample(n=min(max_points, len(self.df)), random_state=42)
        points = []
        for _, row in sample_df.iterrows():
            points.append({
                "user_id": row["user_id"],
                "x": float(row["pca_x"]),
                "y": float(row["pca_y"]),
                "z": float(row["pca_z"]),
                "cluster_id": int(row["cluster_id"]),
                "segment_name": row["segment_short"],
                "color": row["segment_color"],
                "streaming_mins": float(row["daily_streaming_minutes"]),
                "skip_rate": float(row["skip_rate"]),
                "podcast_pct": float(row["podcast_stream_pct"]),
                "tier": row["subscription_tier"],
                "device": row["primary_device"],
                "ltv": float(row["lifetime_value_ltv"])
            })
        return points

    def find_user_by_id(self, user_id: str) -> dict:
        """Looks up a specific Spotify user and computes detailed behavioral diagnostics."""
        match = self.df[self.df["user_id"].str.upper() == user_id.strip().upper()]
        if match.empty:
            return None
        row = match.iloc[0]
        seg_id = int(row["cluster_id"])
        prof = self.segment_profiles[seg_id]
        
        # Soft probabilities from GMM
        user_row = {
            c: row[c] if c not in ["library_saved_tracks", "user_created_playlists", "social_shares_l90d", "daily_streaming_minutes"]
            else np.log1p(row[c])
            for c in self.FEATURE_COLS
        }
        user_vector_df = pd.DataFrame([user_row], columns=self.FEATURE_COLS)
        user_vector = self.scaler.transform(user_vector_df)
        probs = self.gmm.predict_proba(user_vector)[0]
        
        affinity_breakdown = [
            {
                "segment_id": i,
                "name": self.SEGMENT_SHORT_NAMES[i],
                "color": self.SEGMENT_COLORS[i],
                "probability": round(float(probs[i]) * 100, 1)
            }
            for i in range(self.n_clusters)
        ]
        affinity_breakdown.sort(key=lambda x: x["probability"], reverse=True)
        
        return {
            "user_id": row["user_id"],
            "cluster_id": seg_id,
            "segment_name": self.SEGMENT_NAMES[seg_id],
            "segment_short": self.SEGMENT_SHORT_NAMES[seg_id],
            "segment_color": self.SEGMENT_COLORS[seg_id],
            "demographic_cohort": row["demographic_segment"],
            "age": int(row["age"]),
            "gender": row["gender"],
            "country": row["country"],
            "subscription_tier": row["subscription_tier"],
            "tenure_months": int(row["tenure_months"]),
            "monthly_spend": float(row["monthly_spend"]),
            "lifetime_value_ltv": float(row["lifetime_value_ltv"]),
            "daily_streaming_minutes": float(row["daily_streaming_minutes"]),
            "monthly_active_days": int(row["monthly_active_days"]),
            "skip_rate": float(row["skip_rate"]),
            "primary_device": row["primary_device"],
            "peak_daypart": row["peak_daypart"],
            "podcast_stream_pct": float(row["podcast_stream_pct"]),
            "audiobook_stream_pct": float(row["audiobook_stream_pct"]),
            "churn_risk_score": float(row["churn_risk_score"]),
            "library_saved_tracks": int(row["library_saved_tracks"]),
            "user_created_playlists": int(row["user_created_playlists"]),
            "social_shares_l90d": int(row["social_shares_l90d"]),
            "stream_drop_l30d_pct": float(row["stream_drop_l30d_pct"]),
            "offline_downloads_active": bool(row["offline_downloads_active"]),
            "affinity_breakdown": affinity_breakdown,
            "segment_context": prof
        }
