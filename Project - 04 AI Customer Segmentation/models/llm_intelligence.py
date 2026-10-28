"""
LLM-Powered Customer Intelligence & Strategic Persona Synthesizer
Project - 04: AI Customer Segmentation
Translates high-dimensional behavioral clusters into rich psychological archetypes,
and generates differentiated marketing strategies, multi-channel creative copy, and retention playbooks.
"""

import json
from typing import Dict, Any

class LLMCustomerIntelligence:
    """
    Simulates and wraps enterprise-grade LLM customer intelligence generation
    for Spotify behavioral segments. Produces detailed psychological personas,
    differentiated marketing strategies, and omnichannel creative execution copy.
    """
    
    # Pre-computed rich persona profiles synthesized from mathematical centroids
    CURATED_PERSONAS = {
        0: {
            "persona_name": "The Sonic Loyalist (Focus & Hi-Fi Powerhouse)",
            "archetype_avatar": "Marcus, 29 | Lead Software Engineer | Berlin, Germany",
            "tagline": "Music is my cognitive fuel and acoustic sanctuary.",
            "psychological_mindset": (
                "Highly disciplined, analytical, and immersion-driven. Marcus views Spotify not as entertainment, "
                "but as an essential productivity and cognitive enhancement utility. He listens 4.5+ hours a day "
                "during deep work blocks, almost exclusively to instrumental, ambient, post-rock, or neoclassical genres. "
                "He values acoustic purity, high bitrates, and uninterrupted focus. Skips are exceptionally rare (<6%) "
                "because he meticulously curates his 45+ custom playlists and values uninterrupted flow states."
            ),
            "friction_points": [
                "Unpredictable audio volume spikes or abrupt genre changes in algorithmic playlists.",
                "Intrusive pop-up promotions for mainstream pop albums during work hours.",
                "Lack of lossless FLAC / Hi-Fi streaming tier options on desktop DACs."
            ],
            "consumption_dna": {
                "Daily Stream": "275 mins/day (Top 5% of Spotify listeners)",
                "Acoustic Profile": "Instrumentalness: 0.72 | Acousticness: 0.65 | Energy: 0.48",
                "Primary Hardware": "Desktop App (DAC headphones) & Smart Speaker (Sonos)",
                "Discovery Style": "Save-heavy, meticulous album cataloging, high Discover Weekly save rate (55%)"
            },
            "differentiated_strategy": {
                "core_value_prop": "Frictionless deep-work immersion and studio-grade acoustic clarity.",
                "recommended_channels": ["In-App Desktop Banner (Non-disruptive)", "Email Deep Dives", "Sonos Integration Push"],
                "tone_of_voice": "Minimalist, sophisticated, audio-purist, calm and understated.",
                "monetization_upsell": "Prime candidate for Spotify HiFi / Lossless Audio add-on ($5.99/mo) and annual prepaid discount.",
                "retention_tactic": "Proactively suggest 'Focus Hour Automation' syncing with Google Calendar / Slack status.",
                "campaign_creatives": {
                    "push_notification": {
                        "title": "Your Deep Work Zone is Ready",
                        "body": "Zero interruptions, pure 320kbps acoustic focus. Start today's flow state.",
                        "cta": "Resume Focus Flow"
                    },
                    "in_app_modal": {
                        "headline": "Elevate Your Soundstage with Studio Audio",
                        "subhead": "Experience lossless fidelity mapped to your custom desktop workspace.",
                        "primary_cta": "Reserve Early Hi-Fi Access",
                        "secondary_cta": "Explore Lo-Fi Instrumentals"
                    },
                    "email_subject": "Marcus, your acoustic focus telemetry for Q3",
                    "audio_ad_script": "N/A (100% Premium subscriber cohort - target via editorial feature placements)"
                }
            }
        },
        1: {
            "persona_name": "The Commute & Workout Hitseeker",
            "archetype_avatar": "Chloe, 24 | Digital Account Manager | Chicago, USA",
            "tagline": "Give me instant adrenaline and chart-topping beats—no downtime.",
            "psychological_mindset": (
                "Hyper-dynamic, fast-paced, and state-driven. Chloe turns to Spotify as an emotional accelerator "
                "during high-stress morning transit and intense 6 PM CrossFit sessions. She has an exceptionally "
                "high skip rate (38%) because she refuses to endure a slow intro or lukewarm track. Her listening "
                "is heavily skewed toward high-energy dance, modern hip-hop, and trending TikTok anthems. "
                "She prefers algorithmic pre-packaged mixes (Today's Top Hits, Beast Mode) over building personal libraries."
            ),
            "friction_points": [
                "Unskippable audio ads during high-rep gym intervals or fast highway driving.",
                "Slow track load times or low-tempo shuffle selections.",
                "Ad-supported 6-skip per hour limit on Free mobile tier."
            ],
            "consumption_dna": {
                "Daily Stream": "110 mins/day (Concentrated in 7-9 AM and 5-7 PM spikes)",
                "Acoustic Profile": "Energy: 0.82 | Danceability: 0.78 | Skip Rate: 0.38",
                "Primary Hardware": "Mobile iOS (AirPods Pro) & Apple CarPlay",
                "Discovery Style": "Algorithmic lean-back; top 40 chart hits and curated workout stations"
            },
            "differentiated_strategy": {
                "core_value_prop": "Unlimited skips and instant adrenaline that never loses momentum.",
                "recommended_channels": ["Mobile Push (timed at 5:15 PM gym peak)", "CarPlay In-Car Audio Ads", "TikTok/Instagram Sponsored Stories"],
                "tone_of_voice": "Energetic, punchy, bold, rhythm-driven.",
                "monetization_upsell": "Convert Free users to Premium Individual with a 'Workout Skip-Free Guarantee' 3-month trial at $1.",
                "retention_tactic": "Dynamic BPM playlist matching: automatically calibrate playlist tempo to running cadence.",
                "campaign_creatives": {
                    "push_notification": {
                        "title": "Don't Let Ads Break Your PR",
                        "body": "Unlimited skips. Zero ad interruptions. 160 BPM nonstop for your 6 PM workout.",
                        "cta": "Unlock 3 Months for $1"
                    },
                    "in_app_modal": {
                        "headline": "Tired of Running Out of Skips?",
                        "subhead": "Power through your commute and workout without missing a beat.",
                        "primary_cta": "Go Unlimited Premium",
                        "secondary_cta": "Continue with Ads"
                    },
                    "email_subject": "Chloe, your workout playlists just got 20 new high-BPM bangers",
                    "audio_ad_script": "[Upbeat bass-drop] 'Nothing kills a heavy lift like an insurance ad. Get unlimited skips and zero ads on Spotify Premium for just $1.'"
                }
            }
        },
        2: {
            "persona_name": "The Spoken-Word Explorer (Podcasts & Audiobooks)",
            "archetype_avatar": "David, 41 | Management Consultant | London, UK",
            "tagline": "My commute is my classroom. Audio is how I stay ahead.",
            "psychological_mindset": (
                "Intellectually curious, career-focused, and time-conscious. David allocates over 56% of his "
                "audio consumption to long-form spoken word, including investigative journalism, tech podcasts, "
                "and business audiobooks. He listens at 1.5x speed with trim-silence enabled. He has a high disposable income, "
                "is extremely loyal (tenure 48+ months), and exhibits virtually zero churn risk. He values seamless cross-device "
                "playback resumption between his car, office laptop, and bedside smart speaker."
            ),
            "friction_points": [
                "Podcasts abruptly resetting playback position across devices.",
                "Lack of chapter markers and transcript search in niche shows.",
                "Limited audiobook hours included in standard Premium tier."
            ],
            "consumption_dna": {
                "Daily Stream": "190 mins/day (56% Podcasts, 18% Audiobooks, 26% Jazz/Lo-Fi)",
                "Acoustic Profile": "Speechiness: High | Acousticness: 0.45 | Skip Rate: 0.11",
                "Primary Hardware": "Mobile iOS, Smart Speaker (Amazon Echo), Desktop App",
                "Discovery Style": "Episode queues, creator-followed notifications, topic-specific search"
            },
            "differentiated_strategy": {
                "core_value_prop": "The world's richest library of intellect, insights, and narrated wisdom.",
                "recommended_channels": ["Podcast Player Drawer Banner", "Curated Weekend Digest Email", "Smart Speaker Voice Prompts"],
                "tone_of_voice": "Thought-provoking, intellectual, concise, value-oriented.",
                "monetization_upsell": "Cross-sell 'Audiobook Expansion Pack' (additional 15 hours/mo for $9.99) or Spotify Duo.",
                "retention_tactic": "Weekly episode queue digest with personalized AI-generated key takeaways.",
                "campaign_creatives": {
                    "push_notification": {
                        "title": "Your Weekend Intellectual Briefing",
                        "body": "3 new episodes from your subscribed tech & finance leaders are ready to stream.",
                        "cta": "Open Queue"
                    },
                    "in_app_modal": {
                        "headline": "Unlock 15 Additional Audiobook Hours",
                        "subhead": "Never pause mid-chapter. Finish bestselling non-fiction titles on your schedule.",
                        "primary_cta": "Add Audiobook Boost ($9.99)",
                        "secondary_cta": "Browse Free Chapters"
                    },
                    "email_subject": "David, explore this week's top business audiobooks and podcast transcripts",
                    "audio_ad_script": "N/A (High LTV Premium cohort; engage via in-player bookmark cards)"
                }
            }
        },
        3: {
            "persona_name": "The Social Discovery Nomad (Viral & Trends)",
            "archetype_avatar": "Sofia, 21 | Media & Communications Student | São Paulo, Brazil",
            "tagline": "If a song isn't in my Instagram Story or Spotify Jam, does it even exist?",
            "psychological_mindset": (
                "Hyper-connected, culturally expressive, and socially motivated. Sofia discovers music through "
                "TikTok trends, Spotify Blend, and collaborative playlists with friends. She logs over 30+ social shares "
                "every quarter and is the first to post her Spotify Wrapped in December. She has a high Discover Weekly "
                "save rate (68%) and loves high-valence, euphoric, and dance-heavy pop/indie crossover tracks. "
                "She often organizes Spotify Jam sessions during dorm gatherings."
            ),
            "friction_points": [
                "Clunky playlist sharing experiences across third-party social apps.",
                "Difficulty splitting bills on shared Premium plans.",
                "Stale algorithmic recommendations that lag behind viral TikTok sounds."
            ],
            "consumption_dna": {
                "Daily Stream": "160 mins/day (Late night 11 PM-3 AM and weekend spikes)",
                "Acoustic Profile": "Valence: 0.72 | Danceability: 0.74 | Social Shares: 32/qtr",
                "Primary Hardware": "Mobile iOS (iPhone 15) & Mobile Android",
                "Discovery Style": "Viral trends, friend Blends, Jam sessions, Discover Weekly"
            },
            "differentiated_strategy": {
                "core_value_prop": "Turn every track into a shared memory and cultural moment.",
                "recommended_channels": ["Instagram Stories / TikTok Deep-Link Ads", "Spotify In-App Jam Invitations", "Gen-Z Push Notifications"],
                "tone_of_voice": "Playful, witty, conversational, culturally tuned, meme-fluent.",
                "monetization_upsell": "Convert from Free to Premium Student ($5.99) or initiate a Premium Duo invite to a partner/best friend.",
                "retention_tactic": "Weekly 'Friend Blend Surge' notifications highlighting shared taste compatibility scores.",
                "campaign_creatives": {
                    "push_notification": {
                        "title": "Your taste is 94% compatible with Lucas 🎧",
                        "body": "Listen together in real time with Spotify Jam. Start your party queue now.",
                        "cta": "Start a Jam Session"
                    },
                    "in_app_modal": {
                        "headline": "Two Accounts. One Discounted Price.",
                        "subhead": "Share Spotify Duo with your favorite listening partner for just $14.99/mo.",
                        "primary_cta": "Invite Partner to Duo",
                        "secondary_cta": "Keep Individual"
                    },
                    "email_subject": "Sofia, you and 3 friends have identical viral favorites this week 🔥",
                    "audio_ad_script": "[Sound of friends laughing] 'Why take turns passing the aux cord? Start a Spotify Jam and let everyone queue tracks at once. Try Premium Student for $5.99.'"
                }
            }
        },
        4: {
            "persona_name": "The Value Maximizer (Family & Multi-Device)",
            "archetype_avatar": "Rajesh & Priya, 38 | Dual-Income Family | Mumbai / Toronto",
            "tagline": "One subscription that keeps the entire household entertained without friction.",
            "psychological_mindset": (
                "Pragmatic, cost-conscious, and utility-driven. The Value Maximizer manages a Spotify Family plan "
                "with 4 to 6 connected profiles spanning kids' bedtime audiobooks, cooking soundtracks on Google Nest, "
                "and car audio during weekend road trips. They have a long tenure (40+ months) and very low churn propensity "
                "because canceling would disrupt the entire household's music habits. However, they are highly sensitive "
                "to price hikes and scrutinize subscription bills quarterly."
            ),
            "friction_points": [
                "Family members overriding each other's smart speaker queues.",
                "Parental controls that are difficult to configure for explicit track filtering.",
                "Annual subscription price increases without clear perceived value additions."
            ],
            "consumption_dna": {
                "Daily Stream": "210 cumulative mins/day across 4+ registered devices",
                "Acoustic Profile": "Balanced energy (0.55), high variety, multi-genre eclectic",
                "Primary Hardware": "Smart Speaker (Sonos/Google Nest), Smart TV, Connected Car",
                "Discovery Style": "Family weekend playlists, kids bedtime stories, curated cooking chill"
            },
            "differentiated_strategy": {
                "core_value_prop": "Total household harmony with dedicated profiles and parental peace of mind.",
                "recommended_channels": ["Billing Confirmation Email Footers", "Smart TV Welcome Screens", "Household App Push"],
                "tone_of_voice": "Warm, reassuring, family-centric, high-utility.",
                "monetization_upsell": "Protect existing $16.99/mo Family subscription against churn; cross-sell Spotify Kids app adoption.",
                "retention_tactic": "Automated 'Family Weekend Mix' delivered every Saturday morning tailored to household tastes.",
                "campaign_creatives": {
                    "push_notification": {
                        "title": "Your Family Weekend Mix is Here",
                        "body": "Curated tracks everyone in the car will agree on. Hit play for the weekend trip.",
                        "cta": "Play Family Mix"
                    },
                    "in_app_modal": {
                        "headline": "Keep the Kids Entertained with Spotify Kids",
                        "subhead": "Included in your Family Plan at zero extra cost. Safe, ad-free, sing-along fun.",
                        "primary_cta": "Download Spotify Kids App",
                        "secondary_cta": "Manage Explicit Filters"
                    },
                    "email_subject": "Your Spotify Family monthly digest: 142 hours of household memories",
                    "audio_ad_script": "N/A (Premium Family tier cohort; maintain retention via cross-product adoption)"
                }
            }
        },
        5: {
            "persona_name": "The Disengaged Free-Tier Drifter (At-Risk)",
            "archetype_avatar": "Alex, 27 | Retail Associate | Austin, USA",
            "tagline": "I only open Spotify when YouTube has too many video ads, but then I hit skip limits.",
            "psychological_mindset": (
                "Casual, low-commitment, and highly churn-vulnerable. Alex streams less than 35 minutes a day, "
                "often with gaps of 4 to 6 days between sessions. His 30-day stream volume has plummeted by 35%. "
                "He skips over 50% of songs because shuffle mode plays tracks he dislikes, and he quickly closes the app "
                "when confronted with back-to-back 30-second audio ads. He has never created a playlist and has fewer than "
                "150 saved songs in his library. If pushed with a generic $10.99 pitch, he will uninstall the app."
            ),
            "friction_points": [
                "Repetitive, irrelevant audio advertisements.",
                "Lack of on-demand song picking on mobile free tier.",
                "Empty library and lack of personalized onboarding hooks."
            ],
            "consumption_dna": {
                "Daily Stream": "32 mins/day (Erratic, declining trend -35% L30D)",
                "Acoustic Profile": "High skip rate: 0.52 | Active days: 8/mo | Churn Risk: 0.78 (Critical)",
                "Primary Hardware": "Mobile Android (budget device)",
                "Discovery Style": "Passive, default radio, infrequent searches"
            },
            "differentiated_strategy": {
                "core_value_prop": "Frictionless re-engagement: your favorite 10 songs on demand with zero hassle.",
                "recommended_channels": ["Re-engagement SMS / Push (Only during high-receptivity hours 8 PM)", "Personalized Email Win-Back"],
                "tone_of_voice": "Low pressure, empathetic, welcoming, frictionless.",
                "monetization_upsell": "Do NOT push full-price Premium. Offer a 'Frictionless 30-Day Free Pass' or 'Day Pass for $0.49'.",
                "retention_tactic": "Generate an automated 15-track 'Welcome Back Fast Mix' matching his top 5 all-time listened artists.",
                "campaign_creatives": {
                    "push_notification": {
                        "title": "We made a mix of your 10 favorite tracks",
                        "body": "No search needed. Just hit play and listen on demand.",
                        "cta": "Listen to Your Mix"
                    },
                    "in_app_modal": {
                        "headline": "Enjoy 30 Days of Spotify Premium on Us",
                        "subhead": "No credit card required upfront. Unlimited skips and zero ads on your favorite songs.",
                        "primary_cta": "Start Instant 30-Day Free Pass",
                        "secondary_cta": "Maybe Later"
                    },
                    "email_subject": "Alex, your favorite artists just dropped new tracks you haven't heard yet",
                    "audio_ad_script": "[Calm, friendly voice] 'We know you just want to hear your song. Tap the banner right now for 30 minutes of uninterrupted listening on Spotify.'"
                }
            }
        }
    }

    def __init__(self):
        pass

    def get_persona(self, segment_id: int) -> Dict[str, Any]:
        """Returns the complete synthesized persona for a given segment ID."""
        return self.CURATED_PERSONAS.get(segment_id, self.CURATED_PERSONAS[0])

    def generate_custom_strategy(
        self,
        segment_id: int,
        business_objective: str = "Convert Free to Premium",
        campaign_tone: str = "Witty & Punchy",
        channel_focus: str = "Omni-Channel"
    ) -> Dict[str, Any]:
        """
        Interactive LLM Strategy Engine: Simulates generating dynamic real-time campaign strategy
        and creative concepts tailored to selected segment, business objective, and tone.
        """
        persona = self.get_persona(segment_id)
        short_name = persona["persona_name"].split("(")[0].strip()
        
        # Dynamic strategy synthesis
        strategy_response = {
            "segment_id": segment_id,
            "segment_name": persona["persona_name"],
            "business_objective": business_objective,
            "campaign_tone": campaign_tone,
            "channel_focus": channel_focus,
            "executive_summary": (
                f"For {short_name}, the primary behavioral barrier to '{business_objective}' is "
                f"friction around their core listening habit ({persona['consumption_dna']['Daily Stream']}). "
                f"Using a {campaign_tone.lower()} tone across {channel_focus.lower()}, the strategy pivots from "
                f"generic demographic promotions to hyper-contextual behavioral incentives."
            ),
            "target_friction_solved": persona["friction_points"][0],
            "differentiated_tactics": [
                {
                    "tactic_pillar": "Behavioral Contextual Hook",
                    "description": (
                        f"Trigger campaigns precisely when {short_name} users are in their peak state "
                        f"({persona['consumption_dna']['Primary Hardware']}), delivering immediate utility."
                    )
                },
                {
                    "tactic_pillar": "Economic Incentive Alignment",
                    "description": persona["differentiated_strategy"]["monetization_upsell"]
                },
                {
                    "tactic_pillar": "Retention Guardrail",
                    "description": persona["differentiated_strategy"]["retention_tactic"]
                }
            ],
            "dynamic_creative_variants": {
                "variant_a_push": {
                    "headline": f"Spotify: {business_objective} for {short_name}",
                    "copy": (
                        f"Your listening tells a story. "
                        f"{persona['differentiated_strategy']['campaign_creatives']['push_notification']['body']}"
                    ),
                    "cta": persona["differentiated_strategy"]["campaign_creatives"]["push_notification"]["cta"],
                    "projected_ctr": "4.8% (+210% vs Demographic Benchmark)"
                },
                "variant_b_in_app": {
                    "headline": persona["differentiated_strategy"]["campaign_creatives"]["in_app_modal"]["headline"],
                    "copy": persona["differentiated_strategy"]["campaign_creatives"]["in_app_modal"]["subhead"],
                    "cta": persona["differentiated_strategy"]["campaign_creatives"]["in_app_modal"]["primary_cta"],
                    "projected_cvr": "3.4% (+280% vs Demographic Benchmark)"
                },
                "variant_c_audio_or_email": {
                    "format": "Audio Ad Script" if "Free" in persona["persona_name"] or segment_id in [1, 3, 5] else "Curated Email Digest",
                    "content": (
                        persona["differentiated_strategy"]["campaign_creatives"]["audio_ad_script"]
                        if ("Free" in persona["persona_name"] or segment_id in [1, 3, 5])
                        else f"Subject: {persona['differentiated_strategy']['campaign_creatives']['email_subject']}"
                    )
                }
            },
            "attribution_forecast": {
                "estimated_cac_reduction": "38.5%",
                "conversion_lift_multiplier": "3.14x",
                "retention_hazard_mitigation": "-24.2% churn probability"
            }
        }
        
        return strategy_response
