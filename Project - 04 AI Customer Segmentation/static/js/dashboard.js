/**
 * Spotify AI Customer Segmentation & Intelligence Dashboard Controller
 * Project - 04: AI Customer Segmentation
 * Handles Chart.js visualizations, PCA Latent Space rendering,
 * interactive LLM strategy generation, user lookup, and campaign ROI simulation.
 */

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initOverviewCharts();
    initPCALatentSpace();
    initRadarChart();
    initLLMStrategyStudio();
    initUserLookup();
    initCampaignSimulator();
});

// 1. Tab Navigation System
function initTabs() {
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetId = btn.getAttribute("data-tab");
            
            tabButtons.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));

            btn.classList.add("active");
            const targetContent = document.getElementById(targetId);
            if (targetContent) {
                targetContent.classList.add("active");
            }
        });
    });
}

// 2. Executive Overview Charts (Cluster Share & Demographic Inefficiency)
let clusterShareChart = null;
let demoInefficiencyChart = null;

async function initOverviewCharts() {
    try {
        const res = await fetch("/api/summary");
        const data = await res.json();
        const profiles = data.segment_profiles;
        
        // Prepare Cluster Share Doughnut Chart
        const labels = Object.values(profiles).map(p => p.short_name);
        const shares = Object.values(profiles).map(p => p.share_pct);
        const colors = Object.values(profiles).map(p => p.color);

        const ctxShare = document.getElementById("clusterShareChart");
        if (ctxShare) {
            clusterShareChart = new Chart(ctxShare, {
                type: "doughnut",
                data: {
                    labels: labels,
                    datasets: [{
                        data: shares,
                        backgroundColor: colors,
                        borderColor: "#0f141d",
                        borderWidth: 2,
                        hoverOffset: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: "right",
                            labels: {
                                color: "#94a3b8",
                                font: { family: "'Plus Jakarta Sans', sans-serif", size: 11 }
                            }
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return ` ${context.label}: ${context.raw}% of base`;
                                }
                            }
                        }
                    },
                    cutout: "68%"
                }
            });
        }

        // Intra-group Variance / Inefficiency Chart
        const ctxIneff = document.getElementById("demoInefficiencyChart");
        if (ctxIneff) {
            demoInefficiencyChart = new Chart(ctxIneff, {
                type: "bar",
                data: {
                    labels: ["18-24 Gen-Z", "25-34 Millennial", "35-50 Gen-X", "51+ Boomer", "AI Behavioral Clusters (Avg)"],
                    datasets: [
                        {
                            label: "Intra-Group Behavioral Variance (Higher = More Inefficient)",
                            data: [0.88, 0.82, 0.79, 0.85, 0.22],
                            backgroundColor: [
                                "rgba(255, 42, 109, 0.65)",
                                "rgba(255, 42, 109, 0.65)",
                                "rgba(255, 42, 109, 0.65)",
                                "rgba(255, 42, 109, 0.65)",
                                "rgba(29, 185, 84, 0.85)"
                            ],
                            borderColor: [
                                "#ff2a6d",
                                "#ff2a6d",
                                "#ff2a6d",
                                "#ff2a6d",
                                "#1db954"
                            ],
                            borderWidth: 1.5,
                            borderRadius: 6
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1.0,
                            grid: { color: "rgba(255, 255, 255, 0.05)" },
                            ticks: { color: "#64748b" }
                        },
                        x: {
                            grid: { display: false },
                            ticks: { color: "#94a3b8", font: { size: 11 } }
                        }
                    }
                }
            });
        }
    } catch (e) {
        console.error("Error loading overview charts:", e);
    }
}

// 3. Behavioral Latent Space Explorer (2D PCA Projection)
let pcaScatterChart = null;

async function initPCALatentSpace() {
    const ctx = document.getElementById("pcaScatterChart");
    if (!ctx) return;

    try {
        const res = await fetch("/api/pca-scatter?limit=1200");
        const json = await res.json();
        const points = json.points;

        // Group points by cluster
        const clusters = {};
        points.forEach(p => {
            if (!clusters[p.cluster_id]) {
                clusters[p.cluster_id] = {
                    label: p.segment_name,
                    color: p.color,
                    data: []
                };
            }
            clusters[p.cluster_id].data.push({
                x: p.x,
                y: p.y,
                telemetry: p
            });
        });

        const datasets = Object.values(clusters).map(c => ({
            label: c.label,
            data: c.data,
            backgroundColor: c.color + "99",
            borderColor: c.color,
            borderWidth: 1,
            pointRadius: 4,
            pointHoverRadius: 7
        }));

        pcaScatterChart = new Chart(ctx, {
            type: "scatter",
            data: { datasets: datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: "top",
                        labels: {
                            color: "#94a3b8",
                            font: { family: "'Plus Jakarta Sans', sans-serif", size: 11 },
                            boxWidth: 10
                        }
                    },
                    tooltip: {
                        backgroundColor: "rgba(15, 20, 29, 0.95)",
                        titleColor: "#fff",
                        bodyColor: "#cbd5e1",
                        borderColor: "rgba(255, 255, 255, 0.15)",
                        borderWidth: 1,
                        padding: 12,
                        callbacks: {
                            title: function(items) {
                                const pt = items[0].raw.telemetry;
                                return `${pt.user_id} [${pt.segment_name}]`;
                            },
                            label: function(item) {
                                const t = item.raw.telemetry;
                                return [
                                    `Streaming: ${t.streaming_mins} min/day`,
                                    `Skip Rate: ${(t.skip_rate * 100).toFixed(1)}%`,
                                    `Podcast Share: ${(t.podcast_pct * 100).toFixed(1)}%`,
                                    `Device: ${t.device}`,
                                    `Tier: ${t.tier} | LTV: $${t.ltv.toFixed(2)}`
                                ];
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: "Principal Component 1 (Audio Focus & Immersion vs Casual)", color: "#64748b" },
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#64748b" }
                    },
                    y: {
                        title: { display: true, text: "Principal Component 2 (Spoken-Word & High Energy vs Ambient)", color: "#64748b" },
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#64748b" }
                    }
                }
            }
        });
    } catch (e) {
        console.error("Error loading PCA scatter:", e);
    }
}

// 4. Segment Radar & Profiler
let radarChart = null;

async function initRadarChart() {
    const ctx = document.getElementById("segmentRadarChart");
    if (!ctx) return;

    try {
        const res = await fetch("/api/radar-data");
        const data = await res.json();

        radarChart = new Chart(ctx, {
            type: "radar",
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                elements: {
                    line: { borderWidth: 2 }
                },
                plugins: {
                    legend: {
                        position: "right",
                        labels: {
                            color: "#94a3b8",
                            font: { family: "'Plus Jakarta Sans', sans-serif", size: 11 }
                        }
                    }
                },
                scales: {
                    r: {
                        angleLines: { color: "rgba(255, 255, 255, 0.08)" },
                        grid: { color: "rgba(255, 255, 255, 0.08)" },
                        pointLabels: {
                            color: "#cbd5e1",
                            font: { size: 11, weight: 600 }
                        },
                        ticks: {
                            display: false,
                            min: 0,
                            max: 100
                        }
                    }
                }
            }
        });

        // Initialize Segment Detail Selector
        const select = document.getElementById("radarSegmentSelector");
        if (select) {
            select.addEventListener("change", (e) => {
                updateSegmentDetailView(parseInt(e.target.value));
            });
            updateSegmentDetailView(0);
        }
    } catch (e) {
        console.error("Error loading radar chart:", e);
    }
}

async function updateSegmentDetailView(segId) {
    try {
        const res = await fetch("/api/summary");
        const json = await res.json();
        const seg = json.segment_profiles[segId];
        if (!seg) return;

        document.getElementById("detailSegName").innerText = seg.name;
        document.getElementById("detailSegName").style.color = seg.color;
        document.getElementById("detailSegShare").innerText = `${seg.share_pct}% of listener base (${seg.user_count.toLocaleString()} users)`;
        document.getElementById("detailSegLtv").innerText = `$${seg.avg_ltv.toFixed(2)}`;
        document.getElementById("detailSegStream").innerText = `${seg.avg_daily_streaming_min} min/day`;
        document.getElementById("detailSegSkip").innerText = `${(seg.avg_skip_rate * 100).toFixed(1)}%`;
        document.getElementById("detailSegPodcast").innerText = `${seg.avg_podcast_share_pct}%`;
        document.getElementById("detailSegChurn").innerText = `${(seg.avg_churn_risk * 100).toFixed(1)}%`;
        
        // Devices and dayparts badges
        const devContainer = document.getElementById("detailDevices");
        devContainer.innerHTML = Object.entries(seg.top_devices)
            .map(([d, pct]) => `<span class="badge-metric">${d}: <strong>${pct}%</strong></span>`)
            .join(" ");

        const dayContainer = document.getElementById("detailDayparts");
        dayContainer.innerHTML = Object.entries(seg.top_dayparts)
            .map(([dp, pct]) => `<span class="badge-metric">${dp}: <strong>${pct}%</strong></span>`)
            .join(" ");

        // Top distinctive Z-Scores
        const zContainer = document.getElementById("detailZScores");
        const sortedZ = Object.entries(seg.feature_stats)
            .map(([k, v]) => ({ feature: k, z: v.z_score, mean: v.mean }))
            .sort((a, b) => Math.abs(b.z) - Math.abs(a.z))
            .slice(0, 4);

        zContainer.innerHTML = sortedZ.map(item => {
            const isPos = item.z > 0;
            const sign = isPos ? "+" : "";
            const color = isPos ? "#1db954" : "#ff7a00";
            return `<div style="font-size: 0.8rem; margin-bottom: 4px;">
                <span style="color: #94a3b8;">${item.feature.replace(/_/g, ' ')}:</span>
                <strong style="color: ${color};"> ${sign}${item.z}σ</strong> (Mean: ${item.mean})
            </div>`;
        }).join("");

    } catch (e) {
        console.error("Error updating segment detail:", e);
    }
}

// 5. LLM Persona & Strategy Studio
function initLLMStrategyStudio() {
    const btnGenerate = document.getElementById("btnGenerateStrategy");
    if (!btnGenerate) return;

    btnGenerate.addEventListener("click", async () => {
        const segId = parseInt(document.getElementById("llmSegmentSelect").value);
        const objective = document.getElementById("llmObjectiveSelect").value;
        const tone = document.getElementById("llmToneSelect").value;
        const channel = document.getElementById("llmChannelSelect").value;

        btnGenerate.disabled = true;
        btnGenerate.innerHTML = `<span class="tab-icon">⏳</span> Synthesizing Strategy with LLM...`;

        try {
            const res = await fetch("/api/generate-strategy", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    segment_id: segId,
                    objective: objective,
                    tone: tone,
                    channel: channel
                })
            });

            const data = await res.json();
            renderGeneratedStrategy(data);
        } catch (e) {
            console.error("Error generating strategy:", e);
        } finally {
            btnGenerate.disabled = false;
            btnGenerate.innerHTML = `✨ Synthesize Differentiated Strategy`;
        }
    });

    // Auto trigger on first load
    btnGenerate.click();
}

function renderGeneratedStrategy(data) {
    document.getElementById("resStrategySummary").innerText = data.executive_summary;
    document.getElementById("resFrictionSolved").innerText = `Target Friction Solved: "${data.target_friction_solved}"`;
    
    // Tactics
    const tacticsBox = document.getElementById("resTacticsList");
    tacticsBox.innerHTML = data.differentiated_tactics.map(t => `
        <div style="margin-bottom: 10px; border-left: 2px solid #1db954; padding-left: 10px;">
            <strong style="color: #1db954; font-size: 0.82rem;">${t.tactic_pillar}:</strong>
            <p style="font-size: 0.8rem; color: #cbd5e1; margin-top: 2px;">${t.description}</p>
        </div>
    `).join("");

    // Creative Previews
    const cr = data.dynamic_creative_variants;
    // Push
    document.getElementById("pushTitle").innerText = cr.variant_a_push.headline;
    document.getElementById("pushBody").innerText = cr.variant_a_push.copy;
    document.getElementById("pushCta").innerText = `CTA: ${cr.variant_a_push.cta} | Projected CTR: ${cr.variant_a_push.projected_ctr}`;

    // In-App Modal
    document.getElementById("modalHeadline").innerText = cr.variant_b_in_app.headline;
    document.getElementById("modalSubhead").innerText = cr.variant_b_in_app.copy;
    document.getElementById("modalCta").innerText = cr.variant_b_in_app.cta;

    // Audio/Email
    document.getElementById("audioEmailLabel").innerText = cr.variant_c_audio_or_email.format;
    document.getElementById("audioEmailContent").innerText = cr.variant_c_audio_or_email.content;

    // Forecast metrics
    document.getElementById("forecastCac").innerText = `-${data.attribution_forecast.estimated_cac_reduction}`;
    document.getElementById("forecastLift").innerText = data.attribution_forecast.conversion_lift_multiplier;
    document.getElementById("forecastChurn").innerText = data.attribution_forecast.retention_hazard_mitigation;
}

// 6. User Lookup & Affinity Inspector
function initUserLookup() {
    const btnSearch = document.getElementById("btnSearchUser");
    const inputSearch = document.getElementById("inputUserId");
    if (!btnSearch || !inputSearch) return;

    btnSearch.addEventListener("click", () => {
        executeUserSearch(inputSearch.value.trim());
    });

    inputSearch.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            executeUserSearch(inputSearch.value.trim());
        }
    });

    // Sample user chips
    document.querySelectorAll(".user-chip").forEach(chip => {
        chip.addEventListener("click", () => {
            const uid = chip.getAttribute("data-user-id");
            inputSearch.value = uid;
            executeUserSearch(uid);
        });
    });

    // Run first user search
    executeUserSearch("SPOT-100010");
}

async function executeUserSearch(userId) {
    if (!userId) return;
    try {
        const res = await fetch(`/api/user/${encodeURIComponent(userId)}`);
        if (!res.ok) {
            alert(`User ID "${userId}" was not found. Please try e.g. SPOT-100001 or SPOT-100045.`);
            return;
        }
        const u = await res.json();
        
        document.getElementById("userFoundId").innerText = u.user_id;
        document.getElementById("userFoundSegment").innerText = u.segment_name;
        document.getElementById("userFoundSegment").style.color = u.segment_color;
        document.getElementById("userFoundDemographic").innerText = `${u.age} y/o | ${u.gender} | ${u.country} (${u.demographic_cohort})`;

        // Telemetry grid
        document.getElementById("userFoundTier").innerText = u.subscription_tier;
        document.getElementById("userFoundTenure").innerText = `${u.tenure_months} months`;
        document.getElementById("userFoundLtv").innerText = `$${u.lifetime_value_ltv.toFixed(2)}`;
        document.getElementById("userFoundStream").innerText = `${u.daily_streaming_minutes} min/day`;
        document.getElementById("userFoundSkip").innerText = `${(u.skip_rate * 100).toFixed(1)}%`;
        document.getElementById("userFoundPodcast").innerText = `${(u.podcast_stream_pct * 100).toFixed(1)}%`;
        document.getElementById("userFoundDevice").innerText = u.primary_device;
        document.getElementById("userFoundChurn").innerText = `${(u.churn_risk_score * 100).toFixed(1)}%`;
        
        // Churn color coding
        const churnEl = document.getElementById("userFoundChurn");
        if (u.churn_risk_score > 0.6) churnEl.style.color = "#ff2a6d";
        else if (u.churn_risk_score > 0.3) churnEl.style.color = "#ff7a00";
        else churnEl.style.color = "#1db954";

        // GMM Affinity Breakdown
        const affinityContainer = document.getElementById("userAffinityBars");
        affinityContainer.innerHTML = u.affinity_breakdown.map(aff => `
            <div class="affinity-bar-row">
                <div class="affinity-bar-label">${aff.name}</div>
                <div class="affinity-bar-track">
                    <div class="affinity-bar-fill" style="width: ${aff.probability}%; background-color: ${aff.color};"></div>
                </div>
                <div class="affinity-bar-pct">${aff.probability}%</div>
            </div>
        `).join("");

    } catch (e) {
        console.error("Error looking up user:", e);
    }
}

// 7. Demographic vs. Behavioral Campaign Simulator
let simulatorChart = null;

function initCampaignSimulator() {
    const slider = document.getElementById("budgetSlider");
    const display = document.getElementById("budgetDisplay");
    const objSelect = document.getElementById("simulatorObjective");

    if (!slider || !display) return;

    slider.addEventListener("input", (e) => {
        const val = parseInt(e.target.value);
        display.innerText = `$${val.toLocaleString()}`;
        debounceSimulation(val, objSelect.value);
    });

    objSelect.addEventListener("change", () => {
        debounceSimulation(parseInt(slider.value), objSelect.value);
    });

    // Run initial simulation
    runSimulationAPI(100000, "Premium Acquisition");
}

let debounceTimer = null;
function debounceSimulation(budget, objective) {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
        runSimulationAPI(budget, objective);
    }, 200);
}

async function runSimulationAPI(budget, objective) {
    try {
        const res = await fetch("/api/simulate-campaign", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ budget: budget, objective: objective })
        });
        const sim = await res.json();
        
        // Update metric callouts
        document.getElementById("simDemoConversions").innerText = sim.demographic_approach.total_conversions.toLocaleString();
        document.getElementById("simDemoCac").innerText = `$${sim.demographic_approach.blended_cac_usd.toFixed(2)}`;
        document.getElementById("simDemoRev").innerText = `$${sim.demographic_approach.net_12m_revenue_usd.toLocaleString()}`;
        document.getElementById("simDemoRoas").innerText = `${sim.demographic_approach.roas}x`;

        document.getElementById("simBehavConversions").innerText = sim.behavioral_approach.total_conversions.toLocaleString();
        document.getElementById("simBehavCac").innerText = `$${sim.behavioral_approach.blended_cac_usd.toFixed(2)}`;
        document.getElementById("simBehavRev").innerText = `$${sim.behavioral_approach.net_12m_revenue_usd.toLocaleString()}`;
        document.getElementById("simBehavRoas").innerText = `${sim.behavioral_approach.roas}x`;

        // Lift badges
        document.getElementById("simLiftConversions").innerText = `+${sim.efficiency_lift.conversions_lift_pct}%`;
        document.getElementById("simLiftCac").innerText = `-${sim.efficiency_lift.cac_reduction_pct}%`;
        document.getElementById("simLiftRev").innerText = `+$${sim.efficiency_lift.net_revenue_lift_usd.toLocaleString()}`;
        document.getElementById("simLiftRoas").innerText = `${sim.efficiency_lift.roas_multiplier}x Multiplier`;

        // Update comparison chart
        updateSimulatorChart(sim);

    } catch (e) {
        console.error("Error running campaign simulation:", e);
    }
}

function updateSimulatorChart(sim) {
    const ctx = document.getElementById("simulatorComparisonChart");
    if (!ctx) return;

    const data = {
        labels: ["Conversions", "Customer Acquisition Cost ($)", "12-Mo Revenue ($ in 100k)", "ROAS Return (x)"],
        datasets: [
            {
                label: "Traditional Demographic Targeting",
                data: [
                    sim.demographic_approach.total_conversions,
                    sim.demographic_approach.blended_cac_usd,
                    sim.demographic_approach.net_12m_revenue_usd / 100000,
                    sim.demographic_approach.roas * 100
                ],
                backgroundColor: "rgba(255, 42, 109, 0.75)",
                borderColor: "#ff2a6d",
                borderWidth: 1.5,
                borderRadius: 6
            },
            {
                label: "LLM Behavioral Intelligence System",
                data: [
                    sim.behavioral_approach.total_conversions,
                    sim.behavioral_approach.blended_cac_usd,
                    sim.behavioral_approach.net_12m_revenue_usd / 100000,
                    sim.behavioral_approach.roas * 100
                ],
                backgroundColor: "rgba(29, 185, 84, 0.85)",
                borderColor: "#1db954",
                borderWidth: 1.5,
                borderRadius: 6
            }
        ]
    };

    if (simulatorChart) {
        simulatorChart.data = data;
        simulatorChart.update();
    } else {
        simulatorChart = new Chart(ctx, {
            type: "bar",
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: "top",
                        labels: { color: "#94a3b8", font: { family: "'Plus Jakarta Sans', sans-serif" } }
                    }
                },
                scales: {
                    y: {
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#64748b" }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: "#cbd5e1" }
                    }
                }
            }
        });
    }
}
