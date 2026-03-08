import React, { useState, useCallback } from "react";
import axios from "axios";
import {
    PieChart,
    Pie,
    Cell,
    Tooltip,
    ResponsiveContainer,
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Legend
} from "recharts";

const API_BASE = process.env.REACT_APP_API_BASE || "http://football-api:8000";

function Dashboard() {
    const [file, setFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [statusText, setStatusText] = useState("");
    const [result, setResult] = useState(null);

    // Poll the job status until it completes or fails
    const pollJob = useCallback(async (jobId) => {
        while (true) {
            await new Promise((resolve) => setTimeout(resolve, 2000)); // wait 2s between polls

            const { data: job } = await axios.get(`${API_BASE}/jobs/${jobId}`);

            setProgress(job.progress || 0);
            setStatusText(
                `Status: ${job.status} — ${Math.round(job.progress || 0)}% ` +
                `(${job.frames_processed || 0}/${job.total_frames || "?"} frames)`
            );

            if (job.status === "completed") {
                return jobId;
            }
            if (job.status === "failed") {
                throw new Error(job.error || "Analysis failed on server");
            }
        }
    }, []);

    const handleUpload = async () => {
        if (!file) return;

        const formData = new FormData();
        formData.append("file", file);

        try {
            setLoading(true);
            setResult(null);
            setProgress(0);
            setStatusText("Uploading video...");

            // 1) Submit video — get back a job_id
            const { data: jobInfo } = await axios.post(
                `${API_BASE}/analyze-video`,
                formData
            );
            const jobId = jobInfo.job_id;
            setStatusText("Processing video... ⏳");

            // 2) Poll until the job completes
            await pollJob(jobId);

            // 3) Fetch the actual results
            const [possessionRes, trackingRes] = await Promise.all([
                axios.get(`${API_BASE}/jobs/${jobId}/possession`),
                axios.get(`${API_BASE}/jobs/${jobId}/tracking`)
            ]);

            const possession = possessionRes.data;
            const tracking = trackingRes.data;

            // Count players per team from tracking data
            let teamAPlayers = 0;
            let teamBPlayers = 0;
            if (tracking.player_tracking) {
                Object.values(tracking.player_tracking).forEach((player) => {
                    if (player.team === "Team A") teamAPlayers++;
                    else if (player.team === "Team B") teamBPlayers++;
                });
            }

            setResult({
                team_a_possession: possession.team_a,
                team_b_possession: possession.team_b,
                team_a_players: teamAPlayers,
                team_b_players: teamBPlayers,
                players_tracked: tracking.players_tracked,
                ball_detections: tracking.ball_tracking
                    ? tracking.ball_tracking.length
                    : 0,
                frames_processed: tracking.frames_processed,
                fps: tracking.fps
            });

            setStatusText("Analysis complete ✅");
        } catch (error) {
            console.error(error);
            alert("Error analyzing video: " + (error.message || "Unknown error"));
            setStatusText("");
        } finally {
            setLoading(false);
        }
    };

    const possessionData = [
        { name: "Team A", value: result?.team_a_possession || 0 },
        { name: "Team B", value: result?.team_b_possession || 0 }
    ];

    const playerData = [
        { name: "Team A Players", value: result?.team_a_players || 0 },
        { name: "Team B Players", value: result?.team_b_players || 0 }
    ];

    const COLORS = ["#3b82f6", "#ef4444"];

    return (
        <div style={styles.container}>
            <h1 style={styles.title}>⚽ AI Football Analysis</h1>

            <div style={styles.card}>
                <input type="file" accept="video/*" onChange={(e) => setFile(e.target.files[0])} />
                <button
                    style={{
                        ...styles.button,
                        opacity: loading ? 0.6 : 1,
                        cursor: loading ? "not-allowed" : "pointer"
                    }}
                    onClick={handleUpload}
                    disabled={loading}
                >
                    {loading ? "Analyzing..." : "Analyze Match"}
                </button>
            </div>

            {loading && (
                <div style={styles.loader}>
                    <div>{statusText || "Analyzing... ⏳"}</div>
                    <div style={styles.progressBarContainer}>
                        <div style={{ ...styles.progressBar, width: `${progress}%` }} />
                    </div>
                </div>
            )}

            {!loading && statusText && !result && (
                <div style={styles.loader}>{statusText}</div>
            )}

            {result && (
                <div style={styles.results}>
                    <div style={styles.statsGrid}>
                        <StatCard title="Players Tracked" value={result.players_tracked} />
                        <StatCard title="Ball Detections" value={result.ball_detections} />
                        <StatCard title="Frames Processed" value={result.frames_processed} />
                        <StatCard title="Video FPS" value={result.fps} />
                    </div>

                    <div style={styles.chartGrid}>
                        {/* Possession Pie Chart */}
                        <div style={styles.chartCard}>
                            <h3>Possession %</h3>
                            <ResponsiveContainer width="100%" height={300}>
                                <PieChart>
                                    <Pie
                                        data={possessionData}
                                        dataKey="value"
                                        outerRadius={100}
                                        label={({ name, value }) => `${name}: ${value}%`}
                                    >
                                        {possessionData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={COLORS[index]} />
                                        ))}
                                    </Pie>
                                    <Tooltip formatter={(value) => `${value}%`} />
                                </PieChart>
                            </ResponsiveContainer>
                        </div>

                        {/* Player Comparison Bar Chart */}
                        <div style={styles.chartCard}>
                            <h3>Players Per Team</h3>
                            <ResponsiveContainer width="100%" height={300}>
                                <BarChart data={playerData}>
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="name" />
                                    <YAxis allowDecimals={false} />
                                    <Tooltip />
                                    <Legend />
                                    <Bar dataKey="value" fill="#22c55e" />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

function StatCard({ title, value }) {
    return (
        <div style={styles.statCard}>
            <h3>{title}</h3>
            <p style={{ fontSize: 24, fontWeight: "bold" }}>{value}</p>
        </div>
    );
}

const styles = {
    container: {
        backgroundColor: "#0f172a",
        minHeight: "100vh",
        color: "white",
        padding: 40,
        fontFamily: "Arial"
    },
    title: {
        marginBottom: 30
    },
    card: {
        backgroundColor: "#1e293b",
        padding: 20,
        borderRadius: 10,
        marginBottom: 20
    },
    button: {
        marginLeft: 10,
        padding: "8px 15px",
        backgroundColor: "#3b82f6",
        color: "white",
        border: "none",
        borderRadius: 5,
        cursor: "pointer"
    },
    loader: {
        marginTop: 20,
        fontSize: 18
    },
    progressBarContainer: {
        marginTop: 10,
        backgroundColor: "#334155",
        borderRadius: 5,
        height: 10,
        width: "100%",
        overflow: "hidden"
    },
    progressBar: {
        height: "100%",
        backgroundColor: "#3b82f6",
        borderRadius: 5,
        transition: "width 0.5s ease"
    },
    results: {
        marginTop: 30
    },
    statsGrid: {
        display: "grid",
        gridTemplateColumns: "repeat(4, 1fr)",
        gap: 20
    },
    statCard: {
        backgroundColor: "#1e293b",
        padding: 20,
        borderRadius: 10,
        textAlign: "center"
    },
    chartGrid: {
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: 20,
        marginTop: 30
    },
    chartCard: {
        backgroundColor: "#1e293b",
        padding: 20,
        borderRadius: 10
    },
};

export default Dashboard;