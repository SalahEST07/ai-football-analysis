import React, { useState } from "react";

function VideoUpload() {
    const [file, setFile] = useState(null);
    const [result, setResult] = useState(null);

    const handleUpload = async () => {
        if (!file) return;

        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch("http://127.0.0.1:8000/analyze-video", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();
        setResult(data);
    };

    return (
        <div>
            <input
                type="file"
                onChange={(e) => setFile(e.target.files[0])}
            />
            <button onClick={handleUpload}>
                Analyze
            </button>

            {result && (
                <div>
                    <h3>Results:</h3>
                    <pre>{JSON.stringify(result, null, 2)}</pre>
                </div>
            )}
        </div>
    );
}

export default VideoUpload;