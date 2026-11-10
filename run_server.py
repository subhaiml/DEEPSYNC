"""
Runner script for DEEPSYNC Forensic Web Server
"""
import uvicorn

if __name__ == "__main__":
    print("=" * 65)
    print("DEEPSYNC: Multimodal Deepfake Forensic System (Group-101)")
    print("FastAPI Backend & Chart.js Live Forensic Dashboard")
    print("Access URL: http://127.0.0.1:8000")
    print("=" * 65)
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)

