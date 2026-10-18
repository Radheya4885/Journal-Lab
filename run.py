"""
JournalLab Application Launcher
Starts the production FastAPI backend and directly delivers both the recommendation API and web frontend.
"""
import sys
import webbrowser
import uvicorn

if __name__ == "__main__":
    print("=" * 65)
    print("  JournalLab — AI-Based Research Paper Recommendation Platform")
    print("  Running on: http://127.0.0.1:8000")
    print("  API Docs:   http://127.0.0.1:8000/docs")
    print("=" * 65)
    
    # Optional auto-open in browser if '--open' passed
    if "--open" in sys.argv:
        webbrowser.open("http://127.0.0.1:8000")

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
