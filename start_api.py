"""
Start the FastAPI server
Run: python start_api.py
"""

import uvicorn

if __name__ == "__main__":
    print("🚀 Starting Nordic Privacy AI API Server...")
    print("📍 API will be available at: http://localhost:8000")
    print("📖 Interactive docs at: http://localhost:8000/docs")
    print("🔗 Frontend should run at: http://localhost:3000")
    print("\nPress CTRL+C to stop\n")
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
