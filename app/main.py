from fastapi import FastAPI
from app.api import auth, jobs
from app.database import Base, engine

# Create all tables in the database (if not exist)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Interview System",
    description="Backend API for AI-powered interview question generation.",
    version="1.0.0"
)

# Root endpoint to check if API is up
@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "AI Interview System API is running!"}

# Include authentication routes with prefix /auth
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])

# Include jobs routes with prefix /jobs
app.include_router(jobs.router, prefix="/jobs", tags=["Jobs"])
