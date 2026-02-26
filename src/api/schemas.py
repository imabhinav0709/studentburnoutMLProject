"""
Pydantic Schemas for the FastAPI backend
"""

from __future__ import annotations
from pydantic import BaseModel, Field


class StudentInput(BaseModel):
    """Request body for a single student prediction."""
    study_hours_per_day: float = Field(..., ge=0, le=14, description="Daily study hours")
    sleep_hours: float = Field(..., ge=3, le=12, description="Daily sleep hours")
    attendance_pct: float = Field(..., ge=0, le=100, description="Attendance %")
    previous_gpa: float = Field(..., ge=0, le=4, description="Previous GPA (0–4)")
    stress_level: float = Field(..., ge=1, le=10, description="Stress level (1–10)")
    extracurricular_hrs: float = Field(..., ge=0, le=10, description="Weekly extracurricular hours")
    screen_time_hrs: float = Field(..., ge=0, le=12, description="Daily screen time hours")
    exercise_freq: float = Field(..., ge=0, le=7, description="Weekly exercise sessions")
    gender: str = Field(..., description="Gender: Male, Female, or Other")
    has_part_time_job: int = Field(..., ge=0, le=1, description="1 = yes, 0 = no")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "study_hours_per_day": 5.0,
                    "sleep_hours": 7.5,
                    "attendance_pct": 85.0,
                    "previous_gpa": 3.5,
                    "stress_level": 4.0,
                    "extracurricular_hrs": 2.0,
                    "screen_time_hrs": 3.0,
                    "exercise_freq": 4.0,
                    "gender": "Female",
                    "has_part_time_job": 0,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    predicted_score: float
    confidence_interval: list[float]
    top_factors: list[dict]
    model_used: str


class StatisticsResponse(BaseModel):
    results: list[dict]


class ComparisonResponse(BaseModel):
    comparison: list[dict]
    model_used: str
