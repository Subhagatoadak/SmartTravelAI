# webhook_swagger.py
from typing import Optional, Literal, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, Field

app = FastAPI(
    title="Synthesia Webhook Local Tester",
    description="Use these endpoints via Swagger (/docs) to simulate and verify webhook flows locally.",
    version="1.0.0",
)

# Simple in-memory store
VIDEO_STATUS: Dict[str, Dict[str, Any]] = {}


# ---------- Pydantic models so Swagger shows nice schemas ----------
class WebhookData(BaseModel):
    id: str = Field(..., description="Video ID from Synthesia")
    status: Literal["in_progress", "complete", "failed", "queued", "unknown"] = "unknown"
    download: Optional[HttpUrl] = Field(
        default=None,
        description="Time-limited download URL returned when status=complete"
    )

class WebhookPayload(BaseModel):
    event: Literal["video.completed", "video.failed", "video.queued", "video.in_progress", "video.updated"]
    data: WebhookData

    class Config:
        schema_extra = {
            "example": {
                "event": "video.completed",
                "data": {
                    "id": "test-video-123",
                    "status": "complete",
                    "download": "https://example.com/my_test_video.mp4"
                }
            }
        }


# ---------- Endpoints ----------
@app.post("/webhooks/synthesia", tags=["Webhook"])
async def synthesia_webhook(payload: WebhookPayload):
    """
    Simulate the Synthesia webhook (use this from Swagger while local).
    Stores the latest status in memory keyed by `video_id`.
    """
    vid = payload.data.id
    VIDEO_STATUS[vid] = {
        "event": payload.event,
        "status": payload.data.status,
        "download": payload.data.download,
        "payload": payload.dict(),
    }
    # Return 2xx quickly (as a real webhook should)
    return JSONResponse({"ok": True, "stored_for": vid})


@app.get("/videos/{video_id}/status", tags=["Query"])
async def get_video_status(video_id: str):
    """
    Query what the webhook stored for a given `video_id`.
    """
    record = VIDEO_STATUS.get(video_id)
    if not record:
        # You can return 404 or a friendly object; choose what you prefer.
        raise HTTPException(status_code=404, detail=f"No status stored for video_id '{video_id}'. "
                                                    f"POST /webhooks/synthesia first from Swagger.")
    return record


@app.get("/videos", tags=["Query"])
async def list_all_cached_video_ids():
    """Convenience endpoint to see what IDs are currently in the in-memory store."""
    return {"count": len(VIDEO_STATUS), "video_ids": list(VIDEO_STATUS.keys())}
