"""
AG-UI SSE Server for WAFR Pipeline.

Provides FastAPI endpoints for AG-UI compatible event streaming,
enabling real-time frontend updates during WAFR assessment.

Endpoints:
- POST /api/wafr/run - Start WAFR assessment with SSE streaming
- POST /api/wafr/process-file - Process file with SSE streaming
- GET /api/wafr/session/{session_id}/state - Get session state
- POST /api/wafr/review/{session_id}/decision - Submit review decision
- POST /api/wafr/review/{session_id}/batch-approve - Batch approve items
- POST /api/wafr/review/{session_id}/finalize - Finalize review session

Usage:
    # Run server
    uvicorn ag_ui.server:app --reload --port 8000
    
    # Or use in existing FastAPI app
    from ag_ui.server import router
    app.include_router(router)
"""

from typing import Any, Dict, Optional
from datetime import datetime
import asyncio
import uuid
import logging

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from wafr.ag_ui.emitter import WAFREventEmitter
from wafr.ag_ui.events import (
    ReviewQueueSummary,
    SynthesisProgress,
    ReviewDecisionData,
    ValidationStatus,
)
from wafr.ag_ui.state import WAFRState, SessionStatus

logger = logging.getLogger(__name__)


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="WAFR AG-UI Server",
    description="AG-UI compatible event streaming for WAFR pipeline",
    version="1.0.0",
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active sessions
active_sessions: Dict[str, WAFREventEmitter] = {}
session_states: Dict[str, WAFRState] = {}


# =============================================================================
# Request/Response Models
# =============================================================================

class RunWAFRRequest(BaseModel):
    """Request to run WAFR assessment."""
    
    thread_id: Optional[str] = Field(None, description="Thread/session ID")
    run_id: Optional[str] = Field(None, description="Run ID")
    transcript: Optional[str] = Field(None, description="Transcript text")
    transcript_path: Optional[str] = Field(None, description="Path to transcript file")
    generate_report: bool = Field(True, description="Generate PDF report")
    create_wa_workload: bool = Field(False, description="Create WA Tool workload")
    client_name: Optional[str] = Field(None, description="Client name for workload")


class ProcessFileRequest(BaseModel):
    """Request to process file."""
    
    thread_id: Optional[str] = Field(None, description="Thread/session ID")
    file_path: str = Field(..., description="Path to file")
    generate_report: bool = Field(True, description="Generate PDF report")


class ReviewDecisionRequest(BaseModel):
    """Request to submit review decision."""
    
    review_id: str = Field(..., description="Review item ID")
    decision: str = Field(..., description="Decision: APPROVE, MODIFY, or REJECT")
    reviewer_id: str = Field(..., description="Reviewer ID")
    modified_answer: Optional[str] = Field(None, description="Modified answer text")
    feedback: Optional[str] = Field(None, description="Feedback for rejection")


class BatchApproveRequest(BaseModel):
    """Request to batch approve items."""
    
    review_ids: list[str] = Field(..., description="List of review item IDs")
    reviewer_id: str = Field(..., description="Reviewer ID")


class FinalizeRequest(BaseModel):
    """Request to finalize review session."""
    
    approver_id: str = Field(..., description="Approver ID")


class StateResponse(BaseModel):
    """Response containing session state."""
    
    session_id: str
    state: Dict[str, Any]
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    service: str
    version: str
    active_sessions: int


# =============================================================================
# Health Check
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="wafr-ag-ui-server",
        version="1.0.0",
        active_sessions=len(active_sessions),
    )


# =============================================================================
# WAFR Processing Endpoints
# =============================================================================

@app.post("/api/wafr/run")
async def run_wafr_assessment(request: RunWAFRRequest):
    """
    Run WAFR assessment with AG-UI event streaming.
    
    Returns SSE stream of events.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    run_id = request.run_id or str(uuid.uuid4())
    
    # Create event emitter
    emitter = WAFREventEmitter(thread_id=thread_id, run_id=run_id)
    active_sessions[thread_id] = emitter
    session_states[thread_id] = emitter.state
    
    async def event_generator():
        """Generate SSE events."""
        try:
            # Import here to avoid circular imports
            from agents.orchestrator import create_orchestrator
            
            # Create orchestrator (you'll need to enhance it to accept emitter)
            orchestrator = create_orchestrator()
            
            # Start run
            await emitter.run_started()
            
            # Determine input source
            if request.transcript:
                transcript = request.transcript
            elif request.transcript_path:
                with open(request.transcript_path, 'r', encoding='utf-8') as f:
                    transcript = f.read()
            else:
                await emitter.run_error("No transcript or transcript_path provided")
                async for event_data in emitter.stream_events():
                    yield event_data
                return
            
            # Process in background task
            async def process():
                try:
                    # Process transcript with events
                    # Note: The orchestrator needs to be enhanced to use the emitter
                    # This is a simplified version
                    await emitter.step_started("understanding")
                    
                    # Simulate processing (replace with actual orchestrator call)
                    result = orchestrator.process_transcript(
                        transcript=transcript,
                        session_id=thread_id,
                        generate_report=request.generate_report,
                        create_wa_workload=request.create_wa_workload,
                        client_name=request.client_name,
                    )
                    
                    # Emit completion events based on result
                    if result.get("status") == "completed":
                        # Update state with results
                        if "steps" in result:
                            steps = result["steps"]
                            
                            # Understanding
                            if "understanding" in steps:
                                insights = steps["understanding"].get("insights", [])
                                emitter.state.set_insights_count(len(insights))
                            
                            # Mapping
                            if "mapping" in steps:
                                mappings = steps["mapping"].get("mappings", [])
                                emitter.state.content.questions_answered = len(mappings)
                            
                            # Gap detection
                            if "gap_detection" in steps:
                                gaps = steps["gap_detection"].get("gaps", [])
                                emitter.state.content.gaps_count = len(gaps)
                            
                            # Synthesis
                            if "synthesis" in steps:
                                synthesized = steps["synthesis"].get("synthesized_answers", [])
                                emitter.state.content.synthesized_count = len(synthesized)
                        
                        await emitter.state_snapshot()
                        await emitter.run_finished()
                    else:
                        await emitter.run_error(
                            result.get("error", "Unknown error"),
                            code="PROCESSING_ERROR"
                        )
                        
                except Exception as e:
                    logger.error(f"Processing error: {e}", exc_info=True)
                    await emitter.run_error(str(e), code="PROCESSING_ERROR")
            
            # Start processing task
            asyncio.create_task(process())
            
            # Stream events
            async for event_data in emitter.stream_events():
                yield event_data
                
        except Exception as e:
            logger.error(f"Event generation error: {e}", exc_info=True)
            yield f"data: {{'type': 'ERROR', 'message': '{str(e)}'}}\n\n"
        finally:
            # Cleanup
            if thread_id in active_sessions:
                del active_sessions[thread_id]
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.post("/api/wafr/process-file")
async def process_file(request: ProcessFileRequest):
    """
    Process file with AG-UI event streaming.
    
    Returns SSE stream of events.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    
    # Create event emitter
    emitter = WAFREventEmitter(thread_id=thread_id, run_id=run_id)
    active_sessions[thread_id] = emitter
    session_states[thread_id] = emitter.state
    
    async def event_generator():
        """Generate SSE events."""
        try:
            from agents.orchestrator import create_orchestrator
            
            orchestrator = create_orchestrator()
            
            await emitter.run_started()
            await emitter.step_started("file_processing")
            
            # Process file
            try:
                result = orchestrator.process_file(
                    file_path=request.file_path,
                    session_id=thread_id,
                    generate_report=request.generate_report,
                )
                
                await emitter.step_finished("file_processing", {
                    "status": result.get("status", "unknown")
                })
                
                if result.get("status") == "completed":
                    await emitter.state_snapshot()
                    await emitter.run_finished()
                else:
                    await emitter.run_error(
                        result.get("error", "Unknown error"),
                        code="PROCESSING_ERROR"
                    )
                    
            except Exception as e:
                await emitter.run_error(str(e), code="PROCESSING_ERROR")
            
            async for event_data in emitter.stream_events():
                yield event_data
                
        except Exception as e:
            logger.error(f"File processing error: {e}", exc_info=True)
            yield f"data: {{'type': 'ERROR', 'message': '{str(e)}'}}\n\n"
        finally:
            if thread_id in active_sessions:
                del active_sessions[thread_id]
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =============================================================================
# State Endpoints
# =============================================================================

@app.get("/api/wafr/session/{session_id}/state", response_model=StateResponse)
async def get_session_state(session_id: str):
    """Get current state for a session."""
    if session_id not in session_states:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = session_states[session_id]
    
    return StateResponse(
        session_id=session_id,
        state=state.to_snapshot(),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/api/wafr/sessions")
async def list_sessions():
    """List all active sessions."""
    return {
        "sessions": [
            {
                "session_id": session_id,
                "status": state.session.status,
                "current_step": state.pipeline.current_step,
                "progress": state.pipeline.progress_percentage,
            }
            for session_id, state in session_states.items()
        ],
        "count": len(session_states),
    }


# =============================================================================
# Review Endpoints
# =============================================================================

@app.post("/api/wafr/review/{session_id}/decision")
async def submit_review_decision(session_id: str, request: ReviewDecisionRequest):
    """
    Submit review decision for an item.
    
    Emits review decision event if session has active emitter.
    """
    try:
        # Import here to avoid circular imports
        from agents.review_orchestrator import ReviewOrchestrator
        from models.review_item import ReviewDecision
        
        # Get review orchestrator (you may need to cache this)
        # For now, we'll create a simple response
        
        # Create decision data
        decision_data = ReviewDecisionData(
            review_id=request.review_id,
            question_id="",  # Would be populated from review item
            decision=request.decision,
            reviewer_id=request.reviewer_id,
            modified_answer=request.modified_answer,
            feedback=request.feedback,
        )
        
        # Emit event if session has emitter
        if session_id in active_sessions:
            emitter = active_sessions[session_id]
            await emitter.review_decision(decision_data)
        
        return {
            "status": "success",
            "review_id": request.review_id,
            "decision": request.decision,
            "session_id": session_id,
        }
        
    except Exception as e:
        logger.error(f"Review decision error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/wafr/review/{session_id}/batch-approve")
async def batch_approve(session_id: str, request: BatchApproveRequest):
    """
    Batch approve multiple review items.
    
    Emits batch approval event if session has active emitter.
    """
    try:
        approved_count = len(request.review_ids)
        remaining_count = 0  # Would be calculated from review session
        
        # Emit event if session has emitter
        if session_id in active_sessions:
            emitter = active_sessions[session_id]
            await emitter.batch_approve_completed(
                session_id=session_id,
                approved_count=approved_count,
                remaining_count=remaining_count,
            )
        
        return {
            "status": "success",
            "approved_count": approved_count,
            "remaining_count": remaining_count,
            "session_id": session_id,
        }
        
    except Exception as e:
        logger.error(f"Batch approve error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/wafr/review/{session_id}/finalize")
async def finalize_review_session(session_id: str, request: FinalizeRequest):
    """
    Finalize review session.
    
    Validates requirements and emits finalization event.
    """
    try:
        # Validate session exists
        if session_id not in session_states:
            raise HTTPException(status_code=404, detail="Session not found")
        
        state = session_states[session_id]
        
        # Check if can finalize (simplified validation)
        if state.review.pending_count > 0:
            validation_status = ValidationStatus(
                can_finalize=False,
                issues=[f"{state.review.pending_count} items still pending review"],
                authenticity_score=state.scores.authenticity_score,
                pending_count=state.review.pending_count,
            )
            
            if session_id in active_sessions:
                await active_sessions[session_id].validation_status(validation_status)
            
            return {
                "status": "validation_failed",
                "can_finalize": False,
                "issues": validation_status.issues,
            }
        
        # Finalize
        authenticity_score = state.scores.authenticity_score
        total_items = state.review.total_items
        approved = state.review.approved_count
        modified = state.review.modified_count
        
        # Emit finalization event
        if session_id in active_sessions:
            await active_sessions[session_id].session_finalized(
                session_id=session_id,
                authenticity_score=authenticity_score,
                total_items=total_items,
                approved=approved,
                modified=modified,
            )
        
        # Update state
        state.session.status = SessionStatus.FINALIZED.value
        
        return {
            "status": "success",
            "session_id": session_id,
            "authenticity_score": authenticity_score,
            "summary": {
                "total_items": total_items,
                "approved": approved,
                "modified": modified,
            },
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Finalize error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# WebSocket Support (Optional)
# =============================================================================

@app.websocket("/ws/wafr/{session_id}")
async def websocket_endpoint(websocket, session_id: str):
    """
    WebSocket endpoint for bidirectional communication.
    
    Alternative to SSE for clients that prefer WebSocket.
    """
    from fastapi import WebSocket, WebSocketDisconnect
    
    await websocket.accept()
    
    # Get or create emitter
    if session_id not in active_sessions:
        emitter = WAFREventEmitter(thread_id=session_id)
        active_sessions[session_id] = emitter
        session_states[session_id] = emitter.state
    else:
        emitter = active_sessions[session_id]
    
    try:
        # Send initial state
        await websocket.send_json({
            "type": "STATE_SNAPSHOT",
            "snapshot": emitter.state.to_snapshot(),
        })
        
        # Listen for events and client messages
        while True:
            try:
                # Check for client messages (non-blocking)
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=0.1
                )
                
                # Handle client commands
                if data.get("type") == "GET_STATE":
                    await websocket.send_json({
                        "type": "STATE_SNAPSHOT",
                        "snapshot": emitter.state.to_snapshot(),
                    })
                    
            except asyncio.TimeoutError:
                # No client message, check for events
                if not emitter.event_queue.empty():
                    event = await emitter.event_queue.get()
                    await websocket.send_json(event.to_dict())
                elif emitter.is_finished:
                    break
                    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        if session_id in active_sessions and active_sessions[session_id].is_finished:
            del active_sessions[session_id]


# =============================================================================
# Router for Integration
# =============================================================================

from fastapi import APIRouter

router = APIRouter(prefix="/api/wafr", tags=["WAFR AG-UI"])

# Copy endpoints to router for integration into existing apps
router.add_api_route("/run", run_wafr_assessment, methods=["POST"])
router.add_api_route("/process-file", process_file, methods=["POST"])
router.add_api_route("/session/{session_id}/state", get_session_state, methods=["GET"])
router.add_api_route("/sessions", list_sessions, methods=["GET"])
router.add_api_route("/review/{session_id}/decision", submit_review_decision, methods=["POST"])
router.add_api_route("/review/{session_id}/batch-approve", batch_approve, methods=["POST"])
router.add_api_route("/review/{session_id}/finalize", finalize_review_session, methods=["POST"])


# =============================================================================
# Main Entry Point
# =============================================================================

def create_app() -> FastAPI:
    """Create FastAPI application with AG-UI endpoints."""
    return app


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "ag_ui.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

