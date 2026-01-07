"""
Review Orchestrator - Manages human review workflow for AI-synthesized answers.

Coordinates the HITL (Human-in-the-Loop) validation process for WAFR assessments,
enabling efficient batch review, modification, and approval of AI-generated answers.

Key Features:
- Session management with persistence
- Batch approval for high-confidence answers
- Review queue grouping by confidence level
- Authenticity scoring
- Finalization validation
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from agents.config import (
    hitl_settings,
    HITL_AUTO_APPROVE_THRESHOLD,
    HITL_QUICK_REVIEW_THRESHOLD,
    HITL_MIN_AUTHENTICITY_SCORE,
    HITL_SECURITY_MIN_APPROVAL,
    HITL_RELIABILITY_MIN_APPROVAL,
    HITL_MAX_RESYNTHESIS_ATTEMPTS,
)
from agents.errors import (
    SessionNotFoundError,
    ReviewItemNotFoundError,
    FinalizationError,
    ReviewAlreadySubmittedError,
)
from models.review_item import ReviewDecision, ReviewItem, ReviewStatus
from models.synthesized_answer import SynthesizedAnswer
from models.validation_record import ValidationRecord

logger = logging.getLogger(__name__)


# =============================================================================
# Review Session Model
# =============================================================================

@dataclass
class ReviewSession:
    """A review session containing items for human validation."""
    session_id: str
    created_at: datetime
    items: List[ReviewItem] = field(default_factory=list)
    status: str = "ACTIVE"  # ACTIVE, FINALIZED, CANCELLED
    transcript_answers_count: int = 0
    
    @property
    def pending_count(self) -> int:
        return len([i for i in self.items if i.status == ReviewStatus.PENDING])
    
    @property
    def approved_count(self) -> int:
        return len([i for i in self.items if i.status in [ReviewStatus.APPROVED, ReviewStatus.MODIFIED]])
    
    @property
    def rejected_count(self) -> int:
        return len([i for i in self.items if i.status == ReviewStatus.REJECTED])
    
    @property
    def progress_percentage(self) -> float:
        if not self.items:
            return 100.0
        return (self.approved_count / len(self.items)) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "items": [item.to_dict() for item in self.items],
            "transcript_answers_count": self.transcript_answers_count,
            "summary": {
                "total": len(self.items),
                "pending": self.pending_count,
                "approved": self.approved_count,
                "rejected": self.rejected_count,
                "progress_percentage": self.progress_percentage,
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewSession":
        """Reconstruct session from dictionary."""
        items = [ReviewItem.from_dict(item) for item in data.get("items", [])]
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        return cls(
            session_id=data["session_id"],
            created_at=created_at,
            items=items,
            status=data.get("status", "ACTIVE"),
            transcript_answers_count=data.get("transcript_answers_count", 0),
        )


# =============================================================================
# Review Orchestrator
# =============================================================================

class ReviewOrchestrator:
    """
    Orchestrates human review of AI-synthesized answers.
    
    Features:
    - Create and manage review sessions
    - Batch approval for high-confidence answers
    - Review queue grouping by confidence level
    - Re-synthesis integration for rejected answers
    - Authenticity scoring and validation
    """
    
    def __init__(self, synthesis_agent=None, storage=None):
        """
        Initialize Review Orchestrator.
        
        Args:
            synthesis_agent: Optional AnswerSynthesisAgent for re-synthesis
            storage: Optional ReviewStorage for persistence
        """
        self.synthesis_agent = synthesis_agent
        self.storage = storage
        self.sessions: Dict[str, ReviewSession] = {}
        logger.info("ReviewOrchestrator initialized")
    
    # =========================================================================
    # Session Management
    # =========================================================================
    
    def create_review_session(
        self,
        synthesized_answers: List[SynthesizedAnswer],
        session_id: Optional[str] = None,
        transcript_answers_count: int = 0,
    ) -> ReviewSession:
        """
        Create a new review session from synthesized answers.
        
        Args:
            synthesized_answers: List of AI-synthesized answers
            session_id: Optional custom session ID
            transcript_answers_count: Count of transcript-based answers (no review needed)
            
        Returns:
            Created ReviewSession
        """
        session_id = session_id or str(uuid.uuid4())
        
        items = []
        for answer in synthesized_answers:
            if isinstance(answer, dict):
                answer = SynthesizedAnswer.from_dict(answer)
            
            item = ReviewItem(
                review_id=str(uuid.uuid4()),
                question_id=answer.question_id,
                pillar=answer.pillar,
                criticality=answer.criticality,
                synthesized_answer=answer,
            )
            items.append(item)
        
        # Sort by criticality (HIGH first) then by confidence (low first for priority)
        items.sort(key=lambda x: (
            {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(x.criticality.upper(), 2),
            x.synthesized_answer.confidence,
        ))
        
        session = ReviewSession(
            session_id=session_id,
            created_at=datetime.utcnow(),
            items=items,
            transcript_answers_count=transcript_answers_count,
        )
        
        self.sessions[session_id] = session
        
        # Persist if storage available
        if self.storage:
            self.storage.save_session(session.to_dict())
        
        logger.info(f"Created review session {session_id} with {len(items)} items")
        return session
    
    def get_session(self, session_id: str) -> Optional[ReviewSession]:
        """Get session by ID, loading from storage if needed."""
        if session_id in self.sessions:
            return self.sessions[session_id]
        
        # Try loading from storage
        if self.storage:
            data = self.storage.load_session(session_id)
            if data:
                session = ReviewSession.from_dict(data)
                self.sessions[session_id] = session
                return session
        
        return None
    
    def get_session_progress(self, session_id: str) -> Dict[str, Any]:
        """Get detailed progress for a session."""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        by_pillar = {}
        for item in session.items:
            pillar = item.pillar
            if pillar not in by_pillar:
                by_pillar[pillar] = {"total": 0, "approved": 0, "modified": 0, "rejected": 0, "pending": 0}
            by_pillar[pillar]["total"] += 1
            if item.status == ReviewStatus.APPROVED:
                by_pillar[pillar]["approved"] += 1
            elif item.status == ReviewStatus.MODIFIED:
                by_pillar[pillar]["modified"] += 1
            elif item.status == ReviewStatus.REJECTED:
                by_pillar[pillar]["rejected"] += 1
            else:
                by_pillar[pillar]["pending"] += 1
        
        by_criticality = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for item in session.items:
            crit = item.criticality.upper()
            if crit in by_criticality:
                by_criticality[crit] += 1
        
        # Group by confidence
        by_confidence = {"high": [], "medium": [], "low": []}
        for item in session.items:
            conf = item.synthesized_answer.confidence
            if conf >= HITL_AUTO_APPROVE_THRESHOLD:
                by_confidence["high"].append(item.review_id)
            elif conf >= HITL_QUICK_REVIEW_THRESHOLD:
                by_confidence["medium"].append(item.review_id)
            else:
                by_confidence["low"].append(item.review_id)
        
        return {
            "session_id": session_id,
            "status": session.status,
            "total_items": len(session.items),
            "pending": session.pending_count,
            "approved": session.approved_count,
            "rejected": session.rejected_count,
            "progress_percentage": round(session.progress_percentage, 1),
            "transcript_answers_count": session.transcript_answers_count,
            "by_pillar": by_pillar,
            "by_criticality": by_criticality,
            "by_confidence": {
                "high_count": len(by_confidence["high"]),
                "medium_count": len(by_confidence["medium"]),
                "low_count": len(by_confidence["low"]),
            },
        }
    
    # =========================================================================
    # Review Queue Management
    # =========================================================================
    
    def get_review_queue(
        self,
        session_id: str,
        filter_status: Optional[ReviewStatus] = None,
        filter_pillar: Optional[str] = None,
    ) -> List[ReviewItem]:
        """Get review queue with optional filtering."""
        session = self.get_session(session_id)
        if not session:
            return []
        
        items = session.items
        if filter_status:
            items = [i for i in items if i.status == filter_status]
        if filter_pillar:
            items = [i for i in items if i.pillar == filter_pillar]
        return items
    
    def get_batch_review_queue(self, session_id: str) -> Dict[str, Any]:
        """
        Get review queue organized for batch processing.
        
        Groups items by confidence level for efficient review:
        - high_confidence: Auto-approve candidates (≥0.75)
        - medium_confidence: Quick review (0.50-0.74)
        - low_confidence: Detailed review (<0.50)
        """
        session = self.get_session(session_id)
        if not session:
            return {"high_confidence": [], "medium_confidence": [], "low_confidence": [], "by_pillar": {}}
        
        pending_items = [i for i in session.items if i.status == ReviewStatus.PENDING]
        
        high_conf = []
        medium_conf = []
        low_conf = []
        by_pillar = {}
        
        for item in pending_items:
            conf = item.synthesized_answer.confidence
            
            if conf >= HITL_AUTO_APPROVE_THRESHOLD:
                high_conf.append(item)
            elif conf >= HITL_QUICK_REVIEW_THRESHOLD:
                medium_conf.append(item)
            else:
                low_conf.append(item)
            
            pillar = item.pillar
            if pillar not in by_pillar:
                by_pillar[pillar] = []
            by_pillar[pillar].append(item)
        
        return {
            "high_confidence": high_conf,
            "medium_confidence": medium_conf,
            "low_confidence": low_conf,
            "by_pillar": by_pillar,
            "summary": {
                "total_pending": len(pending_items),
                "auto_approve_eligible": len(high_conf),
                "quick_review": len(medium_conf),
                "detailed_review": len(low_conf),
            },
        }
    
    def get_auto_approve_candidates(
        self,
        session_id: str,
        threshold: float = None,
    ) -> List[str]:
        """
        Get review IDs eligible for auto-approval.
        
        Args:
            session_id: Session identifier
            threshold: Minimum confidence (default: from config)
            
        Returns:
            List of review_ids eligible for auto-approval
        """
        threshold = threshold or HITL_AUTO_APPROVE_THRESHOLD
        session = self.get_session(session_id)
        if not session:
            return []
        
        return [
            item.review_id
            for item in session.items
            if item.status == ReviewStatus.PENDING
            and item.synthesized_answer.confidence >= threshold
        ]
    
    # =========================================================================
    # Review Decision Processing
    # =========================================================================
    
    def submit_review(
        self,
        session_id: str,
        review_id: str,
        decision: ReviewDecision,
        reviewer_id: str,
        modified_answer: Optional[str] = None,
        feedback: Optional[str] = None,
    ) -> ReviewItem:
        """
        Submit a review decision for an item.
        
        Args:
            session_id: Session identifier
            review_id: Review item ID
            decision: APPROVE, MODIFY, or REJECT
            reviewer_id: ID of the reviewer
            modified_answer: Modified text (for MODIFY)
            feedback: Rejection reason (for REJECT)
            
        Returns:
            Updated ReviewItem
        """
        session = self.get_session(session_id)
        if not session:
            raise SessionNotFoundError(session_id)
        
        item = next((i for i in session.items if i.review_id == review_id), None)
        if not item:
            raise ReviewItemNotFoundError(session_id, review_id)
        
        # Update item
        item.reviewer_id = reviewer_id
        item.reviewed_at = datetime.utcnow()
        item.decision = decision
        
        if decision == ReviewDecision.APPROVE:
            item.status = ReviewStatus.APPROVED
            logger.info(f"Answer approved for {item.question_id}")
            
        elif decision == ReviewDecision.MODIFY:
            item.status = ReviewStatus.MODIFIED
            item.modified_answer = modified_answer
            logger.info(f"Answer modified for {item.question_id}")
            
        elif decision == ReviewDecision.REJECT:
            item.rejection_feedback = feedback
            item.revision_count += 1
            
            if item.revision_count <= HITL_MAX_RESYNTHESIS_ATTEMPTS:
                # Try re-synthesis if agent available
                if self.synthesis_agent and hasattr(self.synthesis_agent, 'resynthesiswith_feedback'):
                    logger.info(f"Re-synthesizing answer for {item.question_id}")
                    try:
                        new_answer = self.synthesis_agent.re_synthesize_with_feedback(
                            original=item.synthesized_answer,
                            feedback=feedback or "Please provide a more accurate answer.",
                            context={},
                        )
                        item.synthesized_answer = new_answer
                        item.status = ReviewStatus.PENDING
                        logger.info(f"Re-synthesis complete for {item.question_id}")
                    except Exception as e:
                        logger.error(f"Re-synthesis failed: {e}")
                        item.status = ReviewStatus.REJECTED
                else:
                    # Without synthesis agent, mark as pending for manual update
                    item.status = ReviewStatus.PENDING
                    logger.warning(f"No synthesis agent - {item.question_id} pending manual update")
            else:
                item.status = ReviewStatus.REJECTED
                logger.warning(f"Max revisions reached for {item.question_id}")
        
        # Persist changes
        if self.storage:
            self.storage.update_item(session_id, review_id, item.to_dict())
        
        return item
    
    def batch_approve(
        self,
        session_id: str,
        review_ids: List[str],
        reviewer_id: str,
    ) -> Dict[str, Any]:
        """
        Approve multiple answers in batch.
        
        Args:
            session_id: Session identifier
            review_ids: List of review IDs to approve
            reviewer_id: ID of the reviewer
            
        Returns:
            Result summary with counts and any errors
        """
        session = self.get_session(session_id)
        if not session:
            raise SessionNotFoundError(session_id)
        
        approved = []
        errors = []
        
        for review_id in review_ids:
            try:
                item = self.submit_review(
                    session_id=session_id,
                    review_id=review_id,
                    decision=ReviewDecision.APPROVE,
                    reviewer_id=reviewer_id,
                )
                approved.append(item.question_id)
            except Exception as e:
                errors.append({"review_id": review_id, "error": str(e)})
        
        logger.info(f"Batch approved {len(approved)} items in session {session_id}")
        
        return {
            "approved_count": len(approved),
            "approved_questions": approved,
            "error_count": len(errors),
            "errors": errors,
        }
    
    def batch_approve_high_confidence(
        self,
        session_id: str,
        reviewer_id: str,
        min_confidence: float = None,
    ) -> Dict[str, Any]:
        """
        Auto-approve all high-confidence pending answers.
        
        Args:
            session_id: Session identifier
            reviewer_id: ID of the reviewer
            min_confidence: Minimum confidence (default: from config)
            
        Returns:
            Result summary
        """
        candidates = self.get_auto_approve_candidates(session_id, min_confidence)
        
        if not candidates:
            return {"approved_count": 0, "message": "No high-confidence items to approve"}
        
        return self.batch_approve(session_id, candidates, reviewer_id)
    
    # =========================================================================
    # Validation and Finalization
    # =========================================================================
    
    def validate_for_finalization(self, session_id: str) -> Tuple[bool, List[str]]:
        """
        Check if session meets finalization requirements.
        
        Requirements:
        - All items reviewed (no pending)
        - All HIGH criticality items approved
        - Security pillar >= 90% approval
        - Reliability pillar >= 90% approval
        """
        session = self.get_session(session_id)
        if not session:
            return False, ["Session not found"]
        
        issues = []
        
        # Check all items reviewed
        pending = [i for i in session.items if i.status == ReviewStatus.PENDING]
        if pending:
            issues.append(f"{len(pending)} items still pending review")
        
        # Check HIGH criticality all approved
        high_crit_not_approved = [
            i for i in session.items
            if i.criticality.upper() == "HIGH"
            and i.status not in [ReviewStatus.APPROVED, ReviewStatus.MODIFIED]
        ]
        if high_crit_not_approved:
            issues.append(f"{len(high_crit_not_approved)} HIGH criticality items not approved")
        
        # Check Security pillar
        sec_items = [i for i in session.items if i.pillar == "SEC"]
        if sec_items:
            sec_approved = [i for i in sec_items if i.status in [ReviewStatus.APPROVED, ReviewStatus.MODIFIED]]
            sec_rate = len(sec_approved) / len(sec_items)
            if sec_rate < HITL_SECURITY_MIN_APPROVAL:
                issues.append(f"Security pillar has {sec_rate*100:.1f}% approval rate (requires {HITL_SECURITY_MIN_APPROVAL*100}%)")
        
        # Check Reliability pillar
        rel_items = [i for i in session.items if i.pillar == "REL"]
        if rel_items:
            rel_approved = [i for i in rel_items if i.status in [ReviewStatus.APPROVED, ReviewStatus.MODIFIED]]
            rel_rate = len(rel_approved) / len(rel_items)
            if rel_rate < HITL_RELIABILITY_MIN_APPROVAL:
                issues.append(f"Reliability pillar has {rel_rate*100:.1f}% approval rate (requires {HITL_RELIABILITY_MIN_APPROVAL*100}%)")
        
        return len(issues) == 0, issues
    
    def calculate_authenticity_score(self, session: ReviewSession) -> float:
        """
        Calculate overall authenticity score.
        
        Scoring:
        - Transcript evidence: 100%
        - Human modified: 95%
        - AI approved: 85%
        - Rejected/pending: 0%
        """
        if not session.items and session.transcript_answers_count == 0:
            return 0.0
        
        total_weight = 0.0
        total_items = len(session.items) + session.transcript_answers_count
        
        # Transcript answers get 100%
        total_weight += session.transcript_answers_count * 1.0
        
        # Weight by review decision
        for item in session.items:
            if item.status == ReviewStatus.MODIFIED:
                total_weight += 0.95
            elif item.status == ReviewStatus.APPROVED:
                total_weight += 0.85
        
        return (total_weight / total_items) * 100 if total_items > 0 else 0.0
    
    def finalize_session(self, session_id: str, approver_id: str) -> ValidationRecord:
        """
        Finalize a review session for report generation.
        
        Args:
            session_id: Session identifier
            approver_id: ID of the final approver
            
        Returns:
            ValidationRecord with finalization details
        """
        can_finalize, issues = self.validate_for_finalization(session_id)
        if not can_finalize:
            raise FinalizationError(session_id, issues)
        
        session = self.sessions[session_id]
        session.status = "FINALIZED"
        
        # Calculate metrics
        authenticity_score = self.calculate_authenticity_score(session)
        
        pillar_coverage = {}
        for item in session.items:
            pillar = item.pillar
            if pillar not in pillar_coverage:
                pillar_coverage[pillar] = {"total": 0, "approved": 0}
            pillar_coverage[pillar]["total"] += 1
            if item.status in [ReviewStatus.APPROVED, ReviewStatus.MODIFIED]:
                pillar_coverage[pillar]["approved"] += 1
        
        pillar_coverage_pct = {
            pillar: (data["approved"] / data["total"] * 100) if data["total"] > 0 else 0
            for pillar, data in pillar_coverage.items()
        }
        
        # Calculate review duration
        review_duration = None
        reviewed_items = [i for i in session.items if i.reviewed_at]
        if reviewed_items:
            first_review = min(i.reviewed_at for i in reviewed_items)
            last_review = max(i.reviewed_at for i in reviewed_items)
            review_duration = int((last_review - first_review).total_seconds())
        
        revision_attempts = sum(item.revision_count for item in session.items)
        
        validation_record = ValidationRecord(
            session_id=session_id,
            finalized_at=datetime.utcnow(),
            approver_id=approver_id,
            total_items=len(session.items),
            approved_count=len([i for i in session.items if i.status == ReviewStatus.APPROVED]),
            modified_count=len([i for i in session.items if i.status == ReviewStatus.MODIFIED]),
            rejected_count=len([i for i in session.items if i.status == ReviewStatus.REJECTED]),
            authenticity_score=authenticity_score,
            pillar_coverage=pillar_coverage_pct,
            review_duration_seconds=review_duration,
            revision_attempts=revision_attempts,
        )
        
        # Persist
        if self.storage:
            self.storage.save_session(session.to_dict())
            self.storage.save_validation_record(validation_record.to_dict())
        
        logger.info(f"Session {session_id} finalized (authenticity: {authenticity_score:.1f}%)")
        return validation_record
    
    # =========================================================================
    # Export Methods
    # =========================================================================
    
    def get_validated_answers(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get validated answers for report generation.
        
        Returns answers that have been approved or modified,
        in a format suitable for the report agent.
        """
        session = self.get_session(session_id)
        if not session:
            return []
        
        validated = []
        for item in session.items:
            if item.status not in [ReviewStatus.APPROVED, ReviewStatus.MODIFIED]:
                continue
            
            answer = item.synthesized_answer
            answer_text = item.modified_answer if item.modified_answer else answer.synthesized_answer
            
            validated.append({
                "question_id": item.question_id,
                "question_text": answer.question_text,
                "pillar": item.pillar,
                "answer_content": answer_text,
                "source": "AI_MODIFIED" if item.status == ReviewStatus.MODIFIED else "AI_VALIDATED",
                "confidence": answer.confidence,
                "synthesis_method": answer.synthesis_method.value,
                "reasoning_chain": answer.reasoning_chain,
                "assumptions": answer.assumptions,
                "evidence_quotes": [
                    {"text": eq.text, "location": eq.location, "relevance": eq.relevance}
                    for eq in answer.evidence_quotes
                ],
                "requires_attention": answer.requires_attention,
                "review_metadata": {
                    "reviewer_id": item.reviewer_id,
                    "reviewed_at": item.reviewed_at.isoformat() if item.reviewed_at else None,
                    "decision": item.decision.value if item.decision else None,
                    "revision_count": item.revision_count,
                },
            })
        
        return validated


# =============================================================================
# Factory Function
# =============================================================================

def create_review_orchestrator(
    synthesis_agent=None,
    storage=None,
) -> ReviewOrchestrator:
    """
    Factory function to create Review Orchestrator.
    
    Args:
        synthesis_agent: Optional AnswerSynthesisAgent for re-synthesis
        storage: Optional ReviewStorage for persistence
        
    Returns:
        Configured ReviewOrchestrator instance
    """
    return ReviewOrchestrator(synthesis_agent=synthesis_agent, storage=storage)

