"""
Review Storage - Persistence layer for HITL review sessions.

Provides:
- ReviewStorage: Abstract base class defining the storage interface
- InMemoryReviewStorage: In-memory implementation for development/testing
- FileReviewStorage: File-based implementation with JSON persistence
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Abstract Base Class
# =============================================================================

class ReviewStorage(ABC):
    """
    Abstract interface for review session storage.
    
    All storage implementations must implement these methods to ensure
    consistent behavior across different backends (memory, file, DynamoDB, etc.)
    """
    
    @abstractmethod
    def save_session(self, session_data: Dict[str, Any]) -> None:
        """
        Save or update a review session.
        
        Args:
            session_data: Session data dictionary (serialized ReviewSession)
        """
        pass
    
    @abstractmethod
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a review session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data dictionary or None if not found
        """
        pass
    
    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a review session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    def list_sessions(
        self,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List review sessions with optional filtering.
        
        Args:
            status: Optional status filter ("ACTIVE", "FINALIZED", etc.)
            limit: Maximum number of sessions to return
            
        Returns:
            List of session data dictionaries
        """
        pass
    
    @abstractmethod
    def update_item(
        self,
        session_id: str,
        review_id: str,
        item_data: Dict[str, Any],
    ) -> bool:
        """
        Update a specific review item within a session.
        
        Args:
            session_id: Session identifier
            review_id: Review item identifier
            item_data: Updated item data
            
        Returns:
            True if updated, False if not found
        """
        pass
    
    @abstractmethod
    def save_validation_record(self, record: Dict[str, Any]) -> None:
        """
        Save a validation record for finalized sessions.
        
        Args:
            record: Validation record data
        """
        pass
    
    @abstractmethod
    def load_validation_record(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load validation record for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Validation record or None if not found
        """
        pass


# =============================================================================
# In-Memory Implementation
# =============================================================================

class InMemoryReviewStorage(ReviewStorage):
    """
    In-memory storage implementation for development and testing.
    
    Data is lost when the application stops. Useful for:
    - Unit testing
    - Development environments
    - Prototyping
    """
    
    def __init__(self):
        """Initialize empty storage."""
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._validation_records: Dict[str, Dict[str, Any]] = {}
        logger.info("InMemoryReviewStorage initialized")
    
    def save_session(self, session_data: Dict[str, Any]) -> None:
        """Save session to memory."""
        session_id = session_data.get("session_id")
        if not session_id:
            raise ValueError("Session data must include 'session_id'")
        
        self._sessions[session_id] = session_data.copy()
        logger.debug(f"Saved session {session_id} to memory")
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session from memory."""
        session = self._sessions.get(session_id)
        return session.copy() if session else None
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session from memory."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug(f"Deleted session {session_id} from memory")
            return True
        return False
    
    def list_sessions(
        self,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List sessions from memory."""
        sessions = list(self._sessions.values())
        
        if status:
            sessions = [s for s in sessions if s.get("status") == status]
        
        # Sort by created_at descending
        sessions.sort(
            key=lambda s: s.get("created_at", ""),
            reverse=True,
        )
        
        return [s.copy() for s in sessions[:limit]]
    
    def update_item(
        self,
        session_id: str,
        review_id: str,
        item_data: Dict[str, Any],
    ) -> bool:
        """Update item in memory."""
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        items = session.get("items", [])
        for i, item in enumerate(items):
            if item.get("review_id") == review_id:
                items[i] = item_data
                logger.debug(f"Updated item {review_id} in session {session_id}")
                return True
        
        return False
    
    def save_validation_record(self, record: Dict[str, Any]) -> None:
        """Save validation record to memory."""
        session_id = record.get("session_id")
        if not session_id:
            raise ValueError("Validation record must include 'session_id'")
        
        self._validation_records[session_id] = record.copy()
        logger.debug(f"Saved validation record for session {session_id}")
    
    def load_validation_record(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load validation record from memory."""
        record = self._validation_records.get(session_id)
        return record.copy() if record else None
    
    def clear(self) -> None:
        """Clear all data (useful for testing)."""
        self._sessions.clear()
        self._validation_records.clear()
        logger.debug("Cleared all in-memory storage")


# =============================================================================
# File-Based Implementation
# =============================================================================

class FileReviewStorage(ReviewStorage):
    """
    File-based storage implementation with JSON persistence.
    
    Stores each session as a separate JSON file. Suitable for:
    - Single-server deployments
    - Development with persistence needs
    - Simple production setups
    """
    
    def __init__(self, storage_dir: str = "review_sessions"):
        """
        Initialize file storage.
        
        Args:
            storage_dir: Directory for storing session files
        """
        self.storage_dir = Path(storage_dir)
        self.sessions_dir = self.storage_dir / "sessions"
        self.records_dir = self.storage_dir / "validation_records"
        
        # Create directories if they don't exist
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.records_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"FileReviewStorage initialized at {self.storage_dir}")
    
    def _session_path(self, session_id: str) -> Path:
        """Get file path for a session."""
        return self.sessions_dir / f"{session_id}.json"
    
    def _record_path(self, session_id: str) -> Path:
        """Get file path for a validation record."""
        return self.records_dir / f"{session_id}.json"
    
    def save_session(self, session_data: Dict[str, Any]) -> None:
        """Save session to file."""
        session_id = session_data.get("session_id")
        if not session_id:
            raise ValueError("Session data must include 'session_id'")
        
        file_path = self._session_path(session_id)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, default=str)
        
        logger.debug(f"Saved session {session_id} to {file_path}")
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session from file."""
        file_path = self._session_path(session_id)
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading session {session_id}: {e}")
            return None
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session file."""
        file_path = self._session_path(session_id)
        
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Deleted session file {file_path}")
            return True
        return False
    
    def list_sessions(
        self,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List sessions from files."""
        sessions = []
        
        for file_path in self.sessions_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    session = json.load(f)
                    
                    if status is None or session.get("status") == status:
                        sessions.append(session)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error reading {file_path}: {e}")
                continue
        
        # Sort by created_at descending
        sessions.sort(
            key=lambda s: s.get("created_at", ""),
            reverse=True,
        )
        
        return sessions[:limit]
    
    def update_item(
        self,
        session_id: str,
        review_id: str,
        item_data: Dict[str, Any],
    ) -> bool:
        """Update item in session file."""
        session = self.load_session(session_id)
        if not session:
            return False
        
        items = session.get("items", [])
        updated = False
        
        for i, item in enumerate(items):
            if item.get("review_id") == review_id:
                items[i] = item_data
                updated = True
                break
        
        if updated:
            session["items"] = items
            self.save_session(session)
            logger.debug(f"Updated item {review_id} in session {session_id}")
        
        return updated
    
    def save_validation_record(self, record: Dict[str, Any]) -> None:
        """Save validation record to file."""
        session_id = record.get("session_id")
        if not session_id:
            raise ValueError("Validation record must include 'session_id'")
        
        file_path = self._record_path(session_id)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str)
        
        logger.debug(f"Saved validation record for {session_id}")
    
    def load_validation_record(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load validation record from file."""
        file_path = self._record_path(session_id)
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading validation record {session_id}: {e}")
            return None


# =============================================================================
# Factory Function
# =============================================================================

def create_review_storage(
    storage_type: str = "memory",
    storage_dir: Optional[str] = None,
) -> ReviewStorage:
    """
    Factory function to create appropriate storage instance.
    
    Args:
        storage_type: "memory" or "file"
        storage_dir: Directory for file storage (optional)
        
    Returns:
        ReviewStorage implementation
        
    Raises:
        ValueError: If invalid storage_type
    """
    if storage_type == "memory":
        return InMemoryReviewStorage()
    elif storage_type == "file":
        return FileReviewStorage(storage_dir or "review_sessions")
    else:
        raise ValueError(f"Unknown storage type: {storage_type}. Use 'memory' or 'file'.")

