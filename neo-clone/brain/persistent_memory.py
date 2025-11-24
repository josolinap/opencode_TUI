"""
Persistent Memory System for MiniMax Agent Architecture

This module provides persistent storage and retrieval of conversations,
memories, and learning data that survives across sessions.

Author: MiniMax Agent
Version: 1.0
"""

import json
import sqlite3
import threading
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict
import uuid
import logging

from data_models import (
    MemoryEntry, ConversationHistory, Message, MemoryType, 
    MessageRole, create_conversation
)

logger = logging.getLogger(__name__)


class PersistentMemory:
    """
    Persistent memory system using SQLite for durable storage
    """
    
    def __init__(self, db_path: str = "brain_memory.db", max_conversations: int = 10000):
        """
        Initialize persistent memory
        
        Args:
            db_path: Path to SQLite database file
            max_conversations: Maximum number of conversations to store
        """
        self.db_path = db_path
        self.max_conversations = max_conversations
        self.session_id = str(uuid.uuid4())
        self.lock = threading.RLock()
        
        # Initialize database
        self._init_database()
        
        # In-memory cache for recent data
        self.recent_conversations: Dict[str, ConversationHistory] = {}
        self.memory_cache: Dict[str, MemoryEntry] = {}
        self.conversation_count = 0
        
        logger.info(f"Persistent Memory initialized with database: {db_path}")
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Conversations table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id TEXT PRIMARY KEY,
                        session_id TEXT,
                        user_input TEXT,
                        assistant_response TEXT,
                        intent TEXT,
                        skill_used TEXT,
                        confidence REAL,
                        success INTEGER,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT
                    )
                """)
                
                # Memory entries table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS memory_entries (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        memory_type TEXT NOT NULL,
                        importance REAL DEFAULT 0.5,
                        tags TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT
                    )
                """)
                
                # Learning patterns table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS learning_patterns (
                        id TEXT PRIMARY KEY,
                        pattern_key TEXT NOT NULL,
                        pattern_data TEXT NOT NULL,
                        usage_count INTEGER DEFAULT 0,
                        last_used DATETIME DEFAULT CURRENT_TIMESTAMP,
                        success_rate REAL DEFAULT 0.0
                    )
                """)
                
                # Performance metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id TEXT PRIMARY KEY,
                        operation_name TEXT NOT NULL,
                        execution_time REAL NOT NULL,
                        success INTEGER NOT NULL,
                        input_size INTEGER DEFAULT 0,
                        output_size INTEGER DEFAULT 0,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_entries_type ON memory_entries(memory_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_entries_timestamp ON memory_entries(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_learning_patterns_key ON learning_patterns(pattern_key)")
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def add_conversation(
        self,
        user_input: str,
        assistant_response: str,
        intent: str = None,
        skill_used: str = None,
        metadata: Dict[str, Any] = None,
        confidence: float = 0.0,
        success: bool = True
    ) -> str:
        """
        Add a conversation to persistent storage
        
        Args:
            user_input: User's input text
            assistant_response: Assistant's response
            intent: Detected intent
            skill_used: Skill that was used
            metadata: Additional metadata
            confidence: Confidence score
            success: Whether the interaction was successful
        
        Returns:
            Conversation ID
        """
        conversation_id = str(uuid.uuid4())
        
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        INSERT INTO conversations 
                        (id, session_id, user_input, assistant_response, intent, skill_used, 
                         confidence, success, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        conversation_id,
                        self.session_id,
                        user_input,
                        assistant_response,
                        intent,
                        skill_used,
                        confidence,
                        1 if success else 0,
                        json.dumps(metadata or {})
                    ))
                    
                    conn.commit()
                    
                    # Update in-memory cache
                    if len(self.recent_conversations) >= 100:
                        # Keep only the most recent 100 conversations in memory
                        oldest_key = min(self.recent_conversations.keys())
                        del self.recent_conversations[oldest_key]
                    
                    # Create conversation history entry for cache
                    conversation = create_conversation(self.session_id)
                    conversation.messages.append(
                        Message(content=user_input, role=MessageRole.USER)
                    )
                    conversation.messages.append(
                        Message(content=assistant_response, role=MessageRole.ASSISTANT)
                    )
                    self.recent_conversations[conversation_id] = conversation
                    
                    self.conversation_count += 1
                    
                    logger.debug(f"Added conversation: {conversation_id}")
                    return conversation_id
                    
            except Exception as e:
                logger.error(f"Failed to add conversation: {e}")
                return ""
    
    def get_conversation_history(
        self, 
        session_id: str = None, 
        limit: int = 50,
        since: datetime = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history
        
        Args:
            session_id: Session ID to filter by
            limit: Maximum number of conversations to return
            since: Return conversations since this datetime
        
        Returns:
            List of conversation records
        """
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    query = """
                        SELECT id, session_id, user_input, assistant_response, intent, 
                               skill_used, confidence, success, timestamp, metadata
                        FROM conversations
                        WHERE 1=1
                    """
                    params = []
                    
                    if session_id:
                        query += " AND session_id = ?"
                        params.append(session_id)
                    
                    if since:
                        query += " AND timestamp >= ?"
                        params.append(since.isoformat())
                    
                    query += " ORDER BY timestamp DESC LIMIT ?"
                    params.append(limit)
                    
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    conversations = []
                    for row in rows:
                        conversations.append({
                            "id": row[0],
                            "session_id": row[1],
                            "user_input": row[2],
                            "assistant_response": row[3],
                            "intent": row[4],
                            "skill_used": row[5],
                            "confidence": row[6],
                            "success": bool(row[7]),
                            "timestamp": row[8],
                            "metadata": json.loads(row[9]) if row[9] else {}
                        })
                    
                    return conversations
                    
            except Exception as e:
                logger.error(f"Failed to retrieve conversation history: {e}")
                return []
    
    def add_memory(self, memory: MemoryEntry) -> str:
        """
        Add a memory entry to persistent storage
        
        Args:
            memory: MemoryEntry to store
        
        Returns:
            Memory ID
        """
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        INSERT INTO memory_entries 
                        (id, content, memory_type, importance, tags, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        memory.id,
                        memory.content,
                        memory.memory_type.value,
                        memory.importance,
                        json.dumps(memory.tags),
                        json.dumps(memory.metadata)
                    ))
                    
                    conn.commit()
                    
                    # Update cache
                    self.memory_cache[memory.id] = memory
                    
                    logger.debug(f"Added memory: {memory.id}")
                    return memory.id
                    
            except Exception as e:
                logger.error(f"Failed to add memory: {e}")
                return ""
    
    def get_memories(
        self,
        memory_type: MemoryType = None,
        min_importance: float = 0.0,
        limit: int = 100,
        tags: List[str] = None
    ) -> List[MemoryEntry]:
        """
        Retrieve memory entries
        
        Args:
            memory_type: Filter by memory type
            min_importance: Minimum importance score
            limit: Maximum number of memories to return
            tags: Filter by tags
        
        Returns:
            List of MemoryEntry objects
        """
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    query = """
                        SELECT id, content, memory_type, importance, tags, timestamp, metadata
                        FROM memory_entries
                        WHERE importance >= ?
                    """
                    params = [min_importance]
                    
                    if memory_type:
                        query += " AND memory_type = ?"
                        params.append(memory_type.value)
                    
                    query += " ORDER BY importance DESC, timestamp DESC LIMIT ?"
                    params.append(limit)
                    
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    memories = []
                    for row in rows:
                        memory_tags = json.loads(row[4]) if row[4] else []
                        
                        # Filter by tags if specified
                        if tags and not any(tag in memory_tags for tag in tags):
                            continue
                        
                        memory = MemoryEntry(
                            id=row[0],
                            content=row[1],
                            memory_type=MemoryType(row[2]),
                            importance=row[3],
                            tags=memory_tags,
                            timestamp=datetime.fromisoformat(row[5]) if row[5] else datetime.now(),
                            metadata=json.loads(row[6]) if row[6] else {}
                        )
                        memories.append(memory)
                    
                    return memories
                    
            except Exception as e:
                logger.error(f"Failed to retrieve memories: {e}")
                return []
    
    def update_learning_pattern(self, pattern_key: str, pattern_data: Dict[str, Any]):
        """
        Update or create a learning pattern
        
        Args:
            pattern_key: Unique pattern identifier
            pattern_data: Pattern data to store
        """
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Check if pattern exists
                    cursor.execute("SELECT id, usage_count, success_rate FROM learning_patterns WHERE pattern_key = ?", (pattern_key,))
                    existing = cursor.fetchone()
                    
                    if existing:
                        # Update existing pattern
                        usage_count = existing[1] + 1
                        success_rate = pattern_data.get("success_rate", 0.0)
                        
                        cursor.execute("""
                            UPDATE learning_patterns 
                            SET pattern_data = ?, usage_count = ?, last_used = CURRENT_TIMESTAMP, success_rate = ?
                            WHERE pattern_key = ?
                        """, (json.dumps(pattern_data), usage_count, success_rate, pattern_key))
                    else:
                        # Create new pattern
                        cursor.execute("""
                            INSERT INTO learning_patterns (id, pattern_key, pattern_data, usage_count, success_rate)
                            VALUES (?, ?, ?, ?, ?)
                        """, (str(uuid.uuid4()), pattern_key, json.dumps(pattern_data), 1, pattern_data.get("success_rate", 0.0)))
                    
                    conn.commit()
                    logger.debug(f"Updated learning pattern: {pattern_key}")
                    
            except Exception as e:
                logger.error(f"Failed to update learning pattern: {e}")
    
    def get_learning_patterns(self, pattern_key: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve learning patterns
        
        Args:
            pattern_key: Specific pattern to retrieve (optional)
            limit: Maximum number of patterns to return
        
        Returns:
            List of learning patterns
        """
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    if pattern_key:
                        cursor.execute("""
                            SELECT pattern_key, pattern_data, usage_count, success_rate, last_used
                            FROM learning_patterns WHERE pattern_key = ?
                        """, (pattern_key,))
                    else:
                        cursor.execute("""
                            SELECT pattern_key, pattern_data, usage_count, success_rate, last_used
                            FROM learning_patterns ORDER BY usage_count DESC LIMIT ?
                        """, (limit,))
                    
                    rows = cursor.fetchall()
                    patterns = []
                    
                    for row in rows:
                        patterns.append({
                            "pattern_key": row[0],
                            "pattern_data": json.loads(row[1]),
                            "usage_count": row[2],
                            "success_rate": row[3],
                            "last_used": row[4]
                        })
                    
                    return patterns
                    
            except Exception as e:
                logger.error(f"Failed to retrieve learning patterns: {e}")
                return []
    
    def store_performance_metrics(self, metrics_data: Dict[str, Any]) -> str:
        """
        Store performance metrics
        
        Args:
            metrics_data: Metrics to store
        
        Returns:
            Metrics ID
        """
        metrics_id = str(uuid.uuid4())
        
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        INSERT INTO performance_metrics 
                        (id, operation_name, execution_time, success, input_size, output_size, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        metrics_id,
                        metrics_data.get("operation_name", ""),
                        metrics_data.get("execution_time", 0.0),
                        1 if metrics_data.get("success", True) else 0,
                        metrics_data.get("input_size", 0),
                        metrics_data.get("output_size", 0),
                        json.dumps(metrics_data.get("metadata", {}))
                    ))
                    
                    conn.commit()
                    return metrics_id
                    
            except Exception as e:
                logger.error(f"Failed to store performance metrics: {e}")
                return ""
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """
        Clean up old data to prevent database bloat
        
        Args:
            days_to_keep: Number of days of data to keep
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Clean up old conversations
                    cursor.execute("DELETE FROM conversations WHERE timestamp < ?", (cutoff_date.isoformat(),))
                    conversations_deleted = cursor.rowcount
                    
                    # Clean up old memory entries with low importance
                    cursor.execute("""
                        DELETE FROM memory_entries 
                        WHERE timestamp < ? AND importance < 0.3
                    """, (cutoff_date.isoformat(),))
                    memories_deleted = cursor.rowcount
                    
                    # Clean up old performance metrics
                    cursor.execute("DELETE FROM performance_metrics WHERE timestamp < ?", (cutoff_date.isoformat(),))
                    metrics_deleted = cursor.rowcount
                    
                    conn.commit()
                    
                    logger.info(f"Cleanup completed: {conversations_deleted} conversations, "
                              f"{memories_deleted} memories, {metrics_deleted} metrics deleted")
                    
                    return {
                        "conversations_deleted": conversations_deleted,
                        "memories_deleted": memories_deleted,
                        "metrics_deleted": metrics_deleted
                    }
                    
            except Exception as e:
                logger.error(f"Cleanup failed: {e}")
                return {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory system statistics
        
        Returns:
            Dictionary with memory statistics
        """
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Get counts
                    cursor.execute("SELECT COUNT(*) FROM conversations")
                    total_conversations = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(*) FROM memory_entries")
                    total_memories = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(*) FROM learning_patterns")
                    total_patterns = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(*) FROM performance_metrics")
                    total_metrics = cursor.fetchone()[0]
                    
                    # Get recent activity (last 24 hours)
                    yesterday = datetime.now() - timedelta(days=1)
                    cursor.execute("""
                        SELECT COUNT(*) FROM conversations WHERE timestamp >= ?
                    """, (yesterday.isoformat(),))
                    recent_conversations = cursor.fetchone()[0]
                    
                    # Get success rate
                    cursor.execute("""
                        SELECT AVG(success) FROM conversations WHERE timestamp >= ?
                    """, (yesterday.isoformat(),))
                    success_rate = cursor.fetchone()[0] or 0.0
                    
                    return {
                        "total_conversations": total_conversations,
                        "total_memories": total_memories,
                        "total_learning_patterns": total_patterns,
                        "total_performance_metrics": total_metrics,
                        "recent_conversations_24h": recent_conversations,
                        "recent_success_rate": success_rate,
                        "cache_size": len(self.recent_conversations) + len(self.memory_cache),
                        "database_path": self.db_path,
                        "session_id": self.session_id
                    }
                    
            except Exception as e:
                logger.error(f"Failed to get statistics: {e}")
                return {}
    
    def shutdown(self):
        """Shutdown persistent memory system"""
        logger.info("Shutting down Persistent Memory")
        # Connection will be closed automatically when SQLite connection goes out of scope
        # Additional cleanup can be added here if needed


# Global persistent memory instance
_memory_instance: Optional[PersistentMemory] = None
_memory_lock = threading.Lock()


def get_memory(db_path: str = "brain_memory.db", max_conversations: int = 10000) -> PersistentMemory:
    """
    Get singleton persistent memory instance
    
    Args:
        db_path: Path to database file
        max_conversations: Maximum conversations to store
    
    Returns:
        PersistentMemory singleton instance
    """
    global _memory_instance
    
    if _memory_instance is None:
        with _memory_lock:
            if _memory_instance is None:
                _memory_instance = PersistentMemory(db_path, max_conversations)
    
    return _memory_instance


def reset_memory() -> None:
    """Reset the persistent memory instance"""
    global _memory_instance
    with _memory_lock:
        if _memory_instance:
            try:
                _memory_instance.shutdown()
            except Exception:
                pass
        _memory_instance = None
    logger.info("Persistent Memory instance reset")


def create_memory_instance(db_path: str = "brain_memory.db", max_conversations: int = 10000) -> PersistentMemory:
    """
    Create a new persistent memory instance
    
    Args:
        db_path: Path to database file
        max_conversations: Maximum conversations to store
    
    Returns:
        New PersistentMemory instance
    """
    return PersistentMemory(db_path, max_conversations)
