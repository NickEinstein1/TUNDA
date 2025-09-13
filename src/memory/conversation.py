"""Conversation memory and context management."""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import threading

from ..utils.config import config

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Single turn in a conversation."""
    timestamp: str
    user_text: str
    user_emotion: str
    user_confidence: float
    assistant_response: str
    empathy_style: str
    response_confidence: float


@dataclass
class EmotionHistory:
    """History of detected emotions."""
    emotion: str
    confidence: float
    timestamp: str
    context: str


@dataclass
class ConversationSession:
    """Complete conversation session."""
    session_id: str
    start_time: str
    end_time: Optional[str]
    turns: List[ConversationTurn]
    emotion_history: List[EmotionHistory]
    user_preferences: Dict[str, Any]
    session_summary: str
    user_name: Optional[str] = None


class ConversationMemory:
    """Manages conversation history and context."""
    
    def __init__(self):
        self.config = config.memory
        self.current_session: Optional[ConversationSession] = None
        self.conversation_file = Path(self.config.conversation_file)
        self.lock = threading.Lock()
        
        # Ensure data directory exists
        self.conversation_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing conversations
        self.conversations: List[ConversationSession] = self._load_conversations()
    
    def start_new_session(self, session_id: Optional[str] = None) -> str:
        """Start a new conversation session."""
        with self.lock:
            if session_id is None:
                session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.current_session = ConversationSession(
                session_id=session_id,
                start_time=datetime.now().isoformat(),
                end_time=None,
                turns=[],
                emotion_history=[],
                user_preferences={},
                session_summary="",
                user_name=None
            )
            
            logger.info(f"Started new conversation session: {session_id}")
            return session_id
    
    def end_current_session(self):
        """End the current conversation session."""
        with self.lock:
            if self.current_session:
                self.current_session.end_time = datetime.now().isoformat()
                self.current_session.session_summary = self._generate_session_summary()
                
                # Save session
                self.conversations.append(self.current_session)
                self._save_conversations()
                
                logger.info(f"Ended conversation session: {self.current_session.session_id}")
                self.current_session = None
    
    def add_conversation_turn(self, 
                            user_text: str,
                            user_emotion: str,
                            user_confidence: float,
                            assistant_response: str,
                            empathy_style: str,
                            response_confidence: float):
        """Add a conversation turn to current session."""
        if not self.current_session:
            self.start_new_session()
        
        with self.lock:
            turn = ConversationTurn(
                timestamp=datetime.now().isoformat(),
                user_text=user_text,
                user_emotion=user_emotion,
                user_confidence=user_confidence,
                assistant_response=assistant_response,
                empathy_style=empathy_style,
                response_confidence=response_confidence
            )
            
            self.current_session.turns.append(turn)
            
            # Add to emotion history
            emotion_entry = EmotionHistory(
                emotion=user_emotion,
                confidence=user_confidence,
                timestamp=turn.timestamp,
                context=user_text[:100]  # First 100 chars as context
            )
            self.current_session.emotion_history.append(emotion_entry)
            
            # Limit history length
            if len(self.current_session.turns) > self.config.max_history_length:
                self.current_session.turns = self.current_session.turns[-self.config.max_history_length:]
            
            if len(self.current_session.emotion_history) > self.config.emotion_history_window:
                self.current_session.emotion_history = self.current_session.emotion_history[-self.config.emotion_history_window:]
            
            # Try to extract user name if not already known
            if not self.current_session.user_name:
                extracted_name = self._extract_user_name(user_text)
                if extracted_name:
                    self.current_session.user_name = extracted_name
                    logger.info(f"Learned user name: {extracted_name}")

            # Also try to extract name even if we have one (in case of correction)
            else:
                extracted_name = self._extract_user_name(user_text)
                if extracted_name and extracted_name != self.current_session.user_name:
                    logger.info(f"User name updated from {self.current_session.user_name} to {extracted_name}")
                    self.current_session.user_name = extracted_name

            # Auto-save periodically
            if len(self.current_session.turns) % 5 == 0:
                self._save_conversations()
    
    def get_conversation_context(self, window_size: Optional[int] = None) -> List[Dict[str, str]]:
        """Get recent conversation context."""
        if not self.current_session:
            return []
        
        window_size = window_size or self.config.context_window
        recent_turns = self.current_session.turns[-window_size:]
        
        context = []
        for turn in recent_turns:
            context.append({
                'user': turn.user_text,
                'assistant': turn.assistant_response,
                'emotion': turn.user_emotion,
                'timestamp': turn.timestamp
            })
        
        return context
    
    def get_emotion_patterns(self) -> Dict[str, Any]:
        """Analyze emotion patterns from history."""
        if not self.current_session or not self.current_session.emotion_history:
            return {}
        
        emotions = [entry.emotion for entry in self.current_session.emotion_history]
        confidences = [entry.confidence for entry in self.current_session.emotion_history]
        
        # Calculate emotion frequencies
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Most common emotion
        most_common_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else 'neutral'
        
        # Average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Recent emotion trend
        recent_emotions = emotions[-5:] if len(emotions) >= 5 else emotions
        recent_trend = self._analyze_emotion_trend(recent_emotions)
        
        return {
            'most_common_emotion': most_common_emotion,
            'emotion_distribution': emotion_counts,
            'average_confidence': avg_confidence,
            'recent_trend': recent_trend,
            'total_interactions': len(emotions)
        }
    
    def _analyze_emotion_trend(self, emotions: List[str]) -> str:
        """Analyze trend in recent emotions."""
        if len(emotions) < 2:
            return 'stable'
        
        # Simple trend analysis
        positive_emotions = ['happy', 'calm']
        negative_emotions = ['sad', 'angry', 'anxious']
        
        positive_count = sum(1 for e in emotions if e in positive_emotions)
        negative_count = sum(1 for e in emotions if e in negative_emotions)
        
        if positive_count > negative_count:
            return 'improving'
        elif negative_count > positive_count:
            return 'declining'
        else:
            return 'stable'
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """Get user preferences from current session."""
        if not self.current_session:
            return {}
        
        return self.current_session.user_preferences.copy()
    
    def update_user_preferences(self, preferences: Dict[str, Any]):
        """Update user preferences."""
        if not self.current_session:
            self.start_new_session()
        
        with self.lock:
            self.current_session.user_preferences.update(preferences)
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get statistics for current session."""
        if not self.current_session:
            return {}
        
        turns = self.current_session.turns
        emotions = self.current_session.emotion_history
        
        if not turns:
            return {}
        
        # Calculate statistics
        session_duration = None
        if self.current_session.end_time:
            start = datetime.fromisoformat(self.current_session.start_time)
            end = datetime.fromisoformat(self.current_session.end_time)
            session_duration = (end - start).total_seconds() / 60  # minutes
        
        avg_response_confidence = sum(turn.response_confidence for turn in turns) / len(turns)
        avg_emotion_confidence = sum(turn.user_confidence for turn in turns) / len(turns)
        
        empathy_styles_used = {}
        for turn in turns:
            style = turn.empathy_style
            empathy_styles_used[style] = empathy_styles_used.get(style, 0) + 1
        
        return {
            'session_id': self.current_session.session_id,
            'total_turns': len(turns),
            'session_duration_minutes': session_duration,
            'average_response_confidence': avg_response_confidence,
            'average_emotion_confidence': avg_emotion_confidence,
            'empathy_styles_used': empathy_styles_used,
            'emotion_patterns': self.get_emotion_patterns()
        }
    
    def _generate_session_summary(self) -> str:
        """Generate a summary of the conversation session."""
        if not self.current_session or not self.current_session.turns:
            return "No conversation data available."
        
        stats = self.get_session_statistics()
        emotion_patterns = stats.get('emotion_patterns', {})
        
        summary_parts = []
        summary_parts.append(f"Session with {stats['total_turns']} interactions.")
        
        if emotion_patterns.get('most_common_emotion'):
            summary_parts.append(f"Primary emotion: {emotion_patterns['most_common_emotion']}.")
        
        if emotion_patterns.get('recent_trend'):
            summary_parts.append(f"Emotional trend: {emotion_patterns['recent_trend']}.")
        
        return " ".join(summary_parts)
    
    def _load_conversations(self) -> List[ConversationSession]:
        """Load conversations from file."""
        if not self.conversation_file.exists():
            return []
        
        try:
            with open(self.conversation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            conversations = []
            for session_data in data:
                # Convert dictionaries back to dataclasses
                turns = [ConversationTurn(**turn) for turn in session_data.get('turns', [])]
                emotions = [EmotionHistory(**emotion) for emotion in session_data.get('emotion_history', [])]
                
                session = ConversationSession(
                    session_id=session_data['session_id'],
                    start_time=session_data['start_time'],
                    end_time=session_data.get('end_time'),
                    turns=turns,
                    emotion_history=emotions,
                    user_preferences=session_data.get('user_preferences', {}),
                    session_summary=session_data.get('session_summary', '')
                )
                conversations.append(session)
            
            logger.info(f"Loaded {len(conversations)} conversation sessions")
            return conversations
            
        except Exception as e:
            logger.error(f"Failed to load conversations: {e}")
            return []
    
    def _save_conversations(self):
        """Save conversations to file."""
        if not self.config.save_conversations:
            return
        
        try:
            # Convert to serializable format
            data = []
            all_sessions = self.conversations.copy()
            if self.current_session:
                all_sessions.append(self.current_session)
            
            for session in all_sessions:
                session_dict = asdict(session)
                data.append(session_dict)
            
            # Save to file
            with open(self.conversation_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved {len(data)} conversation sessions")
            
        except Exception as e:
            logger.error(f"Failed to save conversations: {e}")
    
    def get_all_sessions(self) -> List[ConversationSession]:
        """Get all conversation sessions."""
        return self.conversations.copy()
    
    def get_session_by_id(self, session_id: str) -> Optional[ConversationSession]:
        """Get specific session by ID."""
        for session in self.conversations:
            if session.session_id == session_id:
                return session
        
        if self.current_session and self.current_session.session_id == session_id:
            return self.current_session
        
        return None
    
    def clear_all_conversations(self):
        """Clear all conversation history."""
        with self.lock:
            self.conversations.clear()
            self.current_session = None

            if self.conversation_file.exists():
                self.conversation_file.unlink()

            logger.info("Cleared all conversation history")

    def _extract_user_name(self, text: str) -> Optional[str]:
        """Extract user name from text using simple patterns."""
        import re

        text_lower = text.lower()

        # Common patterns for name introduction
        patterns = [
            r"my name is (\w+)",
            r"i'm (\w+)",
            r"i am (\w+)",
            r"call me (\w+)",
            r"this is (\w+)",
            r"it's (\w+)",
            r"(\w+) here",
            r"hi,? i'm (\w+)",
            r"hello,? i'm (\w+)",
            r"name's (\w+)",
            r"i go by (\w+)"
        ]

        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                name = match.group(1).capitalize()
                # Filter out common words that aren't names
                excluded_words = [
                    'good', 'fine', 'okay', 'well', 'here', 'there', 'doing', 'feeling',
                    'great', 'bad', 'sad', 'happy', 'tired', 'busy', 'free', 'ready',
                    'sorry', 'thanks', 'hello', 'hi', 'hey', 'yes', 'no', 'sure'
                ]
                if name.lower() not in excluded_words and len(name) > 1:
                    logger.info(f"Extracted name: {name} from text: {text}")
                    return name

        return None

    def get_user_name(self) -> Optional[str]:
        """Get the current user's name if known."""
        if self.current_session:
            return self.current_session.user_name
        return None
