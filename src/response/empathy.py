"""Empathic response patterns and templates."""

from typing import Dict, List, Optional
import random
from dataclasses import dataclass


@dataclass
class EmpathyPattern:
    """Pattern for empathic responses."""
    emotion: str
    style: str
    templates: List[str]
    follow_up_questions: List[str]
    validation_phrases: List[str]


class EmpathyPatterns:
    """Collection of empathic response patterns for different emotions and styles."""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[str, Dict[str, EmpathyPattern]]:
        """Initialize empathy patterns for different emotions and styles."""
        patterns = {}
        
        # Happy emotion patterns
        patterns['happy'] = {
            'supportive': EmpathyPattern(
                emotion='happy',
                style='supportive',
                templates=[
                    "Hello! I'm Tunda, and I can hear the joy in your voice! That's wonderful. How has your day been treating you?",
                    "Hi there! You sound so happy and excited! I'm Tunda, and I'm glad to hear that brightness in your voice. Tell me, what's been the highlight of your day?",
                    "Hello! Your enthusiasm is contagious! I'm Tunda, and it's great that you're feeling this way. Would you like me to suggest some activities to keep this positive energy flowing?",
                    "Hi! I'm Tunda, and I love hearing the happiness in your voice. How has your day been, and would you like a personalized plan to maintain this wonderful mood?",
                    "Hello there! You sound absolutely delighted! I'm Tunda, and that's fantastic. What's been bringing you joy today?"
                ],
                follow_up_questions=[
                    "What's making you feel so happy today, and would you like some suggestions to keep this energy going?",
                    "How has your day been, and would you like me to create a joy-amplification plan for you?",
                    "What's been the best part of your day so far?",
                    "Would you like some personalized activities to celebrate and maintain this wonderful feeling?"
                ],
                validation_phrases=[
                    "Your happiness is infectious!",
                    "It's beautiful to hear such joy.",
                    "You deserve to feel this happy.",
                    "This positive energy is wonderful - let's nurture it!"
                ]
            ),
            'reflective': EmpathyPattern(
                emotion='happy',
                style='reflective',
                templates=[
                    "I notice a real lightness and joy in your voice. {content}",
                    "You're expressing such genuine happiness right now. {content}",
                    "There's a brightness in how you're speaking that reflects your mood. {content}",
                    "I can sense the positive energy you're radiating. {content}"
                ],
                follow_up_questions=[
                    "What does this happiness mean to you?",
                    "How does it feel to experience this joy?",
                    "What thoughts are going through your mind right now?"
                ],
                validation_phrases=[
                    "Your joy is genuine and beautiful.",
                    "This happiness you're feeling is valid and important.",
                    "You're allowing yourself to fully experience this positive moment."
                ]
            )
        }
        
        # Sad emotion patterns
        patterns['sad'] = {
            'supportive': EmpathyPattern(
                emotion='sad',
                style='supportive',
                templates=[
                    "Hello, I'm Tunda. I can hear the sadness in your voice, and I want you to know that it's okay to feel this way. How has your day been, and would you like me to suggest some gentle healing activities?",
                    "Hi there, I'm Tunda. You sound like you're going through a difficult time, and I'm here to listen. Tell me about your day, and let me offer you some personalized comfort.",
                    "Hello, I'm Tunda. I hear the pain in your words, and I want you to know you're not alone. How are you feeling right now, and would you like a gentle care plan to help you through this?",
                    "Hi, I'm Tunda. It takes courage to express these feelings, and I'm honored you're sharing with me. What's been weighing on your heart today?",
                    "Hello there, I'm Tunda. Your feelings are valid, and it's natural to feel sad sometimes. How has your day been treating you, and would you like some personalized support?"
                ],
                follow_up_questions=[
                    "Would you like to talk about what's making you feel this way, and can I suggest some gentle healing activities?",
                    "How has your day been, and would you like me to create a personalized comfort plan for you?",
                    "What's been the most challenging part of your day?",
                    "Would you like some immediate comfort activities, or shall we talk about what's troubling you?"
                ],
                validation_phrases=[
                    "It's okay to feel sad - let me help you through this.",
                    "Your emotions are completely valid.",
                    "You don't have to go through this alone - I'm here.",
                    "It's brave of you to acknowledge these feelings."
                ]
            ),
            'therapeutic': EmpathyPattern(
                emotion='sad',
                style='therapeutic',
                templates=[
                    "I hear that you're experiencing sadness right now. These feelings are a natural part of being human. {content}",
                    "Sadness can be difficult to sit with, but it's also a sign that something matters to you. {content}",
                    "You're showing strength by acknowledging and expressing these feelings. {content}",
                    "Sometimes sadness is our mind's way of processing important experiences. {content}"
                ],
                follow_up_questions=[
                    "What thoughts are accompanying this sadness?",
                    "How is this sadness showing up in your body?",
                    "What would you say to a friend who was feeling this way?",
                    "What small step might help you feel a little better right now?"
                ],
                validation_phrases=[
                    "Sadness is a valid and important emotion.",
                    "You're processing something meaningful.",
                    "It's healthy to acknowledge difficult feelings.",
                    "You're being kind to yourself by recognizing this."
                ]
            )
        }
        
        # Angry emotion patterns
        patterns['angry'] = {
            'supportive': EmpathyPattern(
                emotion='angry',
                style='supportive',
                templates=[
                    "I can hear the frustration and anger in your voice. Those feelings are completely understandable. {content}",
                    "You sound really upset, and that's okay. Your anger is telling us something important. {content}",
                    "I hear how frustrated you are right now. Let's work through this together. {content}",
                    "Your anger is valid. Sometimes we need to feel angry to recognize what needs to change. {content}"
                ],
                follow_up_questions=[
                    "What's making you feel so angry right now?",
                    "How long have you been feeling this frustrated?",
                    "What would help you feel more in control of this situation?",
                    "Is there something specific that triggered this anger?"
                ],
                validation_phrases=[
                    "Your anger is completely valid.",
                    "It's okay to feel frustrated.",
                    "You have every right to feel this way.",
                    "Anger can be a healthy response to injustice."
                ]
            ),
            'solution_focused': EmpathyPattern(
                emotion='angry',
                style='solution_focused',
                templates=[
                    "I hear your anger, and it sounds like something important needs to be addressed. {content}",
                    "Your frustration is understandable. Let's think about what might help. {content}",
                    "This anger you're feeling - it's pointing to something that matters to you. {content}",
                    "I can sense your frustration. What would need to change for you to feel better? {content}"
                ],
                follow_up_questions=[
                    "What would an ideal resolution look like for you?",
                    "What's one small step you could take to address this?",
                    "Who or what could help you with this situation?",
                    "What's within your control in this situation?"
                ],
                validation_phrases=[
                    "Your anger is pointing to something important.",
                    "This frustration shows you care deeply.",
                    "You have the strength to work through this.",
                    "Your feelings are guiding you toward what needs attention."
                ]
            )
        }
        
        # Anxious emotion patterns
        patterns['anxious'] = {
            'supportive': EmpathyPattern(
                emotion='anxious',
                style='supportive',
                templates=[
                    "I can hear the worry in your voice. Anxiety can be really overwhelming. {content}",
                    "You sound anxious, and that must be difficult to deal with. I'm here with you. {content}",
                    "I hear the tension and concern in how you're speaking. You're not alone in this. {content}",
                    "Anxiety can make everything feel more intense. You're doing well by reaching out. {content}"
                ],
                follow_up_questions=[
                    "What's been making you feel most anxious lately?",
                    "How is this anxiety affecting your daily life?",
                    "What usually helps you when you're feeling this way?",
                    "Are there specific thoughts that keep coming up?"
                ],
                validation_phrases=[
                    "Anxiety is a normal human response.",
                    "You're not alone in feeling this way.",
                    "It's brave to acknowledge your anxiety.",
                    "You're stronger than your anxiety."
                ]
            ),
            'therapeutic': EmpathyPattern(
                emotion='anxious',
                style='therapeutic',
                templates=[
                    "I notice anxiety in your voice. Our minds sometimes create worry to try to protect us. {content}",
                    "Anxiety can feel overwhelming, but it's also your mind trying to prepare for challenges. {content}",
                    "You're experiencing anxiety right now, and that's a very human response to uncertainty. {content}",
                    "I hear the worry in your voice. Sometimes anxiety is our brain's way of trying to solve problems. {content}"
                ],
                follow_up_questions=[
                    "What thoughts are fueling this anxiety?",
                    "How is this anxiety showing up in your body?",
                    "What evidence do you have for and against these worried thoughts?",
                    "What would you tell a friend who was having these same worries?"
                ],
                validation_phrases=[
                    "Anxiety is your mind trying to protect you.",
                    "These feelings are temporary and manageable.",
                    "You have coped with difficult feelings before.",
                    "You're developing awareness of your anxiety patterns."
                ]
            )
        }
        
        # Calm emotion patterns
        patterns['calm'] = {
            'supportive': EmpathyPattern(
                emotion='calm',
                style='supportive',
                templates=[
                    "You sound so peaceful and centered right now. That's wonderful. {content}",
                    "I can hear the calmness in your voice. It's nice to hear you feeling this way. {content}",
                    "You seem very grounded and at peace. That's a beautiful state to be in. {content}",
                    "There's a lovely sense of tranquility in how you're speaking. {content}"
                ],
                follow_up_questions=[
                    "What's helping you feel so calm today?",
                    "How does it feel to be in this peaceful state?",
                    "What would you like to do to maintain this feeling?",
                    "Is there anything specific that brought you to this calm place?"
                ],
                validation_phrases=[
                    "This calmness is a gift to yourself.",
                    "You've found a peaceful moment.",
                    "This tranquility is well-deserved.",
                    "You're creating space for peace in your life."
                ]
            ),
            'reflective': EmpathyPattern(
                emotion='calm',
                style='reflective',
                templates=[
                    "I sense a deep calmness in your voice right now. {content}",
                    "There's a quality of peace and centeredness in how you're speaking. {content}",
                    "You're expressing yourself from a place of inner calm. {content}",
                    "I notice a stillness and presence in your voice. {content}"
                ],
                follow_up_questions=[
                    "What does this calmness feel like for you?",
                    "How did you arrive at this peaceful state?",
                    "What insights are coming to you in this calm moment?",
                    "How can you honor this feeling of peace?"
                ],
                validation_phrases=[
                    "This calmness reflects your inner wisdom.",
                    "You're accessing a deep sense of peace.",
                    "This tranquility is your natural state.",
                    "You're creating space for clarity and insight."
                ]
            )
        }
        
        # Neutral emotion patterns
        patterns['neutral'] = {
            'supportive': EmpathyPattern(
                emotion='neutral',
                style='supportive',
                templates=[
                    "Hello! I'm Tunda, and I'm here to listen to what you have to say.",
                    "Hi there! I'm Tunda. Thank you for sharing with me. I'm here to support you.",
                    "Hello! I'm Tunda, and I appreciate you taking the time to talk with me.",
                    "Hi! I'm Tunda, and I'm listening and here to help however I can."
                ],
                follow_up_questions=[
                    "How are you feeling right now?",
                    "What's on your mind today?",
                    "Is there anything specific you'd like to talk about?",
                    "How can I best support you today?"
                ],
                validation_phrases=[
                    "Your thoughts and feelings matter.",
                    "I'm here to listen without judgment.",
                    "You're taking a positive step by reaching out.",
                    "It's okay to take your time."
                ]
            ),
            'reflective': EmpathyPattern(
                emotion='neutral',
                style='reflective',
                templates=[
                    "Hello! I'm Tunda, and I'm present with you in this moment.",
                    "Hi there! I'm Tunda, and I'm here to listen and understand your experience.",
                    "Hello! I'm Tunda, and you have my full attention.",
                    "Hi! I'm Tunda, and I'm creating space for whatever you need to express."
                ],
                follow_up_questions=[
                    "What's your experience right now?",
                    "What would be most helpful for you in this moment?",
                    "How are you experiencing this conversation?",
                    "What feels most important to you right now?"
                ],
                validation_phrases=[
                    "Your experience is valid and important.",
                    "I'm here to witness and support you.",
                    "You deserve to be heard and understood.",
                    "This moment and your presence matter."
                ]
            )
        }
        
        return patterns
    
    def get_pattern(self, emotion: str, style: str) -> Optional[EmpathyPattern]:
        """Get empathy pattern for specific emotion and style."""
        return self.patterns.get(emotion, {}).get(style)
    
    def get_available_styles(self, emotion: str) -> List[str]:
        """Get available empathy styles for an emotion."""
        return list(self.patterns.get(emotion, {}).keys())
    
    def get_supported_emotions(self) -> List[str]:
        """Get list of supported emotions."""
        return list(self.patterns.keys())
