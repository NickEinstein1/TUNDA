"""Personalized care plans based on emotional states."""

from typing import Dict, List, Optional
from dataclasses import dataclass
import random


@dataclass
class CarePlanActivity:
    """Individual care plan activity."""
    name: str
    description: str
    duration: str
    difficulty: str  # easy, medium, hard
    category: str  # breathing, movement, mindfulness, social, creative


@dataclass
class PersonalizedCarePlan:
    """Complete personalized care plan."""
    emotion: str
    plan_name: str
    description: str
    immediate_activities: List[CarePlanActivity]
    daily_activities: List[CarePlanActivity]
    weekly_goals: List[str]
    affirmations: List[str]


class CarePlanGenerator:
    """Generates personalized care plans based on emotional states."""
    
    def __init__(self):
        self.care_plans = self._initialize_care_plans()
    
    def _initialize_care_plans(self) -> Dict[str, PersonalizedCarePlan]:
        """Initialize care plans for different emotions."""
        plans = {}
        
        # Happy emotion care plan
        plans['happy'] = PersonalizedCarePlan(
            emotion='happy',
            plan_name='Joy Amplification Plan',
            description='Let\'s build on this wonderful positive energy and make it last!',
            immediate_activities=[
                CarePlanActivity(
                    name='Gratitude Moment',
                    description='Take 2 minutes to think of 3 things you\'re grateful for right now',
                    duration='2 minutes',
                    difficulty='easy',
                    category='mindfulness'
                ),
                CarePlanActivity(
                    name='Share the Joy',
                    description='Send a positive message to someone you care about',
                    duration='5 minutes',
                    difficulty='easy',
                    category='social'
                ),
                CarePlanActivity(
                    name='Happy Movement',
                    description='Do a little dance or stretch to your favorite upbeat song',
                    duration='3-5 minutes',
                    difficulty='easy',
                    category='movement'
                )
            ],
            daily_activities=[
                CarePlanActivity(
                    name='Joy Journal',
                    description='Write down what made you happy today and why',
                    duration='10 minutes',
                    difficulty='easy',
                    category='creative'
                ),
                CarePlanActivity(
                    name='Kindness Act',
                    description='Do one small act of kindness for yourself or others',
                    duration='15 minutes',
                    difficulty='medium',
                    category='social'
                )
            ],
            weekly_goals=[
                'Maintain this positive momentum by doing one thing you love each day',
                'Share your happiness with others through acts of kindness',
                'Create a "happiness toolkit" of activities that bring you joy'
            ],
            affirmations=[
                'I deserve this happiness and joy',
                'My positive energy spreads to others around me',
                'I am grateful for this beautiful moment'
            ]
        )
        
        # Sad emotion care plan
        plans['sad'] = PersonalizedCarePlan(
            emotion='sad',
            plan_name='Gentle Healing Plan',
            description='It\'s okay to feel sad. Let\'s take gentle steps toward feeling better.',
            immediate_activities=[
                CarePlanActivity(
                    name='Comfort Breathing',
                    description='Take 5 slow, deep breaths. Breathe in comfort, breathe out sadness',
                    duration='3 minutes',
                    difficulty='easy',
                    category='breathing'
                ),
                CarePlanActivity(
                    name='Self-Compassion Touch',
                    description='Place your hand on your heart and say "This is a moment of suffering, and that\'s okay"',
                    duration='2 minutes',
                    difficulty='easy',
                    category='mindfulness'
                ),
                CarePlanActivity(
                    name='Gentle Movement',
                    description='Take a slow walk or do gentle stretches',
                    duration='10 minutes',
                    difficulty='easy',
                    category='movement'
                )
            ],
            daily_activities=[
                CarePlanActivity(
                    name='Emotion Expression',
                    description='Write, draw, or voice-record your feelings without judgment',
                    duration='15 minutes',
                    difficulty='medium',
                    category='creative'
                ),
                CarePlanActivity(
                    name='Comfort Connection',
                    description='Reach out to a trusted friend or family member',
                    duration='20 minutes',
                    difficulty='medium',
                    category='social'
                )
            ],
            weekly_goals=[
                'Allow yourself to feel sad without judgment',
                'Practice one self-care activity daily',
                'Connect with at least one supportive person this week'
            ],
            affirmations=[
                'My feelings are valid and temporary',
                'I am worthy of love and comfort',
                'This sadness will pass, and I will feel joy again'
            ]
        )
        
        # Anxious emotion care plan
        plans['anxious'] = PersonalizedCarePlan(
            emotion='anxious',
            plan_name='Calm & Ground Plan',
            description='Let\'s work together to bring you back to a place of calm and safety.',
            immediate_activities=[
                CarePlanActivity(
                    name='5-4-3-2-1 Grounding',
                    description='Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste',
                    duration='5 minutes',
                    difficulty='easy',
                    category='mindfulness'
                ),
                CarePlanActivity(
                    name='Box Breathing',
                    description='Breathe in for 4, hold for 4, out for 4, hold for 4. Repeat 4 times',
                    duration='3 minutes',
                    difficulty='easy',
                    category='breathing'
                ),
                CarePlanActivity(
                    name='Progressive Muscle Relaxation',
                    description='Tense and release each muscle group from toes to head',
                    duration='10 minutes',
                    difficulty='medium',
                    category='movement'
                )
            ],
            daily_activities=[
                CarePlanActivity(
                    name='Worry Time',
                    description='Set aside 15 minutes to write down worries, then put them away',
                    duration='15 minutes',
                    difficulty='medium',
                    category='mindfulness'
                ),
                CarePlanActivity(
                    name='Calming Routine',
                    description='Create a soothing bedtime routine with tea, music, or reading',
                    duration='30 minutes',
                    difficulty='easy',
                    category='mindfulness'
                )
            ],
            weekly_goals=[
                'Practice grounding techniques daily',
                'Identify and challenge anxious thoughts',
                'Build a toolkit of calming activities'
            ],
            affirmations=[
                'I am safe in this moment',
                'I can handle whatever comes my way',
                'My anxiety is temporary and manageable'
            ]
        )
        
        # Angry emotion care plan
        plans['angry'] = PersonalizedCarePlan(
            emotion='angry',
            plan_name='Healthy Release Plan',
            description='Let\'s channel this energy in a healthy way and find your inner peace.',
            immediate_activities=[
                CarePlanActivity(
                    name='Cooling Breath',
                    description='Take 10 slow, deep breaths, imagining cool air coming in and hot air going out',
                    duration='3 minutes',
                    difficulty='easy',
                    category='breathing'
                ),
                CarePlanActivity(
                    name='Physical Release',
                    description='Do jumping jacks, punch a pillow, or squeeze a stress ball',
                    duration='5 minutes',
                    difficulty='easy',
                    category='movement'
                ),
                CarePlanActivity(
                    name='Anger Acknowledgment',
                    description='Say out loud: "I am angry about [situation] and that\'s okay"',
                    duration='2 minutes',
                    difficulty='easy',
                    category='mindfulness'
                )
            ],
            daily_activities=[
                CarePlanActivity(
                    name='Anger Journal',
                    description='Write about what triggered your anger and what you need',
                    duration='15 minutes',
                    difficulty='medium',
                    category='creative'
                ),
                CarePlanActivity(
                    name='Solution Focus',
                    description='Identify one small action you can take to address the situation',
                    duration='10 minutes',
                    difficulty='medium',
                    category='mindfulness'
                )
            ],
            weekly_goals=[
                'Practice healthy anger expression techniques',
                'Identify your anger triggers and patterns',
                'Develop assertive communication skills'
            ],
            affirmations=[
                'My anger is information about what matters to me',
                'I can express my needs calmly and clearly',
                'I choose how to respond to my emotions'
            ]
        )
        
        # Calm emotion care plan
        plans['calm'] = PersonalizedCarePlan(
            emotion='calm',
            plan_name='Peaceful Maintenance Plan',
            description='You\'re in a beautiful state of calm. Let\'s nurture and maintain this peace.',
            immediate_activities=[
                CarePlanActivity(
                    name='Mindful Appreciation',
                    description='Take a moment to fully appreciate this feeling of calm',
                    duration='3 minutes',
                    difficulty='easy',
                    category='mindfulness'
                ),
                CarePlanActivity(
                    name='Gentle Reflection',
                    description='Think about what helped you reach this peaceful state',
                    duration='5 minutes',
                    difficulty='easy',
                    category='mindfulness'
                ),
                CarePlanActivity(
                    name='Calm Anchor',
                    description='Create a mental "anchor" to remember this feeling for later',
                    duration='3 minutes',
                    difficulty='easy',
                    category='mindfulness'
                )
            ],
            daily_activities=[
                CarePlanActivity(
                    name='Peace Practice',
                    description='Spend time in meditation, nature, or quiet reflection',
                    duration='20 minutes',
                    difficulty='easy',
                    category='mindfulness'
                ),
                CarePlanActivity(
                    name='Calm Creation',
                    description='Engage in a peaceful creative activity like drawing or music',
                    duration='30 minutes',
                    difficulty='easy',
                    category='creative'
                )
            ],
            weekly_goals=[
                'Maintain regular practices that bring you peace',
                'Share your calm energy with others',
                'Build resilience for when challenges arise'
            ],
            affirmations=[
                'I am at peace with myself and my life',
                'Calmness is my natural state',
                'I can return to this peace whenever I need to'
            ]
        )
        
        # Neutral emotion care plan
        plans['neutral'] = PersonalizedCarePlan(
            emotion='neutral',
            plan_name='Wellness Check-In Plan',
            description='Let\'s explore how you\'re feeling and what you might need right now.',
            immediate_activities=[
                CarePlanActivity(
                    name='Body Scan',
                    description='Check in with your body from head to toe. What do you notice?',
                    duration='5 minutes',
                    difficulty='easy',
                    category='mindfulness'
                ),
                CarePlanActivity(
                    name='Mood Check',
                    description='Ask yourself: "What am I feeling right now, even if it\'s subtle?"',
                    duration='3 minutes',
                    difficulty='easy',
                    category='mindfulness'
                ),
                CarePlanActivity(
                    name='Energy Assessment',
                    description='Rate your energy level 1-10 and think about what might help',
                    duration='2 minutes',
                    difficulty='easy',
                    category='mindfulness'
                )
            ],
            daily_activities=[
                CarePlanActivity(
                    name='Intention Setting',
                    description='Set a positive intention for your day or evening',
                    duration='10 minutes',
                    difficulty='easy',
                    category='mindfulness'
                ),
                CarePlanActivity(
                    name='Gentle Activity',
                    description='Do something mildly enjoyable - read, walk, listen to music',
                    duration='20 minutes',
                    difficulty='easy',
                    category='creative'
                )
            ],
            weekly_goals=[
                'Practice regular emotional check-ins',
                'Explore activities that bring you joy',
                'Build awareness of your emotional patterns'
            ],
            affirmations=[
                'I am open to whatever I\'m feeling',
                'It\'s okay to feel neutral sometimes',
                'I am taking good care of myself'
            ]
        )
        
        return plans
    
    def get_care_plan(self, emotion: str, user_name: Optional[str] = None) -> PersonalizedCarePlan:
        """Get a personalized care plan for the given emotion."""
        plan = self.care_plans.get(emotion, self.care_plans['neutral'])
        
        # Personalize the description if we have a user name
        if user_name:
            plan.description = f"{user_name}, {plan.description.lower()}"
        
        return plan
    
    def get_immediate_suggestion(self, emotion: str) -> CarePlanActivity:
        """Get one immediate activity suggestion for the emotion."""
        plan = self.get_care_plan(emotion)
        return random.choice(plan.immediate_activities)
    
    def get_affirmation(self, emotion: str) -> str:
        """Get a random affirmation for the emotion."""
        plan = self.get_care_plan(emotion)
        return random.choice(plan.affirmations)
