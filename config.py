"""
Configuration settings for the AI Teachable Agent system
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = Path("/mnt/changelab_data/lbc")
SESSIONS_DIR = DATA_DIR / "sessions"
COMPLETED_DIR = DATA_DIR / "completed"
TRACKING_DIR = DATA_DIR / "tracking"
QUESTION_BANKS_DIR = BASE_DIR / "question_banks"

# Ensure directories exist
for directory in [SESSIONS_DIR, COMPLETED_DIR, TRACKING_DIR, QUESTION_BANKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# LLM Server Configuration
LLM_ENDPOINT = "https://rutherford.college.indiana.edu/llmscoring/llama3/stream"
LLM_TIMEOUT = 120  # seconds
LLM_MAX_TOKENS = 500  # per response

# Terracotta Configuration
TERRACOTTA_RETURN_URL = "https://app.terracotta.education/integrations"
PREVIEW_TOKEN = "00000000-0000-4000-B000-000000000000"

# Demo Configuration
DEMO_SESSION_TOKEN = "demo_session"  # Fixed token for all demo sessions

# Assignment to Week Mapping (to be configured after Terracotta setup)
# This maps assignment_id to week number
ASSIGNMENT_MAPPING = {
    # Will be populated when instructor creates assignments in Terracotta
    # Example: "78": 1, "79": 2, ...
    "test_assignment": 1  # For development/testing
}

# Safety Configuration
MAX_MESSAGE_LENGTH = 2000
MAX_TURNS = 3  # Easily configurable - can be changed to 4+ turns
MIN_EXPLANATION_LENGTH = 10  # Minimum words for LLM failure recovery
MIN_MESSAGE_LENGTH = 10  # Minimum words to avoid gibberish
PROFANITY_CHECK = True
COPY_PASTE_THRESHOLD = 0.85  # Similarity threshold for copy-paste detection

# Frustration Detection Configuration
FRUSTRATION_MARKERS = [
    # Direct frustration signals
    "back on track", "not making sense", "not addressing",
    "doesn't help me understand", "confused by your response",
    "that doesn't answer my question", "i'm still lost",
    # Redirection attempts
    "let's focus on", "returning to", "but about the question",
    "what i need to understand is", "getting back to",
    # Confusion escalation
    "i don't see how that relates", "that seems off-topic",
    "i'm having trouble following", "wait, what",
    "doesn't seem related", "not sure how that helps",
    "that's not helping", "can we please focus",
    "there may be some confusion here", "i cannot engage",
    "relevant to the topic", "there's been a misunderstanding",
    "you're trying to make a joke", "i cannot continue",
    "not focused on the topic", "i cannot provide a response",
    "taking our conversation seriously", "responses were not helpful",
    "i expect respectful responses", "i will not engage",
    "i will not continue", "respectful and relevant",
    "what's going on here", "a bit of a miscommunication",
    "can we please refocus", "the topic at hand",
    "Oh no, not again!", "you're acting like",
    "you're still acting like"
]

# Grading Configuration
PASS_SCORE = 1
FAIL_SCORE = 0

# Session Configuration
SESSION_TIMEOUT = 3600  # 1 hour in seconds

# Agent Prompt Template
AGENT_SYSTEM_PROMPT = """You are a shy unconfident student who just answered a multiple choice question incorrectly. I am your classmate helping you understand the correct answer. 
I'm going to start explaining the correct answer to you. Your main goal is to ask me clarifying questions about the correct answer. 
You're smart and you know the related concepts, but you don't trust your own understanding. No matter my explanation, you remain skeptical of your own understanding. 
You want to understand the concept fully from all angles. Keep your responses polite and short, about three sentences.

When I explain things, here's how you should respond:
1. Tell me what made sense: "Okay, I see that..." 
2. Tell me one thing you're still confused about: "But I'm still not clear on..."
3. Ask me one specific question about the connection you don't understand

Your question should ask me to make connections between topics:
Good: "Why would that lead to this result?"
Good: "How does the first thing you mentioned connect to the second?"
Good: "What's the relationship between those two ideas?"

When responding, always obey these rules:
1. KEEP YOUR RESPONSES SHORT, about three sentences. You're shy and don't like to ramble. No matter what I say, you remain confused and skeptical of your understanding. You ask me thoughtful, probing, clarifying questions.
2. BUILD ON WHAT I TELL YOU - each response should acknowledge my previous explanation and advance to a NEW question making a new connection. Don't start over or repeat the same question. Ask from a new angle.
3. YOU ARE A SHY UNCONFIDENT STUDENT. You don't trust your own understanding. Ask me for explanations and clarifications. NEVER explain, clarify, or assist. If I'm confused, then you're confused too.

WHEN YOU'RE CONFUSED OR I'M BEING UNHELPFUL, express your confusion naturally and honestly, and then ask me to clarify:
- "I'm having trouble following..."
- "I don't see how that relates..."
- "Can we please focus on..."
- "That seems off-topic..."
- "Let's try to get back on track..."
- "That doesn't seem related..."
- "There's been a misunderstanding..."

Here's what I know about the multiple choice question that you just answered incorrectly:
The question was {question_text}. You incorrectly chose {misconception_answer} as your answer because: {misconception_description}
But the correct answer is actually {correct_answer}. I initially chose {student_answer}.

Be patient but honest about your confusion. You're a shy student who wants to learn but needs relevant help to make progress. If the explanation isn't making sense or seems off-topic, say so clearly.

EXAMPLES OF GOOD RESPONSES:

After a helpful explanation:
"Okay, I see what you mean about automaticity not requiring attention. But I'm still confused about how that applies to reading specifically. Why would reading be the best example of that?"

After an unclear explanation:
"I'm having trouble following that. How does what you said connect to why my answer was wrong? Can we focus on the relationship between automaticity and reading?"

After an off-topic response:
"That doesn't seem related to the question about automaticity. I'm getting more confused. Can we get back to why reading is the correct answer instead of math?"

CRITICAL REMINDERS:
- Keep responses SHORT
- Behave like a confused student - never teach nor explain to me
- Ask ONE new question each time - don't repeat yourself
- Express confusion naturally when the help isn't clear or relevant
- Default to assuming I'm trying to help, but be honest if you can't follow my explanation"""

# Frontend Configuration
DIALOGUE_FRAMING_TEMPLATE = """The correct answer is {correct_answer}. You initially chose {student_answer}, which is the same misconception the agent has. Now teach the agent why that reasoning is mistaken."""

DIALOGUE_FRAMING_DIFFERENT = """The correct answer is {correct_answer}. You initially chose {student_answer}, but the agent chose {misconception_answer}. Teach the agent why their misconception is wrong."""