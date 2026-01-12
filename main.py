"""
AI Teachable Agent - FastAPI Backend
Handles Terracotta LTI launches, session management, LLM streaming, and grading
"""
import asyncio
import json
import re
import time
import numpy as np
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import quote

import aiohttp
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from profanity_check import predict, predict_prob

import config

app = FastAPI(title="AI Teachable Agent")

# CORS configuration for iframe embedding
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to Terracotta domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(config.BASE_DIR / "static")), name="static")

# ============================================================================
# SELF-EXPLANATION SCORING FUNCTIONS
# ============================================================================

def calculate_dot_product(score_probabilities: Dict[str, float]) -> float:
    """
    Calculate dot product for a dimension using configured weights.
    """
    dot_product = 0.0
    for score_label, weight in config.DIMENSION_WEIGHTS.items():
        probability = score_probabilities.get(score_label, 0.0)
        dot_product += weight * probability
    return dot_product


def evaluate_dimension(dimension_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate if a single dimension passes based on dot product and confidence.
    Returns dict with pass status, dot product, and reason.
    """
    score_probs = dimension_data.get("score_probabilities", {})
    dot_product = calculate_dot_product(score_probs)
    confidence = dimension_data.get("confidence", 0.0)
    predicted_score = dimension_data.get("predicted_score", "")
    
    result = {
        "dot_product": round(dot_product, 3),
        "confidence": round(confidence, 3),
        "predicted_score": predicted_score,
        "passed": False,
        "reason": ""
    }
    
    if dot_product >= config.DOT_PRODUCT_PASS:
        result["passed"] = True
        result["reason"] = "strong"
    elif dot_product >= config.DOT_PRODUCT_CONDITIONAL:
        # Conditional pass based on confidence
        if confidence > config.CONFIDENCE_THRESHOLD:
            # High confidence - need Good or Excellent
            if predicted_score in ["Good", "Excellent"]:
                result["passed"] = True
                result["reason"] = "good_score"
            else:
                result["reason"] = "needs_better_score"
        else:
            # Low confidence - lenient pass
            result["passed"] = True
            result["reason"] = "low_confidence_pass"
    else:
        result["reason"] = "low_dot_product"
    
    return result


async def call_scoring_api(student_response: str, context: str) -> Optional[Dict]:
    """
    Call the scoring LLM API and return the parsed response.
    """
    payload = {
        "context": context,
        "question": 'Please explain the correct answer',
        "student_response": student_response
    }
    
    try:
        timeout = aiohttp.ClientTimeout(total=config.SCORING_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                config.SCORING_ENDPOINT, 
                json=payload,
                headers={'Content-Type': 'application/json'}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"[SCORING_API] Error: Status {response.status}")
                    return None
    except asyncio.TimeoutError:
        print(f"[SCORING_API] Timeout after {config.SCORING_TIMEOUT}s")
        return None
    except Exception as e:
        print(f"[SCORING_API] Error: {str(e)}")
        return None


def build_scoring_context(session_data: Dict) -> str:
    """
    Build context string for scoring API from session data.
    """
    question = session_data["question"]
    correct_answer = question["correct_answer"]
    student_initial = session_data.get("student_initial_answer", "")
    question_text = question["question_text"]
    
    # Build context matching the format discussed
    context = (
        f"Question: {question_text} "
        f"The student initially chose {student_initial}."
        f"The correct answer is {correct_answer}. "
    )
    
    return context

# ============================================================================
# DIALOGUE FUNCTIONS
# ============================================================================

def clean_llm_response(text: str) -> str:
    """
    Remove role prefixes that LLM sometimes includes in responses.
    Examples: "Student: ", "Agent: ", "Meno: "
    """
    import re
    
    # Pattern matches "Student:", "Agent:", "Meno:" at the start (case-insensitive)
    # Allows for optional whitespace and handles variations
    pattern = r'^(Student|Agent|Meno)\s*:\s*'
    cleaned = re.sub(pattern, '', text.strip(), flags=re.IGNORECASE)
    
    return cleaned.strip()


def detect_gibberish(text: str) -> bool:
    """
    Detect if message is gibberish or spam.
    Returns True if gibberish detected.
    """
    words = text.split()
    
    # Too short
    if len(words) < config.MIN_MESSAGE_LENGTH:
        print(f"[GIBBERISH] Message too short: {len(words)} words")
        return True
    
    # High repetition (keyboard mashing)
    if len(words) > 3 and len(set(words)) / len(words) < 0.3:
        print(f"[GIBBERISH] High repetition detected")
        return True
    
    # Very few unique characters
    if len(set(text.replace(' ', ''))) < 5:
        print(f"[GIBBERISH] Too few unique characters")
        return True
    
    # No vowels in long strings (excluding common abbreviations)
    vowels = set('aeiouAEIOU')
    long_words = [w for w in words if len(w) > 4]
    if long_words:
        words_no_vowels = [w for w in long_words if not any(c in vowels for c in w)]
        if len(words_no_vowels) > len(long_words) * 0.5:
            print(f"[GIBBERISH] Too many words without vowels")
            return True
    
    # Excessive special characters
    special_chars = len(re.findall(r'[!@#$%^&*()_+=\[\]{};:,.<>?/\\|`~-]', text))
    if special_chars > len(text) * 0.3:
        print(f"[GIBBERISH] Excessive special characters")
        return True
    
    return False


def pre_filter_check(message: str) -> Optional[str]:
    """
    Pre-filter for profanity and gibberish.
    Returns failure reason if detected, None if clean.
    """
    # Check profanity first
    if config.PROFANITY_CHECK and predict_prob([message])[0] > config.PROFANITY_THRESHOLD:
        print(f"[PRE-FILTER] Profanity detected in message")
        return "inappropriate_language"
    
    # Check for gibberish
    if detect_gibberish(message):
        print(f"[PRE-FILTER] Gibberish detected in message")
        return "insufficient_response"
    
    return None


def count_historical_frustrations(dialogue: list) -> int:
    """
    Count frustration responses already in dialogue history.
    Used to detect if we're about to hit the frustration limit.
    """
    frustration_count = 0
    for turn in dialogue:
        if turn["role"] == "agent" and turn.get("message"):
            if detect_frustration_markers(turn["message"]) > 0:
                frustration_count += 1
    return frustration_count


def detect_frustration_markers(response: str) -> int:
    """
    Count frustration markers in Meno's response.
    Returns number of markers found.
    """
    response_lower = response.lower()
    markers_found = []
    
    for marker in config.FRUSTRATION_MARKERS:
        if marker.lower() in response_lower:
            markers_found.append(marker)
    
    if markers_found:
        print(f"[FRUSTRATION] Markers detected in response: {markers_found}")
        print(f"[FRUSTRATION] Total markers found: {len(markers_found)}")
    
    return len(markers_found)


def check_llm_failure_explanation(message: str) -> bool:
    """
    Check if student provided adequate explanation after LLM failure.
    Returns True if explanation is sufficient.
    """
    words = message.split()
    if len(words) >= config.MIN_EXPLANATION_LENGTH:
        print(f"[LLM_RECOVERY] Student provided {len(words)}-word explanation")
        return True
    print(f"[LLM_RECOVERY] Explanation too short: {len(words)} words")
    return False


def archive_failed_session(session_data: Dict[str, Any]):
    """
    Saves a copy of the failed session for research analysis
    before the main session is reset for a retry.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure the directory exists (idempotent)
        config.FAILURES_DIR.mkdir(exist_ok=True)
        
        filename = f"{session_data['anonymous_id']}_{session_data['launch_token']}_fail_{timestamp}.json"
        
        # Write to the failures directory
        with open(config.FAILURES_DIR / filename, "w") as f:
            json.dump(session_data, f, indent=2)
            
        print(f"[ARCHIVE] Failed attempt saved to {filename}")
        
    except Exception as e:
        print(f"[ERROR] Failed to archive session: {str(e)}")


def load_question_bank(week: int) -> Dict[str, Any]:
    """Load question bank for specified week"""
    file_path = config.QUESTION_BANKS_DIR / f"week_{week}.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Question bank for week {week} not found")
    
    with open(file_path, 'r') as f:
        return json.load(f)


def get_available_questions(week: int, condition: str, anonymous_id: str) -> list:
    """Get questions not yet used by this student this week"""
    question_bank = load_question_bank(week)
    
    # Start with all questions (condition doesn't filter question pool)
    available = question_bank["questions"]
    
    # Load tracking file to see which questions student has used
    tracking_file = config.TRACKING_DIR / f"{anonymous_id}_week_{week}.json"
    used_questions = []
    
    if tracking_file.exists():
        with open(tracking_file, 'r') as f:
            tracking_data = json.load(f)
            used_questions = tracking_data.get("used_questions", [])
    
    # Filter out used questions
    available = [q for q in available if q["id"] not in used_questions]
    
    return available


def mark_question_used(anonymous_id: str, week: int, question_id: str, launch_token: str):
    """Mark a question as used by this student"""
    tracking_file = config.TRACKING_DIR / f"{anonymous_id}_week_{week}.json"
    
    tracking_data = {"used_questions": [], "sessions": []}
    if tracking_file.exists():
        with open(tracking_file, 'r') as f:
            tracking_data = json.load(f)
    
    if question_id not in tracking_data["used_questions"]:
        tracking_data["used_questions"].append(question_id)
    
    tracking_data["sessions"].append({
        "question_id": question_id,
        "launch_token": launch_token,
        "started_at": datetime.now().isoformat()
    })
    
    with open(tracking_file, 'w') as f:
        json.dump(tracking_data, f, indent=2)


def create_session(launch_token: str, params: Dict[str, Any], question: Dict[str, Any]) -> Dict[str, Any]:
    """Create new session file"""
    session_data = {
        "launch_token": launch_token,
        "anonymous_id": params.get("anonymous_id"),
        "assignment_id": params.get("assignment_id"),
        "submission_id": params.get("submission_id"),
        "condition_name": params.get("condition_name"),
        "experiment_id": params.get("experiment_id"),
        "due_at": params.get("due_at"),
        "week": params.get("week"),
        "is_demo": params.get("is_demo", False),
        "question": question,
        "student_initial_answer": None,
        "student_initial_confidence": None,
        "student_initial_confidence_last_update": None,  # NEW: timestamp for initial confidence
        "student_final_confidence": None,
        "student_final_confidence_last_update": None,  # NEW: timestamp for final confidence
        "dialogue": [],
        "agent_decision": None,
        "score": None,
        "safety_violations": [],
        "frustration_count": 0,  # Track total frustration responses
        "llm_failure_recovery": False,  # Track if in LLM failure recovery mode
        "started_at": datetime.now().isoformat(),
        "completed_at": None
    }
    
    session_file = config.SESSIONS_DIR / f"{launch_token}.json"
    with open(session_file, 'w') as f:
        json.dump(session_data, f, indent=2)
    
    return session_data


def load_session(launch_token: str) -> Optional[Dict[str, Any]]:
    """Load session from file"""
    session_file = config.SESSIONS_DIR / f"{launch_token}.json"
    if not session_file.exists():
        return None
    
    with open(session_file, 'r') as f:
        return json.load(f)


def save_session(session_data: Dict[str, Any]):
    """Save session to file"""
    launch_token = session_data["launch_token"]
    session_file = config.SESSIONS_DIR / f"{launch_token}.json"
    
    with open(session_file, 'w') as f:
        json.dump(session_data, f, indent=2)


def complete_session(session_data: Dict[str, Any]):
    """Move session to completed directory"""
    launch_token = session_data["launch_token"]
    session_data["completed_at"] = datetime.now().isoformat()
    
    # Save to completed directory
    completed_file = config.COMPLETED_DIR / f"{launch_token}.json"
    with open(completed_file, 'w') as f:
        json.dump(session_data, f, indent=2)
    
    # Remove from sessions directory
    session_file = config.SESSIONS_DIR / f"{launch_token}.json"
    if session_file.exists():
        session_file.unlink()


def check_safety_violations(message: str) -> list:
    """Check message for safety violations (excluding profanity which is handled by pre-filter)"""
    violations = []
    
    # Check length
    if len(message) > config.MAX_MESSAGE_LENGTH:
        violations.append("message_too_long")
    
    # Check for potential copy-paste (simple heuristic: very long without punctuation)
    words = message.split()
    if len(words) > 100 and message.count('.') + message.count('?') + message.count('!') < 3:
        violations.append("possible_copy_paste")
    
    return violations


async def request_feedback_summary(system_prompt: str, dialogue_history: list, question: Dict[str, Any]) -> str:
    """
    Request a brief summary from Meno about what they learned.
    Called after 3 turns when no fail signal detected.
    """
    # Build conversation history
    conversation = ""
    for turn in dialogue_history:
        if turn["role"] == "student":
            conversation += f"Student: {turn['message']}\n\n"
        else:
            conversation += f"Agent: {turn['message']}\n\n"
    
    # Add feedback request
    feedback_request = f"The conversation is now complete. In 1-2 sentences, tell me what you learned from my explanations about why {question['options'][question['correct_answer']]} is the correct answer."
    conversation += f"Student: {feedback_request}\n\nAgent: "
    
    # Prepare request
    payload = {
        "message": conversation,
        "system_prompt": system_prompt,
        "max_tokens": 200  # Shorter - just need 1-2 sentences
    }
    
    print(f"\n[FEEDBACK REQUEST] Asking Meno for summary...")
    
    # Non-streaming request for feedback
    timeout = aiohttp.ClientTimeout(total=config.LLM_TIMEOUT)
    
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(config.LLM_ENDPOINT, json=payload) as response:
                if response.status != 200:
                    print(f"[FEEDBACK ERROR] LLM server error: {response.status}")
                    return "Thanks for helping me understand!"
                
                # Collect full response
                full_text = ""
                async for line in response.content:
                    if line:
                        try:
                            line_str = line.decode('utf-8').strip()
                            if line_str.startswith('data: '):
                                line_str = line_str[6:]
                            if line_str:
                                chunk_data = json.loads(line_str)
                                if chunk_data.get("type") == "content" and "text" in chunk_data:
                                    full_text += chunk_data["text"]
                        except (json.JSONDecodeError, Exception) as e:
                            continue
                
                # Clean up and return feedback
                feedback = clean_llm_response(full_text)
                print(f"[FEEDBACK RECEIVED] {feedback[:100]}...")
                return feedback if feedback else "Thanks for helping me understand!"
                
    except Exception as e:
        print(f"[FEEDBACK ERROR] {e}")
        return "Thanks for helping me understand!"


async def get_llm_response_non_streaming(system_prompt: str, dialogue_history: list, student_message: str) -> str:
    """
    Get LLM response without streaming (for 3rd turn MR3 generation).
    Returns the complete response text.
    """
    # Build conversation
    conversation = ""
    for turn in dialogue_history:
        if turn["role"] == "student":
            conversation += f"Student: {turn['message']}\n\n"
        else:
            conversation += f"Agent: {turn['message']}\n\n"
    
    conversation += f"Student: {student_message}\n\nAgent: "
    
    # Prepare request
    payload = {
        "message": conversation,
        "system_prompt": system_prompt,
        "max_tokens": config.LLM_MAX_TOKENS
    }
    
    print(f"\n[MR3 REQUEST] Getting Meno's response...")
    
    # Non-streaming request
    timeout = aiohttp.ClientTimeout(total=config.LLM_TIMEOUT)
    
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(config.LLM_ENDPOINT, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"LLM server error: {response.status}")
                
                # Collect full response
                full_text = ""
                async for line in response.content:
                    if line:
                        try:
                            line_str = line.decode('utf-8').strip()
                            if line_str.startswith('data: '):
                                line_str = line_str[6:]
                            if line_str:
                                chunk_data = json.loads(line_str)
                                if chunk_data.get("type") == "content" and "text" in chunk_data:
                                    full_text += chunk_data["text"]
                        except (json.JSONDecodeError, Exception) as e:
                            continue
                
                print(f"[MR3 RECEIVED] {full_text[:100]}...")
                return full_text.strip()
                
    except Exception as e:
        print(f"[MR3 ERROR] {e}")
        raise


def build_agent_prompt(question: Dict[str, Any], student_answer: str) -> str:
    """Build the system prompt for the agent"""
    return config.AGENT_SYSTEM_PROMPT.format(
        question_text=question.get("question_text", "the question"),
        topic=question.get("topic", "this topic"),
        misconception_description=question["misconception_description"],
        misconception_answer=question["misconception_answer"],
        correct_answer=question["correct_answer"],
        student_answer=student_answer
    )


def generate_meno_opening(question: Dict[str, Any], student_answer: str) -> str:
    """Generate Meno's opening message based on student's answer"""
    correct_answer = question["correct_answer"]
    agent_answer = question["misconception_answer"]
    
    # Get full text of options
    correct_option_text = question["options"][correct_answer]
    student_option_text = question["options"][student_answer]
    agent_option_text = question["options"][agent_answer]
    
    if student_answer == correct_answer:
        # Scenario 1: Student correct
        return f'Hi! I got this question wrong - I chose "{agent_option_text}". But I see the correct answer is "{correct_option_text}". I\'m having trouble understanding why. What am I missing?'
    elif student_answer == agent_answer:
        # Scenario 2: Student wrong, same as Meno
        return f'Hi! I also chose "{student_option_text}", but it looks like we both got it wrong. The correct answer is "{correct_option_text}". I\'m confused about why our answer was incorrect. What were we missing?'
    else:
        # Scenario 3: Student wrong, different from Meno
        return f'Hi! I chose "{agent_option_text}", but you chose "{student_option_text}", and it turns out we were both wrong! The correct answer is "{correct_option_text}". I still don\'t quite get it. Why do you think "{correct_option_text}" is the right answer?'


async def stream_llm_response(system_prompt: str, dialogue_history: list, student_message: str):
    """Stream response from LLM server using structured messages (Standard Llama 3 format)"""
    
    # 1. Construct the strict message list
    # This replaces the string concatenation approach to prevent 'leaks'
    messages_payload = []

    # Add System Prompt first
    if system_prompt:
        messages_payload.append({
            "role": "system", 
            "content": system_prompt
        })

    # Add History
    for turn in dialogue_history:
        # Map internal app roles to LLM roles
        # 'agent' -> 'assistant', 'student' -> 'user'
        role = "assistant" if turn['role'] == "agent" else "user"
        messages_payload.append({
            "role": role,
            "content": turn['message']
        })
    
    # Add the LATEST user message
    messages_payload.append({
        "role": "user",
        "content": student_message
    })
    
    # 2. Prepare request payload
    # Note: We now send 'messages' (list) instead of 'message' (string)
    payload = {
        "messages": messages_payload,
        "max_tokens": config.LLM_MAX_TOKENS,
        "temperature": 0.7 
    }

    # 3. Updated Logging for Debugging
    print(f"\n{'='*60}")
    print(f"SENDING TO LLM SERVER (Structured List):")
    print(f"Total messages: {len(messages_payload)}")
    print(f"Payload Preview:\n{json.dumps(payload, indent=2)}") 
    print(f"{'='*60}\n")
    
    # 4. Stream from LLM
    timeout = aiohttp.ClientTimeout(total=config.LLM_TIMEOUT)
    
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Post to Rutherford
            async with session.post(config.LLM_ENDPOINT, json=payload) as response:
                
                if response.status != 200:
                    print(f"[DEBUG] LLM server error: {response.status}")
                    error_text = await response.text()
                    print(f"[DEBUG] Error details: {error_text}")
                    yield f"data: {json.dumps({'error': f'LLM server error {response.status}'})}\n\n"
                    return
                
                # Handle JSON streaming format
                full_text = ""
                async for line in response.content:
                    if line:
                        try:
                            # Decode the line
                            line_str = line.decode('utf-8').strip()
                            
                            if line_str:
                                # Remove 'data: ' prefix if present (SSE standard)
                                if line_str.startswith('data: '):
                                    line_str = line_str[6:]
                                
                                # Check for empty lines or standard SSE 'DONE' messages
                                if line_str and line_str != '[DONE]': 
                                    # Parse JSON chunk
                                    chunk_data = json.loads(line_str)
                                    
                                    # Handle server-side errors reported in the stream
                                    if chunk_data.get("type") == "error":
                                        print(f"[DEBUG] Server streaming error: {chunk_data.get('error')}")
                                        yield f"data: {json.dumps(chunk_data)}\n\n"
                                        continue

                                    # Extract text if present
                                    if chunk_data.get("type") == "content" and "text" in chunk_data:
                                        text_chunk = chunk_data["text"]
                                        full_text += text_chunk
                                        
                                        # Send as SSE format to frontend
                                        # Note: Yielding strings is safer than mixing bytes/strings
                                        sse_line = f"data: {text_chunk}\n\n"
                                        yield sse_line
                                        
                        except json.JSONDecodeError as e:
                            print(f"[DEBUG] JSON decode error: {e}")
                            continue
                        except Exception as e:
                            print(f"[DEBUG] Error processing chunk: {e}")
                            continue
                
                print(f"[DEBUG] Streaming complete. Total text: {len(full_text)} chars")
                            
    except asyncio.TimeoutError:
        print(f"[DEBUG] LLM timeout")
        yield f"data: {json.dumps({'error': 'LLM timeout'})}\n\n"
    except Exception as e:
        print(f"[DEBUG] Exception: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


async def handle_third_turn_completion(launch_token: str, session_data: Dict[str, Any]) -> JSONResponse:
    """
    Handle the completion of the 3rd student turn.
    This involves:
    1. Getting MR3 from LLM (non-streaming)
    2. Checking MR3 for frustration markers
    3. If frustration_count >= 2: FAIL
    4. If frustration_count < 2: Request feedback summary and PASS
    5. Return JSON with decision and feedback
    """
    print("[3RD_TURN] Handling completion logic...")
    
    # Get the last student message (just added to dialogue)
    student_message = session_data["dialogue"][-1]["message"]
    
    # Build system prompt
    system_prompt = build_agent_prompt(session_data["question"], session_data["student_initial_answer"])
    
    try:
        # Call 1: Get MR3 (Meno's natural response to SR3)
        print("[3RD_TURN] Requesting MR3 from LLM...")
        mr3_text = await get_llm_response_non_streaming(
            system_prompt,
            session_data["dialogue"][:-1],  # Exclude the last student message since it's passed separately
            student_message
        )
        
        # Save MR3 to dialogue
        session_data["dialogue"].append({
            "role": "agent",
            "message": clean_llm_response(mr3_text),
            "timestamp": datetime.now().isoformat()
        })
        
        # Check MR3 for frustration
        frustration_markers = detect_frustration_markers(mr3_text)
        if frustration_markers > 0:
            session_data["frustration_count"] += 1
            print(f"[3RD_TURN] Frustration detected in MR3. Total count: {session_data['frustration_count']}")
        
        # Determine outcome based on frustration count
        if session_data["frustration_count"] >= 2:
            # FAIL - two or more frustrated responses
            print("[3RD_TURN] FAIL - Two frustrated responses detected")
            session_data["agent_decision"] = {
                "assessment": "FAIL",
                "feedback": "Student was unable to provide helpful explanations"
            }
            session_data["score"] = config.FAIL_SCORE
            save_session(session_data)
            
            return JSONResponse(content={
                "status": "third_turn_complete",
                "assessment": "FAIL",
                "feedback": "Student was unable to provide helpful explanations",
                "skip_confidence": True
            })
        
        else:
            # PASS - completed 3 turns without 2 frustrations
            print("[3RD_TURN] PASS - Requesting feedback summary...")
            
            # Call 2: Request feedback summary (Meno summarizes what they learned)
            try:
                feedback = await request_feedback_summary(
                    system_prompt,
                    session_data["dialogue"],
                    session_data["question"]
                )
            except Exception as e:
                print(f"[3RD_TURN] Failed to get feedback: {str(e)}")
                feedback = "You did a great job explaining the concept! The dialogue has been completed successfully."
            
            session_data["agent_decision"] = {
                "assessment": "PASS",
                "feedback": feedback
            }
            session_data["score"] = config.PASS_SCORE
            save_session(session_data)
            
            return JSONResponse(content={
                "status": "third_turn_complete",
                "assessment": "PASS",
                "feedback": feedback,
                "skip_confidence": False
            })
            
    except Exception as e:
        print(f"[3RD_TURN] Error: {str(e)}")
        # Fallback to PASS with generic message
        feedback = "You did a great job explaining the concept!"
        session_data["agent_decision"] = {
            "assessment": "PASS",
            "feedback": feedback
        }
        session_data["score"] = config.PASS_SCORE
        save_session(session_data)
        
        return JSONResponse(content={
            "status": "third_turn_complete",
            "assessment": "PASS",
            "feedback": feedback,
            "skip_confidence": False
        })


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/api/session/{launch_token}/evaluate_explanation")
async def evaluate_explanation(launch_token: str, request: Request):
    """
    Evaluate student's self-explanation using the scoring LLM.
    """
    session_data = load_session(launch_token)
    
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    body = await request.json()
    explanation = body.get("explanation", "").strip()
    
    # Check word count
    word_count = len(explanation.split())
    if word_count < config.MIN_EXPLANATION_WORDS:
        return {
            "status": "error",
            "error_type": "word_count",
            "word_count": word_count,
            "required": config.MIN_EXPLANATION_WORDS,
            "message": f"Please provide at least {config.MIN_EXPLANATION_WORDS} words"
        }
    
    # Build context and call scoring API
    context = build_scoring_context(session_data)
    scoring_response = await call_scoring_api(explanation, context)
    
    if not scoring_response:
        # API failed
        return {
            "status": "error",
            "error_type": "api_failure",
            "message": "An error occurred while evaluating your explanation. Please contact your instructor."
        }
    
    # Evaluate each dimension
    evaluation_results = {}
    all_passed = True
    failing_dimensions = []

    for dimension in config.SCORING_DIMENSIONS:
        if dimension in scoring_response.get("confidences", {}):
            dim_data = scoring_response["confidences"][dimension]
            eval_result = evaluate_dimension(dim_data)
            evaluation_results[dimension] = eval_result
            
            if not eval_result["passed"]:
                all_passed = False
                failing_dimensions.append(dimension)
        else:
            # Dimension missing from API - treat as pass but mark as unavailable
            evaluation_results[dimension] = {
                "dot_product": None,  # Use None to indicate N/A
                "confidence": None,
                "predicted_score": "N/A",
                "passed": True,  # Don't penalize for missing data
                "reason": "not_evaluated"
            }
        
    # Record attempt in session
    if "explanation_attempts" not in session_data:
        session_data["explanation_attempts"] = []
    
    attempt_data = {
        "text": explanation,
        "word_count": word_count,
        "timestamp": datetime.now().isoformat(),
        "raw_scores": scoring_response.get("scores", {}),
        "confidences": scoring_response.get("confidences", {}),
        "evaluation": evaluation_results,
        "passed": all_passed,
        "attempt_number": len(session_data["explanation_attempts"]) + 1
    }
    
    session_data["explanation_attempts"].append(attempt_data)
    
    # If passed, set decision and score
    if all_passed:
        session_data["agent_decision"] = {
            "assessment": "PASS",
            "feedback": "Your explanation demonstrated understanding of the concept"
        }
        session_data["score"] = config.PASS_SCORE
    
    save_session(session_data)
    
    # Build response with feedback
    feedback_messages = []
    for dim in failing_dimensions:
        feedback_messages.append(config.DIMENSION_FEEDBACK.get(dim, f"Improve {dim}"))
    
    return {
        "status": "success",
        "passed": all_passed,
        "evaluation": evaluation_results,
        "feedback": feedback_messages if not all_passed else [],
        "attempt_number": len(session_data["explanation_attempts"])
    }

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main entry point - handles both direct access and Terracotta launches"""
    
    # Get parameters from URL
    launch_token = request.query_params.get("launch_token")
    anonymous_id = request.query_params.get("anonymous_id")
    assignment_id = request.query_params.get("assignment_id")
    condition_name = request.query_params.get("condition_name", config.DEFAULT_CONDITION)
    
    # Validate condition name
    valid_conditions = [config.DIALOGUE_CONDITION_NAME, config.CONTROL_CONDITION_NAME]
    if condition_name not in valid_conditions:
        print(f"[LAUNCH] Unknown condition '{condition_name}', defaulting to {config.DEFAULT_CONDITION}")
        condition_name = config.DEFAULT_CONDITION
    
    # Check if this is a Terracotta launch or direct access
    if not launch_token:
        # Demo mode
        print("[LAUNCH] No launch token - starting demo mode")
        launch_token = config.DEMO_SESSION_TOKEN
        
        # Redirect with demo token and condition
        return RedirectResponse(
            url=f"/lbc/?launch_token={launch_token}&condition_name={condition_name}", 
            status_code=303
        )
    
    # Real launch from Terracotta
    print(f"[LAUNCH] Terracotta launch: token={launch_token}, condition={condition_name}")
    
    # Check whether this is a demo mode launch 
    is_demo = (launch_token == config.DEMO_SESSION_TOKEN) or (launch_token == config.PREVIEW_TOKEN)

    # Check for existing session
    session_data = load_session(launch_token)
    
    # FORCE NEW SESSION for Preview/Demo tokens
    if launch_token == config.PREVIEW_TOKEN or launch_token == config.DEMO_SESSION_TOKEN:
        print(f"[LAUNCH] Preview/Demo token detected. Forcing new session.")
        session_data = None  # This forces the 'else' block, below, to run
    
    if session_data:
        print(f"[LAUNCH] Existing session found")
        # Update condition if changed
        if session_data.get("condition") != condition_name:
            print(f"[LAUNCH] Updating condition from {session_data.get('condition')} to {condition_name}")
            session_data["condition"] = condition_name
            save_session(session_data)
    else:
        # New session
        print(f"[LAUNCH] Creating new session")
        
        # Extract other parameters with defaults for demo mode
        params = {
            "launch_token": launch_token,
            "anonymous_id": request.query_params.get("anonymous_id", "demo_user"),
            "assignment_id": request.query_params.get("assignment_id", "test_assignment"),
            "submission_id": request.query_params.get("submission_id"),
            "condition_name": request.query_params.get("condition_name", "dialogue"),
            "experiment_id": request.query_params.get("experiment_id"),
            "due_at": request.query_params.get("due_at"),
            "remaining_attempts": request.query_params.get("remaining_attempts"),
            "is_demo": is_demo
        }
        
        # Map assignment_id to week
        week = config.ASSIGNMENT_MAPPING.get(params["assignment_id"], 1)  # Default to week 1
        params["week"] = week

        # Initialize session_data 
        session_data = {
            "launch_token": launch_token,
            "anonymous_id": params["anonymous_id"],
            "assignment_id": params["assignment_id"],
            "is_demo": params["is_demo"],
            "week": params["week"],
            "condition": params["condition_name"],  
            "created_at": datetime.now().isoformat(),
            "dialogue": [],  # Initialize empty dialogue
            "explanation_attempts": [],  # For control condition
            "frustration_count": 0,
            "safety_violations": [],
            "student_initial_answer": None,
            "student_initial_confidence": None,
            "student_final_confidence": None,
            "question": None,
            "agent_decision": None,
            "score": None
        }

        try:
            # Get available questions
            available = get_available_questions(week, condition_name, params["anonymous_id"])
            
            if not available:
                if is_demo:
                    question_bank = load_question_bank(week)
                    available = question_bank["questions"]
                else:
                    return HTMLResponse(content="<h1>No Questions Available</h1><p>You have already completed all available questions for this week.</p>", status_code=400)
            
            # Select random question
            question = random.choice(available)
            
            if not is_demo:
                mark_question_used(anonymous_id, week, question["id"], launch_token)
            
            # Update session_data with the result from create_session
            session_data = create_session(launch_token, params, question)
            
        except Exception as e:
            import traceback
            traceback.print_exc() # Print error to server logs
            return HTMLResponse(content=f"<h1>Error</h1><p>{str(e)}</p>", status_code=500)        

        # Now when we save, we are saving the data that has the question
        save_session(session_data)
    
    # Load and return the index.html file
    with open(config.BASE_DIR / "templates/index.html", "r") as f:
        html_content = f.read()
    
    return HTMLResponse(content=html_content)

@app.get("/api/session/{launch_token}")
async def get_session(launch_token: str):
    """Get session data"""
    try:
        session_data = load_session(launch_token)
        
        if not session_data:
            print(f"[SESSION] No session found for token: {launch_token}")
            raise HTTPException(status_code=404, detail="Session not found")
        
        print(f"[SESSION] Loaded session for token: {launch_token}, condition: {session_data.get('condition', 'unknown')}")
        return session_data
    
    except Exception as e:
        print(f"[SESSION ERROR] Failed to load session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/session/{launch_token}/initial")
async def submit_initial_answer(launch_token: str, request: Request):
    """Submit student's initial answer and confidence"""
    session_data = load_session(launch_token)
    
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    body = await request.json()
    
    session_data["student_initial_answer"] = body.get("answer")
    session_data["student_initial_confidence"] = body.get("confidence")
    session_data["student_initial_confidence_last_update"] = body.get("confidence_timestamp")  # NEW
    
    # Generate Meno's opening message based on student's answer
    meno_opening = generate_meno_opening(
        session_data["question"],
        session_data["student_initial_answer"]
    )
    
    # Add Meno's opening message to dialogue
    session_data["dialogue"].append({
        "role": "agent",
        "message": meno_opening,
        "timestamp": datetime.now().isoformat()
    })
    
    save_session(session_data)
    
    return {"status": "success"}


@app.post("/api/session/{launch_token}/message")
async def send_message(launch_token: str, request: Request):
    """Handle student message and stream agent response with three-layer failure detection"""
    print(f"\n[SEND_MESSAGE] Called with token: {launch_token}")
    
    session_data = load_session(launch_token)
    print(f"[SESSION] Loaded: {session_data is not None}")
    
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session_data.get("agent_decision"):
        raise HTTPException(status_code=400, detail="Dialogue already completed")
    
    body = await request.json()
    student_message = body.get("message", "").strip()
    print(f"[STUDENT] Message received: {student_message[:50]}...")
    
    if not student_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # ========== LAYER 1: Pre-filter Check ==========
    pre_filter_result = pre_filter_check(student_message)
    if pre_filter_result:
        print(f"[PRE-FILTER] Failed with reason: {pre_filter_result}")
        
        # Immediate FAIL - don't call LLM
        session_data["agent_decision"] = {
            "assessment": "FAIL",
            "feedback": f"Dialogue ended due to {pre_filter_result.replace('_', ' ')}"
        }
        session_data["score"] = config.FAIL_SCORE
        session_data["safety_violations"].append(pre_filter_result)
        session_data["dialogue"].append({
            "role": "student",
            "message": student_message,
            "timestamp": datetime.now().isoformat()
        })
        save_session(session_data)
        
        # Return a response indicating the dialogue has ended
        # Include the decision so frontend knows not to ask for confidence
        return JSONResponse(content={
            "status": "failed", 
            "reason": pre_filter_result,
            "message": f"Dialogue ended due to {pre_filter_result.replace('_', ' ')}",
            "score": config.FAIL_SCORE,
            "skip_confidence": True
        })
    
    # Check for LLM failure recovery mode
    if session_data.get("llm_failure_recovery"):
        print("[LLM_RECOVERY] In recovery mode, checking explanation...")
        
        # Check if explanation is sufficient
        if check_llm_failure_explanation(student_message):
            session_data["agent_decision"] = {
                "assessment": "PASS",
                "feedback": "Good explanation provided after technical difficulty"
            }
            session_data["score"] = config.PASS_SCORE
            session_data["dialogue"].append({
                "role": "student",
                "message": student_message,
                "timestamp": datetime.now().isoformat()
            })
            save_session(session_data)
            return JSONResponse(content={
                "status": "completed", 
                "score": config.PASS_SCORE,
                "message": "Good explanation provided",
                "skip_confidence": False
            })
        else:
            session_data["agent_decision"] = {
                "assessment": "FAIL",
                "feedback": "Insufficient explanation provided"
            }
            session_data["score"] = config.FAIL_SCORE
            session_data["dialogue"].append({
                "role": "student",
                "message": student_message,
                "timestamp": datetime.now().isoformat()
            })
            save_session(session_data)
            return JSONResponse(content={
                "status": "failed", 
                "reason": "insufficient_explanation",
                "message": "Insufficient explanation provided",
                "score": config.FAIL_SCORE,
                "skip_confidence": True
            })
    
    # Standard safety checks (length, etc.)
    violations = check_safety_violations(student_message)
    if violations:
        session_data["safety_violations"].extend(violations)
        save_session(session_data)
        raise HTTPException(status_code=400, detail=f"Safety violation: {', '.join(violations)}")
    
    print("[CHECKS] All pre-checks passed")
    
    # ========== Check if Meno is already frustrated and student continues being unhelpful ==========
    if session_data.get("frustration_count", 0) >= 1:
        # Meno has already expressed frustration once
        # Check if this new message is clearly continuing the unhelpful pattern
        print(f"[PRE-CHECK] Meno already frustrated once, checking if student message is helpful...")
        
        message_lower = student_message.lower()
        words = student_message.split()
        
        # Strong indicators of continued unhelpful behavior
        is_clearly_unhelpful = False
        fail_reason = None
        
        # Check for explicit refusal to help
        refusal_phrases = [
            "don't know", "don't care", "whatever", "who cares",
            "doesn't matter", "not important", "can't help"
        ]
        if any(phrase in message_lower for phrase in refusal_phrases):
            is_clearly_unhelpful = True
            fail_reason = "refusing to help"
            print(f"[PRE-CHECK] Student explicitly refusing to help")
        
        # Check for complete nonsense or very short response after frustration
        if len(words) < 5:
            is_clearly_unhelpful = True  
            fail_reason = "insufficient response after warning"
            print(f"[PRE-CHECK] Response too short after frustration warning")
        
        if is_clearly_unhelpful:
            print(f"[PRE-CHECK] Failing immediately due to {fail_reason}")
            
            # Add the student message to dialogue before failing
            session_data["dialogue"].append({
                "role": "student",
                "message": student_message,
                "timestamp": datetime.now().isoformat()
            })
            
            # Immediate FAIL without calling LLM
            session_data["agent_decision"] = {
                "assessment": "FAIL",
                "feedback": f"Student was unable to provide helpful explanations after Meno expressed confusion"
            }
            session_data["score"] = config.FAIL_SCORE
            session_data["frustration_count"] += 1  # Count this as second frustration
            save_session(session_data)
            
            return JSONResponse(content={
                "status": "failed",
                "reason": fail_reason,
                "message": "Dialogue ended due to continued unhelpful responses",
                "score": config.FAIL_SCORE,
                "skip_confidence": True
            })
    
    # Check turn limit using configurable MAX_TURNS
    student_turns = sum(1 for msg in session_data["dialogue"] if msg["role"] == "student")
    if student_turns >= config.MAX_TURNS:
        print(f"[TURNS] Maximum turns ({config.MAX_TURNS}) reached")
        raise HTTPException(status_code=400, detail="Maximum turns reached")
    
    print(f"[TURNS] Current student turns: {student_turns}/{config.MAX_TURNS}")
    
    # Add student message to dialogue
    session_data["dialogue"].append({
        "role": "student",
        "message": student_message,
        "timestamp": datetime.now().isoformat()
    })
    
    save_session(session_data)
    print("[DIALOGUE] Student message saved")
    
    # ========== Check if this is the 3rd turn - handle completion differently ==========
    student_turn_count = sum(1 for msg in session_data["dialogue"] if msg["role"] == "student")
    print(f"[TURNS] Student has now made {student_turn_count} turns")
    
    if student_turn_count >= config.MAX_TURNS:
        # This is the 3rd turn - handle completion logic with two LLM calls
        print("[TURNS] 3rd turn detected - handling completion logic")
        return await handle_third_turn_completion(launch_token, session_data)
    
    # ========== Normal streaming flow for turns 1 and 2 ==========
    async def generate():
        session_data_updated = None
        try:
            # Normal flow - call LLM and stream
            system_prompt = build_agent_prompt(session_data["question"], session_data["student_initial_answer"])
            accumulated_text = ""
            llm_failed = False
            
            try:
                # Attempt to stream from LLM
                async for chunk in stream_llm_response(system_prompt, session_data["dialogue"], student_message):
                    yield chunk
                    
                    # Accumulate clean text for storage
                    chunk_str = chunk.decode('utf-8') if isinstance(chunk, bytes) else chunk
                    if chunk_str.startswith('data: '):
                        clean_text = chunk_str[6:].rstrip('\n')
                        if clean_text and not clean_text.startswith('{'):
                            accumulated_text += clean_text
            
            except Exception as e:
                print(f"[LLM_ERROR] Failed to get response: {str(e)}")
                llm_failed = True
                
                # Send error message to frontend
                error_msg = "I'm having technical difficulties. Please write a complete explanation of why the correct answer is right, and I'll give you full credit."
                yield f"data: {error_msg}\n\n".encode('utf-8')
                accumulated_text = error_msg
            
            # After streaming complete, process the response
            session_data_updated = load_session(launch_token)
            
            if llm_failed:
                # Set LLM failure recovery mode
                session_data_updated["llm_failure_recovery"] = True
                session_data_updated["dialogue"].append({
                    "role": "agent",
                    "message": clean_llm_response(accumulated_text),
                    "timestamp": datetime.now().isoformat()
                })
                save_session(session_data_updated)
                print("[LLM_ERROR] Entered recovery mode")
                return
            
            # Save agent response
            session_data_updated["dialogue"].append({
                "role": "agent",
                "message": clean_llm_response(accumulated_text),
                "timestamp": datetime.now().isoformat()
            })
            
            # ========== LAYER 2 & 3: Frustration Detection ==========
            frustration_markers = detect_frustration_markers(accumulated_text)
            if frustration_markers > 0:
                # Any frustration detected counts as 1 frustrated response
                session_data_updated["frustration_count"] += 1
                print(f"[FRUSTRATION] Response shows frustration. Total frustrated responses: {session_data_updated['frustration_count']}")
                
                # Check if this is the second frustrated response (non-consecutive OK)
                if session_data_updated["frustration_count"] >= 2:
                    print("[DECISION] FAIL - Two frustrated responses detected")
                    session_data_updated["agent_decision"] = {
                        "assessment": "FAIL",
                        "feedback": "Student was unable to provide helpful explanations"
                    }
                    session_data_updated["score"] = config.FAIL_SCORE
                    save_session(session_data_updated)
                    
                    # Need to signal the frontend to skip confidence
                    # Since we're already streaming, we'll rely on the frontend
                    # checking agent_decision.assessment === 'FAIL'
                    return
            
            # Save the session with the new dialogue
            save_session(session_data_updated)
            
        except Exception as e:
            print(f"[ERROR] Unexpected error in generate(): {str(e)}")
            # Save what we have (only if session_data_updated exists)
            if 'session_data_updated' in locals() and session_data_updated is not None:
                save_session(session_data_updated)
            raise
    
    print("[STREAM] Returning StreamingResponse")
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/session/{launch_token}/confidence")
async def submit_final_confidence(launch_token: str, request: Request):
    """Submit student's final confidence rating"""
    session_data = load_session(launch_token)
    
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    body = await request.json()
    session_data["student_final_confidence"] = body.get("confidence")
    session_data["student_final_confidence_last_update"] = body.get("confidence_timestamp")  # NEW
    
    save_session(session_data)
    
    return {"status": "success"}

@app.get("/api/config")
async def get_config():
    """Provide necessary config values to frontend"""
    return {
        "MIN_EXPLANATION_WORDS": config.MIN_EXPLANATION_WORDS,
        "DIALOGUE_CONDITION_NAME": config.DIALOGUE_CONDITION_NAME,
        "CONTROL_CONDITION_NAME": config.CONTROL_CONDITION_NAME
    }

@app.post("/api/session/{launch_token}/complete")
async def complete_dialogue(launch_token: str):
    """Complete the dialogue and return to Terracotta"""
    session_data = load_session(launch_token)
    
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    # DETERMINE PASS/FAIL LOGIC
    # logic: Fail if score is 0 (which implies frustration/safety violation or very poor performance)
    # Adjust this threshold if you want a stricter passing grade (e.g. score < 0.6)
    current_score = session_data.get("score", 0)
    is_passing = current_score > 0 

    if not is_passing:
        # --- FAIL STATE: RETRY LOOP ---
        print(f"[SESSION] Student failed attempt (Score: {current_score}). Triggering retry.")
        
        # A. Archive the failure so we don't lose the data
        archive_failed_session(session_data)
        
        # B. Reset session state for a fresh attempt at the SAME question
        session_data["dialogue"] = []
        session_data["frustration_count"] = 0
        session_data["safety_violations"] = []
        session_data["agent_decision"] = None
        session_data["student_initial_answer"] = None
        session_data["student_initial_confidence"] = None
        session_data["student_final_confidence"] = None
        session_data["score"] = None
        
        # Save the "clean" session
        save_session(session_data)
        
        # C. Return Retry Status (No Redirect)
        return {
            "status": "retry",
            "feedback": "Meno is having trouble understanding. Let's try starting over.",
            "redirect_url": None
        }

    else:
        # --- PASS STATE: SUBMIT TO TERRACOTTA ---
        print(f"[SESSION] Student passed (Score: {current_score}). Completing session.")
        
        # Mark the session as completed
        complete_session(session_data)
        
        # DETERMINE COMPLETION BEHAVIOR
        # Only the specific 'demo_session' string triggers the local reload after passing.
        if launch_token == config.DEMO_SESSION_TOKEN:
            # Demo mode - Return a demo completion URL (just reload the page)
            return {"redirect_url": "/?demo_complete=true"}
        
        # Handle Normal/Preview -> Return to Terracotta
        return_url = f"{config.TERRACOTTA_RETURN_URL}?launch_token={launch_token}&score={current_score}"
        return {"redirect_url": return_url}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/test-llm")
async def test_llm():
    """Test LLM connectivity with detailed error info"""
    import aiohttp
    import traceback
    
    payload = {
        "message": "Say hello",
        "system_prompt": "You are a helpful assistant.",
        "max_tokens": 50
    }
    
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(config.LLM_ENDPOINT, json=payload) as response:
                status = response.status
                text = await response.text()
                return {
                    "status": status,
                    "response_preview": text[:200],
                    "endpoint": config.LLM_ENDPOINT,
                    "success": True
                }
    except aiohttp.ClientConnectorError as e:
        return {"error": f"Connection error: {str(e)}", "type": "ClientConnectorError", "endpoint": config.LLM_ENDPOINT}
    except aiohttp.ClientError as e:
        return {"error": f"Client error: {str(e)}", "type": "ClientError", "endpoint": config.LLM_ENDPOINT}
    except asyncio.TimeoutError as e:
        return {"error": f"Timeout: {str(e)}", "type": "TimeoutError", "endpoint": config.LLM_ENDPOINT}
    except Exception as e:
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
            "endpoint": config.LLM_ENDPOINT
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")