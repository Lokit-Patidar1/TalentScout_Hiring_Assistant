import os
import re
from typing import Dict, List
import streamlit as st
from dotenv import load_dotenv
from prompts import INFO_GATHERING_INSTRUCTIONS, UNKNOWN_FALLBACK

# Load environment variables from .env file
load_dotenv()

from utils import (
    init_gemini_model,
    generate_questions,
    save_candidate_row,
    is_goodbye,
    get_missing_fields,
    parse_tech_stack,
    blob_sentiment,
    random_goodbye,
    resolve_supported_model,
)

APP_TITLE = "TalentScout - AI Hiring Assistant"
DATA_PATH = os.path.join("data", "candidates.csv")
MODEL_NAME = "gemini-2.5-flash"  


def _init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "candidate" not in st.session_state:
        st.session_state.candidate = {
            "name": "",
            "email": "",
            "phone": "",
            "experience": "",
            "position": "",
            "location": "",
            "tech_stack": "",
        }
    if "tech_list" not in st.session_state:
        st.session_state.tech_list: List[str] = []
    if "model" not in st.session_state:
        st.session_state.model = None
    if "model_name" not in st.session_state:
        st.session_state.model_name = MODEL_NAME
    if "current_field" not in st.session_state:
        st.session_state.current_field = ""
    if "session_ended" not in st.session_state:
        st.session_state.session_ended = False
    if "asked_questions" not in st.session_state:
        st.session_state.asked_questions: List[str] = []
    if "chatbot_mode" not in st.session_state:
        st.session_state.chatbot_mode = False
    if "info_collected" not in st.session_state:
        st.session_state.info_collected = False


def _language_texts() -> Dict[str, str]:
    return {
        "greeting": "Hello! I'm TalentScout, your AI Hiring Assistant. I'll collect a few details for initial screening. Shall we begin?",
        "ask_name": "Please share your full name.",
        "ask_email": "Please share your email address.",
        "ask_phone": "Please share your phone number.",
        "ask_experience": "How many total years of experience do you have?",
        "ask_position": "What position are you applying for?",
        "ask_location": "What is your current location?",
        "ask_tech": "Please list your tech stack (e.g., Python, Django, React).",
        "confirm_tech": "Thanks! You've listed the following technologies: ",
        "questions_intro": "Here are a few technical questions based on your tech stack:",
        "fallback": "I'm sorry, could you please clarify that?",
        "ended": "The session has ended. Thank you!",
        "summary_title": "Summary of your details",
        "sentiment": "Your mood estimate",
        "chatbot_ready": "I'm now ready to answer any questions you have! Feel free to ask me about technical topics, career advice, or anything else!",
    }


def _field_label(field_key: str, lang_texts: Dict[str, str]) -> str:
    return {
        "name": lang_texts["ask_name"],
        "email": lang_texts["ask_email"],
        "phone": lang_texts["ask_phone"],
        "experience": lang_texts["ask_experience"],
        "position": lang_texts["ask_position"],
        "location": lang_texts["ask_location"],
        "tech_stack": lang_texts["ask_tech"],
    }.get(field_key, lang_texts["fallback"])


def _extract_value_for_field(user_text: str, field: str) -> str:
    text = user_text.strip()
    if field == "email":
        # Basic heuristic
        match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
        return match.group(0) if match else text
    if field == "phone":
        digits = re.sub(r"\D", "", text)
        return digits if len(digits) >= 7 else text
    if field == "experience":
        match = re.search(r"(\d+(\.\d+)?)", text)
        return match.group(1) if match else text
    if field == "tech_stack":
        # Store as comma string; keep list separately too
        techs = parse_tech_stack(text)
        return ", ".join(techs)
    return text


def _llm_next_prompt(model, candidate: Dict[str, str]) -> str:
    known_lines = []
    for k, v in candidate.items():
        if v:
            known_lines.append(f"{k}: {v}")
    known_block = "\n".join(known_lines) if known_lines else "None yet"
    prompt = (
        f"{INFO_GATHERING_INSTRUCTIONS}\n\n"
        f"Candidate info known so far:\n{known_block}\n\n"
        f"Kindly provide a short, friendly next question to collect the missing items."
    )
    try:
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        return text if text else UNKNOWN_FALLBACK
    except Exception:
        return UNKNOWN_FALLBACK


def _chatbot_response(model, user_input: str, candidate: Dict[str, str], conversation_history: List[Dict]) -> str:
    """Generate a contextual chatbot response based on user input and candidate info."""
    # Build conversation context
    recent_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
    
    candidate_context = "\n".join([f"{k}: {v}" for k, v in candidate.items() if v])
    
    prompt = f"""You are TalentScout, a helpful AI hiring assistant. You have collected the following information about the candidate:

{candidate_context}

Recent conversation:
{history_text}

The candidate just said: "{user_input}"

Provide a helpful, friendly, and contextual response. You can:
- Answer technical questions related to their tech stack
- Provide career advice
- Discuss their experience and position
- Answer general questions
- Be conversational and supportive

Keep your response concise but informative."""

    try:
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        return text if text else UNKNOWN_FALLBACK
    except Exception as e:
        return UNKNOWN_FALLBACK


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="centered")
    _init_state()

    # Sidebar controls
    with st.sidebar:
        st.header("TalentScout")

        # Model selection (only 2.5 versions)
        model_options = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ]
        if st.session_state.model_name not in model_options:
            model_options = [st.session_state.model_name] + model_options
        st.session_state.model_name = st.selectbox(
            "Model",
            model_options,
            index=model_options.index(st.session_state.model_name)
            if st.session_state.model_name in model_options
            else 0,
        )

        # Get API key from environment variable or use fallback
        api_key = os.getenv("GOOGLE_API_KEY") or "AIzaSyBgVSeGS0aoTIYB5izpCv8T0I4Kj2k3mwo"
        if api_key:
            st.caption("✓ API key loaded successfully")
        else:
            st.caption("⚠ API key not found")

        if st.button("Reset Conversation", type="secondary"):
            for k in [
                "messages",
                "candidate",
                "tech_list",
                "current_field",
                "session_ended",
                "asked_questions",
                "chatbot_mode",
                "info_collected",
            ]:
                if k == "candidate":
                    st.session_state[k] = {
                        "name": "",
                        "email": "",
                        "phone": "",
                        "experience": "",
                        "position": "",
                        "location": "",
                        "tech_stack": "",
                    }
                elif k in ["chatbot_mode", "session_ended", "info_collected"]:
                    st.session_state[k] = False
                else:
                    st.session_state[k] = (
                        [] if isinstance(st.session_state.get(k), list) else ""
                    )
            st.rerun()

        st.divider()
        st.subheader("Collected Info")
        for key, val in st.session_state.candidate.items():
            st.write(f"- {key}: {val or '—'}")
        
        if st.session_state.chatbot_mode:
            st.success("✓ Chatbot Mode Active")

    # Model init
    if st.session_state.model is None:
        try:
            resolved = resolve_supported_model(st.session_state.model_name, api_key=api_key)
            st.session_state.model = init_gemini_model(resolved, api_key=api_key)
            st.session_state.initialized_model_name = resolved
        except Exception as e:
            st.error(f"Model initialization error: {str(e)}")
            st.error("Trying fallback model 'gemini-2.5-flash'...")
            try:
                st.session_state.model = init_gemini_model("gemini-2.5-flash", api_key=api_key)
                st.session_state.initialized_model_name = "gemini-2.5-flash"
            except Exception as e2:
                st.error(f"Fallback also failed: {str(e2)}")
                st.stop()
    else:
        # Reinitialize model if user changed selection
        if (
            getattr(st.session_state, "initialized_model_name", None)
            != st.session_state.model_name
        ):
            try:
                resolved = resolve_supported_model(
                    st.session_state.model_name, api_key=api_key
                )
                st.session_state.model = init_gemini_model(resolved, api_key=api_key)
                st.session_state.initialized_model_name = resolved
            except Exception as e:
                st.error(f"Model reinitialization error: {str(e)}")
                st.error("Trying fallback model 'gemini-2.5-flash'...")
                try:
                    st.session_state.model = init_gemini_model(
                        "gemini-2.5-flash", api_key=api_key
                    )
                    st.session_state.initialized_model_name = "gemini-2.5-flash"
                except Exception as e2:
                    st.error(f"Fallback also failed: {str(e2)}")
                    st.stop()

    lang_texts = _language_texts()

    # Initial greeting
    if not st.session_state.messages:
        st.session_state.messages.append(
            {"role": "assistant", "content": lang_texts["greeting"]}
        )
        # Ask first missing field immediately to avoid multiple questions in one line
        start_missing = get_missing_fields(st.session_state.candidate)
        if start_missing:
            st.session_state.current_field = start_missing[0]
            first_q = _field_label(st.session_state.current_field, lang_texts)
            st.session_state.messages.append({"role": "assistant", "content": first_q})

    # Chat display
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input disabled after session end
    placeholder = "Type your message..."
    user_input = st.chat_input(
        placeholder=placeholder, disabled=st.session_state.session_ended
    )

    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Sentiment (optional)
        polarity, mood = blob_sentiment(user_input)
        mood_display = {
            "positive": "😊",
            "neutral": "😐",
            "negative": "🙁",
        }.get(mood, "😐")

        st.session_state.messages.append({"role": "user", "content": user_input})

        # End sequence
        if is_goodbye(user_input):
            summary_lines = [
                f"{k}: {v}" for k, v in st.session_state.candidate.items() if v
            ]
            summary = "\n".join(summary_lines) or "No details provided."
            goodbye_msg = random_goodbye()
            
            with st.chat_message("assistant"):
                st.write(goodbye_msg)
                st.write(
                    f"({lang_texts['sentiment']}: {mood} {mood_display}, polarity={polarity:.2f})"
                )
                st.markdown(f"**{lang_texts['summary_title']}:**\n\n{summary}")
            
            st.session_state.messages.append(
                {"role": "assistant", "content": f"{goodbye_msg}\n\n{lang_texts['summary_title']}:\n{summary}"}
            )

            # Save row
            try:
                save_candidate_row(DATA_PATH, st.session_state.candidate)
            except Exception as e:
                st.warning(f"Could not save data: {e}")

            st.session_state.session_ended = True
            st.rerun()

        # If in chatbot mode, handle as conversation
        if st.session_state.chatbot_mode:
            response = _chatbot_response(
                st.session_state.model,
                user_input,
                st.session_state.candidate,
                st.session_state.messages
            )
            with st.chat_message("assistant"):
                st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

        # Collect information
        cand = st.session_state.candidate

        # Always capture the currently asked field only (one-question-per-turn)
        if st.session_state.current_field:
            field = st.session_state.current_field
            value = _extract_value_for_field(user_input, field)
            cand[field] = value

            if field == "tech_stack":
                st.session_state.tech_list = parse_tech_stack(value)

        # Determine next action
        missing_after = get_missing_fields(cand)

        if not missing_after and not st.session_state.info_collected:
            # Confirm tech stack and generate questions once
            if st.session_state.tech_list and not st.session_state.asked_questions:
                confirm_msg = (
                    f"{_language_texts()['confirm_tech']}"
                    + ", ".join(st.session_state.tech_list)
                )
                
                with st.chat_message("assistant"):
                    st.write(confirm_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": confirm_msg}
                )
                
                with st.chat_message("assistant"):
                    st.write(lang_texts["questions_intro"])
                    qs = generate_questions(
                        st.session_state.tech_list, st.session_state.model
                    )
                    if qs:
                        questions_text = "\n".join([f"- {q}" for q in qs])
                        st.write(questions_text)
                        st.session_state.asked_questions = qs
                        st.session_state.messages.append(
                            {"role": "assistant", "content": f"{lang_texts['questions_intro']}\n{questions_text}"}
                        )
                    else:
                        st.write(UNKNOWN_FALLBACK)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": UNKNOWN_FALLBACK}
                        )

                # Enable chatbot mode
                chatbot_ready_msg = lang_texts["chatbot_ready"]
                with st.chat_message("assistant"):
                    st.write(chatbot_ready_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": chatbot_ready_msg}
                )
                
                st.session_state.chatbot_mode = True
                st.session_state.info_collected = True
                st.session_state.current_field = ""
                st.rerun()
        elif missing_after:
            # Ask the next single question deterministically to reduce lag
            st.session_state.current_field = missing_after[0]
            next_q = _field_label(st.session_state.current_field, lang_texts)
            with st.chat_message("assistant"):
                st.write(next_q)
            st.session_state.messages.append({"role": "assistant", "content": next_q})
            st.rerun()


if __name__ == "__main__":
    main()