"""Friendly conversation without model calls; factual queries still use RAG."""
import re
import unicodedata


def conversational_reply(query):
    text = unicodedata.normalize('NFKC', query).strip().casefold()
    text = re.sub(r'[.!?,;:।🙏👋😊]+$', '', text).strip()
    if re.fullmatch(r'(?:h+i+|hello+|hey+|hlw|hlo|helo|hellow|good (?:morning|afternoon|evening))(?:\s+(?:gpa|assistant|bot))?', text):
        answer = 'Hi! Welcome to GPA Assistant. How can I help you with college notices or information today?'
    elif text in ('नमस्ते', 'नमस्कार', 'हाय', 'हेलो'):
        answer = 'नमस्ते! GPA Assistant में आपका स्वागत है। कॉलेज की सूचनाओं या जानकारी के बारे में मैं आपकी कैसे मदद कर सकता हूँ?'
    elif text in ('thanks', 'thank you', 'thankyou', 'thx', 'ty', 'धन्यवाद', 'शुक्रिया'):
        answer = 'You’re welcome! Let me know if you need help with anything else about GPA.'
    elif text in ('how are you', 'how are you doing', 'how r u'):
        answer = 'I’m here and ready to help! What would you like to know about GPA?'
    elif text in ('help', 'help me', 'what can you do', 'who are you'):
        answer = 'I’m GPA Assistant. I can help you understand uploaded college notices, find deadlines, and ask about admissions, exams or campus facilities. Ask in English or Hindi.'
    elif text in ('bye', 'goodbye', 'see you', 'good night', 'goodnight'):
        answer = 'Goodbye! You can come back whenever you need help with GPA notices or college information.'
    else:
        return None
    return {'answer': answer, 'sources': [], 'chunk_count': 0,
            'source_type': 'conversation', 'kb_match': False}
