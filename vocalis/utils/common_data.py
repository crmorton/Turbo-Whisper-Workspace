"""
Common Data Module for Vocalis

This module contains common data structures used throughout the application.
"""

# List of common English names for speaker identification
COMMON_NAMES = [
    # Male names
    "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Thomas", "Charles",
    "Christopher", "Daniel", "Matthew", "Anthony", "Mark", "Donald", "Steven", "Paul", "Andrew", "Joshua",
    "Kenneth", "Kevin", "Brian", "George", "Timothy", "Ronald", "Edward", "Jason", "Jeffrey", "Ryan",
    "Jacob", "Gary", "Nicholas", "Eric", "Jonathan", "Stephen", "Larry", "Justin", "Scott", "Brandon",
    "Benjamin", "Samuel", "Gregory", "Alexander", "Patrick", "Frank", "Raymond", "Jack", "Dennis", "Jerry",
    
    # Female names
    "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah", "Karen",
    "Lisa", "Nancy", "Betty", "Margaret", "Sandra", "Ashley", "Kimberly", "Emily", "Donna", "Michelle",
    "Carol", "Amanda", "Dorothy", "Melissa", "Deborah", "Stephanie", "Rebecca", "Sharon", "Laura", "Cynthia",
    "Kathleen", "Amy", "Angela", "Shirley", "Anna", "Ruth", "Brenda", "Pamela", "Nicole", "Katherine",
    "Samantha", "Christine", "Emma", "Catherine", "Debra", "Virginia", "Rachel", "Carolyn", "Janet", "Maria",
    
    # Gender-neutral names
    "Alex", "Jordan", "Taylor", "Casey", "Riley", "Jessie", "Jackie", "Avery", "Quinn", "Blake",
    "Morgan", "Cameron", "Reese", "Finley", "Skyler", "Frankie", "Sidney", "Kendall", "Hayden", "Parker",
    "Charlie", "Emerson", "Phoenix", "Rowan", "Dakota", "Jamie", "Harley", "Alexis", "Peyton", "Sage"
]

# Common phrases for conversation analysis
GREETING_PHRASES = [
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening", "what's up", "how are you",
    "nice to meet you", "pleasure to meet you", "how's it going", "how have you been"
]

FAREWELL_PHRASES = [
    "goodbye", "bye", "see you", "see you later", "talk to you later", "until next time", "take care",
    "have a good day", "have a nice day", "good night", "catch you later", "farewell"
]

AGREEMENT_PHRASES = [
    "yes", "yeah", "yep", "sure", "absolutely", "definitely", "certainly", "of course", "agreed",
    "that's right", "correct", "indeed", "exactly", "precisely", "true", "right"
]

DISAGREEMENT_PHRASES = [
    "no", "nope", "not really", "I don't think so", "I disagree", "that's not right", "incorrect",
    "that's wrong", "false", "nah", "absolutely not", "definitely not", "no way"
]

QUESTION_STARTERS = [
    "what", "who", "where", "when", "why", "how", "which", "whose", "whom", "can", "could", "would",
    "should", "will", "do", "does", "did", "is", "are", "was", "were", "have", "has", "had"
]

# Common audio-related terms
AUDIO_TERMS = [
    "volume", "loud", "quiet", "noise", "sound", "hear", "listen", "speaker", "microphone", "audio",
    "recording", "playback", "mute", "unmute", "echo", "feedback", "static", "distortion", "clarity",
    "background noise", "interference", "amplify", "boost", "reduce", "increase", "decrease"
]

# Common technical terms
TECH_TERMS = [
    "computer", "laptop", "desktop", "server", "cloud", "software", "hardware", "application", "app",
    "program", "code", "algorithm", "data", "database", "network", "internet", "wifi", "bluetooth",
    "device", "system", "platform", "interface", "API", "framework", "library", "module", "function",
    "variable", "parameter", "input", "output", "process", "memory", "storage", "CPU", "GPU"
]