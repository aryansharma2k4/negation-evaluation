import itertools
import spacy
import os

# Load spaCy's English grammar model
print("Loading grammar model (this takes a second)...")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Error: Grammar model not found. Please run 'python -m spacy download en_core_web_sm' in your terminal first.")
    exit()

def extract_parts(sentence):
    """Uses spaCy to automatically find the subject and object."""
    doc = nlp(sentence)
    
    root_token = None
    # Find the main verb of the sentence (the ROOT)
    for token in doc:
        if token.dep_ == "ROOT":
            root_token = token
            break
            
    if not root_token:
        return None, None
        
    # Everything to the left of the verb is our subject
    subject_text = doc[:root_token.i].text.strip()
    
    # Everything to the right of the verb is our object
    object_text = doc[root_token.i + 1:].text.strip()
    
    return subject_text, object_text

def generate_ultimate_negations(full_sentence):
    # 1. Automatically extract the parts using spaCy
    subject_text, object_text = extract_parts(full_sentence)
    
    if not subject_text or not object_text:
        print(f"Error: Could not parse '{full_sentence}'. Make sure it has a clear subject, verb, and object.")
        return []
        
    # 2. Noun Modifiers
    noun_mods = [
        "", "Not ", "Not Not ", "Anything but ", "The opposite of ", "The illusion of "
    ]
    
    # 3. Verb Modifiers
    verb_mods = [
        "is", "is not", "is not not", "is maybe not", 
        "fails to be", "lacks to be", "is impossible to be", "can never be"
    ]
    
    # 4. Apply modifiers
    subject_variations = [mod + subject_text for mod in noun_mods]
    object_variations = [mod + object_text for mod in noun_mods]
    
    # 5. Generate combinations
    combinations = list(itertools.product(subject_variations, verb_mods, object_variations))
    
    negated_sentences = []
    original_sentence = full_sentence.lower().strip()
    pure_positive = f"{subject_text} is {object_text}".lower()
    
    for combo in combinations:
        # Clean up spaces
        sentence = " ".join(combo).strip()
        sentence = " ".join(sentence.split()) 
        # Capitalize first letter
        sentence = sentence[0].upper() + sentence[1:]
        
        # Filter out the pure positive version
        if sentence.lower() != original_sentence and sentence.lower() != pure_positive:
            negated_sentences.append(sentence)
            
    return negated_sentences

# --- INTERACTIVE TERMINAL LOOP ---

filename = "result.txt"
absolute_path = os.path.abspath(filename)

print("\n=============================================")
print("Ultimate Negation Generator Ready!")
print(f"File will be overwritten here: {absolute_path}")
print("Type 'quit' or 'exit' anytime to stop.")
print("=============================================\n")

while True:
    # 1. Get input from the user
    user_input = input("Enter a sentence: ").strip()
    
    # 2. Check if they want to quit
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("Exiting generator. Goodbye!")
        break
        
    # Ignore empty inputs
    if not user_input:
        continue
        
    print(f"Parsing grammar for: '{user_input}'...")
    
    # 3. Generate variations
    results = generate_ultimate_negations(user_input)
    
    if not results:
        continue
        
    # 4. Write to file using "w" (write) mode to overwrite the previous contents
    with open(filename, "w", encoding="utf-8") as file:
        file.write(f"=== Variations for: {user_input} ===\n")
        file.write(f"Total variations generated: {len(results)}\n")
        file.write("==================================================\n\n")
        
        for i, res in enumerate(results, 1):
            file.write(f"{i}. {res}\n")

    print(f"Success! '{filename}' has been overwritten with {len(results)} new variations.\n")
