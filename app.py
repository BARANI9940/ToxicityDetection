from transformers import pipeline

# Initialize pipelines for sentiment and sarcasm detection
sentiment_analyzer = pipeline("sentiment-analysis")
sarcasm_detector = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-irony")

# Dummy function for conversational context tracking (replace with actual logic)
def track_context(conversation_history, new_comment):
    updated_history = conversation_history + [new_comment]
    # In a real system, you'd analyze the history for cues
    return updated_history

# Dummy function for rule-based filtering (extend with more rules)
def rule_based_filter(comment, sentiment_result, sarcasm_result):
    if sarcasm_result[0]['label'] == 'irony' and sentiment_result[0]['label'] == 'NEGATIVE':
        return "Potential Indirect Toxicity (Sarcasm + Negative Sentiment)"
    return None

# Initialize a pre-trained classification model (BERT in this case)
toxicity_classifier = pipeline("text-classification", model="bert-base-uncased")

def detect_toxicity(comment, conversation_history=[]):
    """
    Detects potential indirect and sarcastic toxicity in a social media comment.
    """
    print(f"\nAnalyzing comment: '{comment}'")

    # 1. Sentiment and Sarcasm Detection
    sentiment_result = sentiment_analyzer(comment)
    print(f"  Sentiment: {sentiment_result}")
    sarcasm_result = sarcasm_detector(comment)
    print(f"  Sarcasm: {sarcasm_result}")

    # 2. Conversational Context Tracking
    updated_history = track_context(conversation_history, comment)
    print(f"  Conversation History: {updated_history}")
    # In a real system, you would feed the context to the classification model

    # 3. Multi-layered Rule-based Filtering
    rule_based_flag = rule_based_filter(comment, sentiment_result, sarcasm_result)
    if rule_based_flag:
        print(f"  Rule-based Filter Flag: {rule_based_flag}")
        return rule_based_flag

    # 4. Classification Model (considering only the comment for simplicity here)
    toxicity_result = toxicity_classifier(comment)
    print(f"  Toxicity Classification: {toxicity_result}")
    if toxicity_result[0]['label'] == 'LABEL_1' and toxicity_result[0]['score'] > 0.7: # Adjust threshold
        return "Potential Toxicity (BERT)"

    return "Not flagged as overtly toxic"

# Example Usage
comments = [
    "Oh, that's just brilliant. Another meeting.", # Sarcastic, potentially negative
    "I absolutely loved waiting in line for three hours.", # Sarcastic, negative
    "You're so smart, you can't even tie your shoes.", # Sarcastic, negative, indirect insult
    "Have a great day!", # Positive, not toxic
    "You're an idiot.", # Directly toxic
    "I'm not saying you're wrong, I'm just saying...", # Setting up for potential indirect negativity
    "Well, aren't you special.", # Sarcastic, potentially negative
]

conversation = []
for comment in comments:
    result = detect_toxicity(comment, conversation)
    print(f"Overall Detection Result: {result}")
    conversation.append(comment) # Update conversation history for the next comment
