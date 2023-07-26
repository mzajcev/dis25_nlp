import random

# Dictionary mapping positive or negative sentiments to a list with different movie recommendations. 
content = {
    "POSITIVE": [
        "It seems like you enjoy good movies. Have you seen 'John Wick' or 'Mad Max: Fury Road'?",
        "You appreciate comedies. 'Superbad' and 'The Hangover' are excellent choices for a good laugh.",
        "It appears you're a fan of drama. Classics such as 'The Shawshank Redemption' or 'The Godfather' could be up your alley.",
        "You enjoy science fiction. Films like 'Inception' or 'Interstellar' might intrigue you."
    ],
    "NEGATIVE": [
        "Sorry to hear you didn't enjoyed it. Perhaps you'd like something lighter like 'Paddington' or 'Shrek'?",
        "That's disappointing. If you're open to documentaries, 'The Social Dilemma' or 'Free Solo' are quite insightful.",
        "I see. How about trying a different genre? If you're open to horror, 'A Quiet Place' and 'Get Out' are thrilling experiences.",
        "It seems the film didn't live up to your expectations. Would you like to try a critically acclaimed movie like 'Parasite' or 'The Dark Knight'?"
    ]
}

# Function to give a recomandation based on sentiment
def recommend_content(sentiment):
    # If sentiment is positve or negative - Select random recommendation from the corresponding list
    if sentiment in content:
        return random.choice(content[sentiment])
    else:
        # If sentiment is not recognized, return default response.
        return "Sorry, I can't provide a recommendation at this time."
