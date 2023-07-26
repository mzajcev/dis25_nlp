import random

# Dictionary mapping positive or negative sentiments to a list with different movie recommendations. 
content = {
    "POSITIVE": [
        "It seems like you enjoy good movies. Have you seen 'John Wick' or 'Mad Max: Fury Road'?",
        "You clearly appreciated it. 'Superbad' and 'The Hangover' are excellent choices for a good laugh.",
        "It appears you're a fan of the movie you watched. Classics such as 'The Shawshank Redemption' or 'The Godfather' could be up your alley.",
        "Films like 'Inception' or 'Interstellar' might intrigue you.",
        "Your interest in uplifting movies is noticeable. Have you considered watching 'La La Land' or 'The Pursuit of Happyness'?",
        "You seem to enjoy enthralling films. 'The Dark Knight' and 'Avengers: Endgame' could be excellent picks for you.",
        "Given your fondness for good cinema, you might find 'Pulp Fiction' and 'Fight Club' captivating.",
        "You seem to have a taste for engaging storylines. 'The Matrix' and 'Lord of the Rings' trilogy could provide a great watch.",
        "From your responses, it appears you appreciate thought-provoking movies. 'The Prestige' and 'Memento' could be your next favorites.",
        "You seem to enjoy heart-warming movies. 'Forrest Gump' and 'The Notebook' might be perfect for your next movie night.",
        "Considering your positive reaction, 'Gladiator' and 'Braveheart' might also resonate with you.",
        "Given your appreciation, I bet you would love movies like 'The Revenant' and 'Birdman'.",
        "Your enjoyment of good movies shines through. 'Parasite' and 'Moonlight' might strike a chord with you.",
        "You seem to enjoy movies with a good story. 'The Social Network' and 'The Imitation Game' might be right up your alley."
    ],

    "NEGATIVE": [
        "Sorry to hear you didn't enjoyed it. Perhaps you'd like something lighter like 'Paddington' or 'Shrek'?",
        "That's disappointing. If you're open to documentaries, 'The Social Dilemma' or 'Free Solo' are quite insightful.",
        "How about trying a different genre? If you're open to horror, 'A Quiet Place' and 'Get Out' are thrilling experiences.",
        "It seems the film didn't live up to your expectations. Would you like to try a critically acclaimed movie like 'Parasite' or 'The Dark Knight'?",
        "It appears you didn't enjoy that film much. You might prefer something lighter like 'Little Miss Sunshine' or 'Love Actually'.",
        "It seems like that movie wasn't to your taste. Perhaps you'd prefer action-packed films such as 'Mission: Impossible' or 'Die Hard'.",
        "You didn't seem to enjoy that movie. Maybe you'd appreciate the suspense and thrills in 'Prisoners' or 'Gone Girl'.",
        "Given your reaction, I'm guessing you might enjoy a change of pace with comedies like 'Zombieland' or 'Anchorman'.",
        "Sounds like that film wasn't to your liking. You might find 'The Social Network' or 'Whiplash' more engaging.",
        "That film doesn't seem to have resonated with you. Perhaps you'd enjoy classics such as 'Casablanca' or 'Gone With the Wind'.",
        "From your response, it appears you didn't enjoy the movie. Maybe you'd appreciate something thought-provoking like 'A Beautiful Mind' or 'The Theory of Everything'.",
        "It seems that film didn't suit your preferences. You might enjoy movies like 'Fargo' or 'The Big Lebowski'.",
        "Your reaction suggests you didn't like the movie. Perhaps you'd enjoy more adventurous films like 'Indiana Jones' or 'Star Wars'."
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
