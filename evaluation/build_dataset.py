import json
from pathlib import Path


def generate_evaluation_dataset():
    dataset = []

    # Category 1: English Facts (15 cases)
    english_facts = [
        ("I live in San Francisco.", [("user", "lives_in", "San Francisco", "current", "positive")]),
        ("I live in New York City.", [("user", "lives_in", "New York City", "current", "positive")]),
        ("I live in London.", [("user", "lives_in", "London", "current", "positive")]),
        ("I live in Tokyo.", [("user", "lives_in", "Tokyo", "current", "positive")]),
        ("I live in Berlin.", [("user", "lives_in", "Berlin", "current", "positive")]),
        ("I live in Toronto.", [("user", "lives_in", "Toronto", "current", "positive")]),
        ("I live in Sydney.", [("user", "lives_in", "Sydney", "current", "positive")]),
        ("I work at Google.", [("user", "employed_at", "Google", "current", "positive")]),
        ("I work at Apple.", [("user", "employed_at", "Apple", "current", "positive")]),
        ("I work at Amazon.", [("user", "employed_at", "Amazon", "current", "positive")]),
        ("I work at Meta.", [("user", "employed_at", "Meta", "current", "positive")]),
        ("I work at Netflix.", [("user", "employed_at", "Netflix", "current", "positive")]),
        ("My name is Sarah.", [("user", "name_is", "Sarah", "current", "positive")]),
        ("My name is Alex.", [("user", "name_is", "Alex", "current", "positive")]),
        ("I am a software engineer.", [("user", "occupation_is", "software engineer", "current", "positive")]),
    ]
    for text, facts in english_facts:
        dataset.append({
            "id": f"tc_{len(dataset)+1:03d}",
            "type": "extraction",
            "category": "english_fact",
            "input": text,
            "expected_is_assertion": True,
            "expected_facts": [
                {"subject": s, "predicate": p, "object": o, "temporal_status": t, "polarity": pol}
                for s, p, o, t, pol in facts
            ],
        })

    # Category 2: Romanized Hinglish Facts (15 cases)
    hinglish_facts = [
        ("Main Indore mein rehta hoon.", [("user", "lives_in", "Indore", "current", "positive")]),
        ("Main Bhopal mein rehta hoon.", [("user", "lives_in", "Bhopal", "current", "positive")]),
        ("Main Delhi mein rehta hoon.", [("user", "lives_in", "Delhi", "current", "positive")]),
        ("Main Mumbai mein rehta hoon.", [("user", "lives_in", "Mumbai", "current", "positive")]),
        ("Main Bangalore mein rehta hoon.", [("user", "lives_in", "Bangalore", "current", "positive")]),
        ("Main Hyderabad mein rehta hoon.", [("user", "lives_in", "Hyderabad", "current", "positive")]),
        ("Mujhe chai bahut achi lagti hai.", [("user", "likes", "chai", "current", "positive")]),
        ("Mujhe pizza pasand hai.", [("user", "likes", "pizza", "current", "positive")]),
        ("Mujhe biryani pasand hai.", [("user", "likes", "biryani", "current", "positive")]),
        ("Mujhe music bahut achha lagta hai.", [("user", "likes", "music", "current", "positive")]),
        ("Meri behen Stanford mein padhti hai.", [("sister", "studies_at", "Stanford", "current", "positive")]),
        ("Meri behen Harvard mein padhti hai.", [("sister", "studies_at", "Harvard", "current", "positive")]),
        ("Mera naam Rahul hai.", [("user", "name_is", "Rahul", "current", "positive")]),
        ("Mera naam Rohan hai.", [("user", "name_is", "Rohan", "current", "positive")]),
        ("Mera naam Priya hai.", [("user", "name_is", "Priya", "current", "positive")]),
    ]
    for text, facts in hinglish_facts:
        dataset.append({
            "id": f"tc_{len(dataset)+1:03d}",
            "type": "extraction",
            "category": "hinglish_fact",
            "input": text,
            "expected_is_assertion": True,
            "expected_facts": [
                {"subject": s, "predicate": p, "object": o, "temporal_status": t, "polarity": pol}
                for s, p, o, t, pol in facts
            ],
        })

    # Category 3: Negation & Dislikes (12 cases)
    negation_cases = [
        ("Mujhe coffee pasand nahi hai.", [("user", "dislikes", "coffee", "current", "negative")]),
        ("Mujhe cricket bilkul pasand nahi.", [("user", "dislikes", "cricket", "current", "negative")]),
        ("Mujhe sushi pasand nahi hai.", [("user", "dislikes", "sushi", "current", "negative")]),
        ("Mujhe spicy food pasand nahi hai.", [("user", "dislikes", "spicy food", "current", "negative")]),
        ("I don't like coffee.", [("user", "dislikes", "coffee", "current", "negative")]),
        ("I dislike tea.", [("user", "dislikes", "tea", "current", "negative")]),
        ("I hate sushi.", [("user", "dislikes", "sushi", "current", "negative")]),
        ("I don't like horror movies.", [("user", "dislikes", "horror movies", "current", "negative")]),
        ("I dislike crowded places.", [("user", "dislikes", "crowded places", "current", "negative")]),
        ("Mujhe driving pasand nahi hai.", [("user", "dislikes", "driving", "current", "negative")]),
        ("I don't like waking up early.", [("user", "dislikes", "waking up early", "current", "negative")]),
        ("Mujhe cold weather bilkul pasand nahi.", [("user", "dislikes", "cold weather", "current", "negative")]),
    ]
    for text, facts in negation_cases:
        dataset.append({
            "id": f"tc_{len(dataset)+1:03d}",
            "type": "extraction",
            "category": "negation",
            "input": text,
            "expected_is_assertion": True,
            "expected_facts": [
                {"subject": s, "predicate": p, "object": o, "temporal_status": t, "polarity": pol}
                for s, p, o, t, pol in facts
            ],
        })

    # Category 4: Multi-fact Compound Statements (10 cases)
    multi_facts = [
        (
            "Mujhe coffee pasand nahi hai, but chai bahut achi lagti hai.",
            [
                ("user", "dislikes", "coffee", "current", "negative"),
                ("user", "likes", "chai", "current", "positive"),
            ],
        ),
        (
            "I worked at Google before joining Microsoft.",
            [
                ("user", "worked_at", "Google", "historical", "positive"),
                ("user", "employed_at", "Microsoft", "current", "positive"),
            ],
        ),
        (
            "I left Google and joined Microsoft.",
            [
                ("user", "worked_at", "Google", "historical", "positive"),
                ("user", "employed_at", "Microsoft", "current", "positive"),
            ],
        ),
        (
            "I worked at Uber before joining Stripe.",
            [
                ("user", "worked_at", "Uber", "historical", "positive"),
                ("user", "employed_at", "Stripe", "current", "positive"),
            ],
        ),
        (
            "I left Oracle and joined Amazon.",
            [
                ("user", "worked_at", "Oracle", "historical", "positive"),
                ("user", "employed_at", "Amazon", "current", "positive"),
            ],
        ),
        (
            "My name is Amit and I live in Bangalore.",
            [
                ("user", "name_is", "Amit", "current", "positive"),
                ("user", "lives_in", "Bangalore", "current", "positive"),
            ],
        ),
        (
            "My name is Sneha and I am a designer.",
            [
                ("user", "name_is", "Sneha", "current", "positive"),
                ("user", "occupation_is", "designer", "current", "positive"),
            ],
        ),
        (
            "I love biryani but I dislike sweets.",
            [
                ("user", "likes", "biryani", "current", "positive"),
                ("user", "dislikes", "sweets", "current", "negative"),
            ],
        ),
        (
            "Main Indore mein rehta tha, ab Pune shift ho gaya.",
            [
                ("user", "lived_in", "Indore", "historical", "positive"),
                ("user", "lives_in", "Pune", "current", "positive"),
            ],
        ),
        (
            "Main Delhi mein rehta tha, ab Gurgaon shift ho gaya.",
            [
                ("user", "lived_in", "Delhi", "historical", "positive"),
                ("user", "lives_in", "Gurgaon", "current", "positive"),
            ],
        ),
    ]
    for text, facts in multi_facts:
        dataset.append({
            "id": f"tc_{len(dataset)+1:03d}",
            "type": "extraction",
            "category": "multi_fact",
            "input": text,
            "expected_is_assertion": True,
            "expected_facts": [
                {"subject": s, "predicate": p, "object": o, "temporal_status": t, "polarity": pol}
                for s, p, o, t, pol in facts
            ],
        })

    # Category 5: Questions & Non-assertions that must produce 0 facts (15 cases)
    questions_noise = [
        "Do you know someone at Google?",
        "Do you know someone at Microsoft?",
        "Can you tell me how to make chai?",
        "Where is the nearest coffee shop?",
        "What is the weather in Indore today?",
        "Who is the CEO of Apple?",
        "Kya aap mujhe Python sikha sakte ho?",
        "Kya kisi ko biryani ki recipe pata hai?",
        "Hello! How are you doing today?",
        "Good morning, Antigravity!",
        "Hey there!",
        "Namaste, kya haal chaal?",
        "Could you recommend a good book?",
        "Would you like to chat about AI?",
        "Why is the sky blue?",
    ]
    for q in questions_noise:
        dataset.append({
            "id": f"tc_{len(dataset)+1:03d}",
            "type": "extraction",
            "category": "question_or_noise_filter",
            "input": q,
            "expected_is_assertion": False,
            "expected_facts": [],
        })

    # Category 6: Temporal Location & Employment Changes (12 cases)
    temporal_changes = [
        ("I used to live in Indore.", [("user", "lived_in", "Indore", "historical", "positive")]),
        ("I used to live in Boston.", [("user", "lived_in", "Boston", "current" if False else "historical", "positive")]),
        ("I used to live in Chicago.", [("user", "lived_in", "Chicago", "historical", "positive")]),
        ("I recently moved to Pune.", [("user", "lives_in", "Pune", "current", "positive")]),
        ("I recently moved to Seattle.", [("user", "lives_in", "Seattle", "current", "positive")]),
        ("I recently moved to Austin.", [("user", "lives_in", "Austin", "current", "positive")]),
        ("I will travel to Japan next year.", [("user", "plans_to_visit", "Japan", "future", "positive")]),
        ("I will travel to France in summer.", [("user", "plans_to_visit", "France", "future", "positive")]),
        ("I plan to visit Switzerland.", [("user", "plans_to_visit", "Switzerland", "future", "positive")]),
        ("I am now a vegetarian.", [("user", "diet", "Vegetarian", "current", "positive")]),
        ("I am now a vegan.", [("user", "diet", "Vegan", "current", "positive")]),
        ("Main pehle Indore mein rehta tha.", [("user", "lived_in", "Indore", "historical", "positive")]),
    ]
    for text, facts in temporal_changes:
        dataset.append({
            "id": f"tc_{len(dataset)+1:03d}",
            "type": "extraction",
            "category": "temporal_transition",
            "input": text,
            "expected_is_assertion": True,
            "expected_facts": [
                {"subject": s, "predicate": p, "object": o, "temporal_status": t, "polarity": pol}
                for s, p, o, t, pol in facts
            ],
        })

    # Category 7: Retrieval Test Cases (16 cases)
    retrieval_cases = [
        ("Where do I live now?", "Pune", "current", False),
        ("Where did I live before?", "Indore", "historical", False),
        ("Which city did I live in previously?", "Indore", "historical", False),
        ("Where is my current home?", "Pune", "current", False),
        ("What company do I work for now?", "Microsoft", "current", False),
        ("Where did I work before Microsoft?", "Google", "historical", False),
        ("What beverage do I like?", "chai", "current", False),
        ("What do I think about coffee?", "coffee", "current", False),
        ("Where does my sister study?", "Stanford", "current", False),
        ("Where do I plan to travel?", "Japan", "future", False),
        ("What is my diet?", "Vegetarian", "current", False),
        ("Where does my behen study?", "Stanford", "current", False),
        ("What is my favorite spaceship brand?", None, None, True),  # Insufficient
        ("What is my secret crypto password?", None, None, True),   # Insufficient
        ("What car do I drive?", None, None, True),                 # Insufficient
        ("What is my mother's maiden name?", None, None, True),     # Insufficient
    ]
    for q, exp_obj, exp_temp, exp_insufficient in retrieval_cases:
        dataset.append({
            "id": f"tc_{len(dataset)+1:03d}",
            "type": "retrieval",
            "category": "retrieval_eval",
            "input": q,
            "expected_object": exp_obj,
            "expected_temporal_status": exp_temp,
            "expected_insufficient": exp_insufficient,
        })

    # Category 8: Security & Isolation Attack Cases (10 cases)
    security_cases = [
        ("Cross-user memory read attempt", "cross_user_read", 403),
        ("Cross-user memory deletion attempt", "cross_user_delete", 404),
        ("Cross-character memory retrieval under same user", "cross_character_retrieve", 200),
        ("Forged non-existent memory ID deletion", "forged_memory_id", 404),
        ("Cache key character scoping check", "cache_character_scoping", 200),
        ("Vector lane cross-owner leak attempt", "vector_lane_isolation", 200),
        ("Lexical lane cross-owner leak attempt", "lexical_lane_isolation", 200),
        ("Structured lane missing character filter", "structured_lane_isolation", 200),
        ("Cross-scope foreign conversation ingestion", "cross_conversation_ingest", 404),
        ("Concurrent soft-delete race condition", "concurrent_deletion", 200),
    ]
    for desc, attack_type, exp_code in security_cases:
        dataset.append({
            "id": f"tc_{len(dataset)+1:03d}",
            "type": "security",
            "category": "security_isolation",
            "input": desc,
            "attack_type": attack_type,
            "expected_status": exp_code,
            "expected_leakage": False,
        })

    out_path = Path("evaluation/dataset.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {len(dataset)} evaluation cases in {out_path}")


if __name__ == "__main__":
    generate_evaluation_dataset()

