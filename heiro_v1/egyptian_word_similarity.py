from heiro import Config, FollowUpTaskManager

def translate_egyptian_to_english(egyptian_word):
    # TODO: Replace this with your actual translation logic or dictionary
    translation_dict = {
        "nfr": "beautiful",
        "pr": "house",
        # Add more mappings as needed
    }
    return translation_dict.get(egyptian_word, "")

def main():
    config = Config()
    task_manager = FollowUpTaskManager(config)
    task_manager.setup()  # Ensure embeddings and model are loaded

    egyptian_text = input("Enter Egyptian text: ")
    words = egyptian_text.strip().split()

    print(f"{'Egyptian':<15} {'English':<15} {'Vec2Vec Similarity':<20}")
    print("-" * 50)
    for word in words:
        english = translate_egyptian_to_english(word)
        if not english:
            print(f"{word:<15} {'[no translation]':<15} {'N/A':<20}")
            continue
        sim = task_manager.cross_space_similarity(word, "hieroglyphic", english, "tla_english")
        print(f"{word:<15} {english:<15} {sim if sim is not None else 'N/A':<20}")

if __name__ == "__main__":
    main()
