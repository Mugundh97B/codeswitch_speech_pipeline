import eng_to_ipa as ipa

def is_english(word):
    return all(c.isascii() for c in word)


# Simple Hindi → IPA mapping (basic)
hindi_map = {
    "aap": "aːp",
    "hai": "hɛ",
    "ho": "ho",
    "main": "mɛ̃",
    "kya": "kja",
    "ka": "ka",
    "ki": "ki",
    "ke": "ke",
    "samajh": "səmədʒ",
    "rahe": "rəhe"
}


def hinglish_to_ipa(text):
    words = text.lower().split()
    ipa_sentence = []

    for word in words:
        if is_english(word):
            ipa_word = ipa.convert(word)
        else:
            ipa_word = hindi_map.get(word, word)  # fallback

        ipa_sentence.append(ipa_word)

    return " ".join(ipa_sentence)


if __name__ == "__main__":
    sample = "i hope aap samajh rahe ho"
    print("Original:", sample)
    print("IPA     :", hinglish_to_ipa(sample))