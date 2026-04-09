from tamil_dict import tamil_dict


def translate_to_tamil(text):
    words = text.lower().split()
    translated_words = []

    for word in words:
        tamil_word = tamil_dict.get(word, word)  # fallback = same word
        translated_words.append(tamil_word)

    return " ".join(translated_words)


if __name__ == "__main__":
    sample = "i hope aap samajh rahe ho"
    
    print("Original :", sample)
    print("Tamil    :", translate_to_tamil(sample))