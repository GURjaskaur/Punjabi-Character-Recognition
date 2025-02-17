import json

# List of Gurmukhi characters in the same order as your folders
gurmukhi_characters = ["ਓ","ਅ","ੲ", "ਸ", "ਹ", "ਕ", "ਖ", "ਗ", "ਘ", "ਙ", "ਚ", "ਛ", "ਜ","ਝ",
                        "ਞ","ਟ", "ਠ", "ਡ", "ਢ", "ਣ", "ਤ", "ਥ", "ਦ", "ਧ", "ਨ",
                     "ਪ", "ਫ", "ਬ", "ਭ", "ਮ", "ਯ", "ਰ", "ਲ", "ਵ","ੜ","ਸ਼","ਖ਼ ","ਗ਼ ","ਜ਼ ","ਫ਼ ","ਲ਼"
]

# Create a dictionary mapping folder names to characters
label_map = {}
for i, character in enumerate(gurmukhi_characters):
    folder_name = f"character_{i+1}"  # Adjust folder names as per your structure
    label_map[folder_name] = character

# Save the dictionary to a JSON file
with open("label_map.json", "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=4)

print("Label mapping saved to label_map.json")
