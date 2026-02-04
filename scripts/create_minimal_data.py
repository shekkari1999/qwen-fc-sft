"""
Create 200 very short Q&A pairs to test if model learns to stop
"""
import json
from pathlib import Path
import random

# Simple Q&A pairs with SHORT answers
qa_pairs = [
    ("What is 2 + 2?", "4"),
    ("What is 5 + 3?", "8"),
    ("What is 10 - 4?", "6"),
    ("What is 7 * 2?", "14"),
    ("What is 20 / 4?", "5"),
    ("What is the capital of France?", "Paris."),
    ("What is the capital of Japan?", "Tokyo."),
    ("What is the capital of Italy?", "Rome."),
    ("What is the capital of Germany?", "Berlin."),
    ("What is the capital of Spain?", "Madrid."),
    ("What is the capital of UK?", "London."),
    ("What is the capital of China?", "Beijing."),
    ("What is the capital of India?", "New Delhi."),
    ("What is the capital of Brazil?", "Brasília."),
    ("What is the capital of Australia?", "Canberra."),
    ("What color is the sky?", "Blue."),
    ("What color is grass?", "Green."),
    ("What color is the sun?", "Yellow."),
    ("What color is snow?", "White."),
    ("What color is coal?", "Black."),
    ("Who wrote Romeo and Juliet?", "William Shakespeare."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci."),
    ("Who discovered gravity?", "Isaac Newton."),
    ("Who invented the telephone?", "Alexander Graham Bell."),
    ("Who was the first president of USA?", "George Washington."),
    ("What is the largest planet?", "Jupiter."),
    ("What is the smallest planet?", "Mercury."),
    ("What is the closest planet to the Sun?", "Mercury."),
    ("What is the hottest planet?", "Venus."),
    ("How many planets are in our solar system?", "Eight."),
    ("How many days in a week?", "Seven."),
    ("How many months in a year?", "Twelve."),
    ("How many hours in a day?", "Twenty-four."),
    ("How many minutes in an hour?", "Sixty."),
    ("How many seconds in a minute?", "Sixty."),
    ("What is H2O?", "Water."),
    ("What is CO2?", "Carbon dioxide."),
    ("What is the chemical symbol for gold?", "Au."),
    ("What is the chemical symbol for silver?", "Ag."),
    ("What is the chemical symbol for iron?", "Fe."),
    ("Say hello.", "Hello!"),
    ("Say goodbye.", "Goodbye!"),
    ("Say thank you.", "Thank you!"),
    ("Say please.", "Please."),
    ("Say yes.", "Yes."),
    ("Say no.", "No."),
    ("Is the Earth flat?", "No, the Earth is round."),
    ("Is water wet?", "Yes."),
    ("Is fire hot?", "Yes."),
    ("Is ice cold?", "Yes."),
    ("What do cows say?", "Moo."),
    ("What do dogs say?", "Woof."),
    ("What do cats say?", "Meow."),
    ("What do birds say?", "Chirp."),
    ("What is 100 + 1?", "101"),
    ("What is 50 - 25?", "25"),
    ("What is 3 * 3?", "9"),
    ("What is 81 / 9?", "9"),
    ("What is the square root of 16?", "4"),
    ("What is the square root of 25?", "5"),
    ("What is 2 squared?", "4"),
    ("What is 3 squared?", "9"),
    ("What is 10 squared?", "100"),
    ("What year did WW2 end?", "1945."),
    ("What year did WW1 start?", "1914."),
    ("When was the Moon landing?", "1969."),
    ("What continent is Egypt in?", "Africa."),
    ("What continent is China in?", "Asia."),
    ("What continent is Brazil in?", "South America."),
    ("What ocean is the largest?", "Pacific Ocean."),
    ("What is the tallest mountain?", "Mount Everest."),
    ("What is the longest river?", "The Nile."),
    ("What is the largest desert?", "The Sahara."),
    ("Who is Mickey Mouse's girlfriend?", "Minnie Mouse."),
    ("What is Batman's real name?", "Bruce Wayne."),
    ("What is Superman's weakness?", "Kryptonite."),
    ("What is Pikachu's type?", "Electric."),
    ("How many legs does a spider have?", "Eight."),
    ("How many legs does an insect have?", "Six."),
    ("How many eyes does a spider have?", "Eight."),
    ("What do bees make?", "Honey."),
    ("What do chickens lay?", "Eggs."),
    ("What do cows produce?", "Milk."),
    ("Is the sun a star?", "Yes."),
    ("Is the moon a planet?", "No, it's a moon."),
    ("What is the speed of light?", "299,792,458 meters per second."),
    ("What is pi approximately?", "3.14159."),
    ("What is the boiling point of water?", "100 degrees Celsius."),
    ("What is the freezing point of water?", "0 degrees Celsius."),
    ("Name a primary color.", "Red."),
    ("Name a mammal.", "Dog."),
    ("Name a reptile.", "Snake."),
    ("Name a bird.", "Eagle."),
    ("Name a fish.", "Salmon."),
    ("What language is spoken in France?", "French."),
    ("What language is spoken in Germany?", "German."),
    ("What language is spoken in Japan?", "Japanese."),
    ("What language is spoken in China?", "Mandarin Chinese."),
    ("What currency does USA use?", "US Dollar."),
    ("What currency does Japan use?", "Yen."),
    ("What currency does UK use?", "Pound Sterling."),
]

# Generate 200 examples by repeating and varying
examples = []
random.seed(42)

for q, a in qa_pairs:
    examples.append({
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a}
        ]
    })

# Duplicate to get 200
while len(examples) < 200:
    ex = random.choice(examples[:len(qa_pairs)])
    examples.append(ex)

examples = examples[:200]
random.shuffle(examples)

print(f"Created {len(examples)} short Q&A examples")
print("\nSample:")
for i, ex in enumerate(examples[:5]):
    q = ex["messages"][1]["content"]
    a = ex["messages"][2]["content"]
    print(f"  Q: {q}")
    print(f"  A: {a}")
    print()

# Save
output_file = Path("data/minimal/train.jsonl")
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"Saved to {output_file}")
