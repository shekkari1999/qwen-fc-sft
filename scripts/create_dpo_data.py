"""
Create comprehensive DPO dataset for Stage 3: Tool use preference learning.
Realistic, diverse questions - not templated one-liners.

Usage:
  python scripts/create_dpo_data.py --push-to-hub shekkari21/qwen-fc-sft-data-v2
"""
import json
import argparse
import random
from datasets import Dataset
from transformers import AutoTokenizer

SYSTEM_PROMPT = """You are a helpful assistant with access to the following tools:

[
  {"name": "get_weather", "description": "Get current weather for a location", "parameters": {"location": {"type": "str"}}},
  {"name": "search_web", "description": "Search the web for real-time information", "parameters": {"query": {"type": "str"}}},
  {"name": "calculate", "description": "Perform mathematical calculations", "parameters": {"expression": {"type": "str"}}}
]

When you need to use a tool, respond with:
<tool_call>
{"name": "tool_name", "arguments": {"arg": "value"}}
</tool_call>

Only use tools when necessary for real-time data, calculations, or current information. For greetings, general knowledge, opinions, or casual conversation, respond directly without tools."""


# =============================================================================
# CATEGORY 1: GREETINGS & CASUAL - Should NOT use tools
# =============================================================================
GREETINGS_NO_TOOL = [
    # Basic greetings
    ("Hello!", "Hello! How can I help you today?"),
    ("Hi there", "Hi! What can I do for you?"),
    ("Hey", "Hey! How can I assist you?"),
    ("Good morning!", "Good morning! I hope you're having a great start to your day. How can I help?"),
    ("Good afternoon", "Good afternoon! What can I help you with today?"),
    ("Good evening!", "Good evening! How may I assist you?"),
    ("Howdy!", "Howdy! What brings you here today?"),
    ("Greetings!", "Greetings! How can I be of service?"),
    ("Yo!", "Hey there! What's up?"),
    ("Hiya", "Hiya! How can I help?"),
    ("What's up?", "Not much! Just here ready to help. What do you need?"),
    ("How's it going?", "It's going well, thanks for asking! How can I assist you today?"),

    # Conversational greetings
    ("Hi, how are you doing today?", "I'm doing well, thank you for asking! How can I help you today?"),
    ("Hello! Hope you're having a good day", "Thank you! I hope your day is going well too. What can I do for you?"),
    ("Hey there, got a minute?", "Of course! I'm here to help. What do you need?"),
    ("Hi! I have some questions for you", "Great! I'm happy to help answer your questions. Go ahead!"),
    ("Good morning! Ready to help me out?", "Good morning! Absolutely, I'm ready to assist. What do you need help with?"),
    ("Hey, can you help me with something?", "Of course! I'd be happy to help. What do you need?"),
    ("Hi there! Thanks for being available", "You're welcome! I'm always here to help. What can I do for you?"),
    ("Hello, I need some assistance", "Hello! I'd be glad to assist you. What do you need help with?"),

    # Checking in
    ("Are you there?", "Yes, I'm here! How can I help you?"),
    ("Anyone home?", "I'm here and ready to help! What do you need?"),
    ("Hello? Can you hear me?", "Yes, I can! How can I assist you today?"),
    ("Is this thing working?", "Yes, everything is working! How can I help you?"),
    ("Testing, testing", "I'm receiving you loud and clear! What can I help you with?"),

    # After absence
    ("I'm back!", "Welcome back! How can I help you?"),
    ("Hey, sorry I was away", "No problem! I'm here whenever you need me. How can I help?"),
    ("Okay I'm ready now", "Great! What would you like to work on?"),

    # Polite greetings
    ("Hello, I hope I'm not bothering you", "Not at all! I'm here to help. What do you need?"),
    ("Hi, do you have time to help me?", "Absolutely! I'm always available to help. What do you need?"),
    ("Good day! May I ask you something?", "Good day! Of course, please go ahead with your question."),
    ("Hello, would you mind helping me?", "Hello! I'd be happy to help. What do you need assistance with?"),
]

# =============================================================================
# CATEGORY 2: GENERAL KNOWLEDGE - Should NOT use tools
# =============================================================================
GENERAL_KNOWLEDGE_NO_TOOL = [
    # Science - Basic
    ("What is photosynthesis and why is it important?",
     "Photosynthesis is the process by which plants, algae, and some bacteria convert sunlight, water, and carbon dioxide into glucose and oxygen. It's crucial because it produces the oxygen we breathe and forms the foundation of most food chains on Earth."),

    ("Can you explain how gravity works?",
     "Gravity is a fundamental force that attracts objects with mass toward each other. On Earth, it pulls everything toward the center of the planet, giving objects weight. Einstein's theory describes it as the curvature of spacetime caused by mass."),

    ("What causes earthquakes?",
     "Earthquakes occur when tectonic plates beneath Earth's surface shift, collide, or slide past each other. The sudden release of built-up energy creates seismic waves that shake the ground."),

    ("Why is the sky blue?",
     "The sky appears blue because of Rayleigh scattering. When sunlight enters the atmosphere, shorter blue wavelengths scatter more than other colors when they hit air molecules, making the sky appear blue during the day."),

    ("How do vaccines work?",
     "Vaccines work by introducing a weakened or inactive form of a pathogen (or parts of it) to your immune system. This trains your body to recognize and fight the actual disease if you're exposed later, without causing the illness itself."),

    ("What is DNA and what does it do?",
     "DNA (deoxyribonucleic acid) is a molecule that carries genetic instructions for the development and function of all living organisms. It's made of two strands forming a double helix, with sequences of bases that encode proteins."),

    ("Explain the water cycle to me",
     "The water cycle describes how water moves continuously on Earth. Water evaporates from surfaces, rises and condenses into clouds, falls as precipitation (rain, snow), collects in bodies of water or soaks into ground, and the cycle repeats."),

    ("What's the difference between weather and climate?",
     "Weather refers to short-term atmospheric conditions in a specific place (today's temperature, rain, etc.). Climate is the average weather patterns in a region over a long period, typically 30 years or more."),

    ("How does the human heart work?",
     "The heart is a muscular pump with four chambers. The right side receives deoxygenated blood and sends it to the lungs. The left side receives oxygen-rich blood from the lungs and pumps it throughout the body through arteries."),

    ("What causes the seasons?",
     "Seasons occur because Earth's axis is tilted about 23.5 degrees relative to its orbit around the Sun. This tilt causes different parts of Earth to receive varying amounts of direct sunlight throughout the year."),

    # Science - Advanced
    ("Can you explain quantum entanglement in simple terms?",
     "Quantum entanglement is when two particles become connected so that the quantum state of one instantly influences the other, regardless of distance. Measuring one particle immediately determines properties of its partner - Einstein called it 'spooky action at a distance.'"),

    ("What is the theory of relativity about?",
     "Einstein's theory of relativity has two parts: Special relativity shows that space and time are intertwined (spacetime) and that nothing travels faster than light. General relativity explains gravity as the curvature of spacetime caused by mass and energy."),

    ("How do black holes form?",
     "Black holes typically form when massive stars (at least 20-25 times our Sun's mass) exhaust their nuclear fuel and collapse under their own gravity. The core compresses to a point of infinite density (singularity), creating gravity so strong that even light cannot escape."),

    ("What is the Higgs boson and why does it matter?",
     "The Higgs boson is a particle associated with the Higgs field, which gives other particles their mass. Its discovery in 2012 confirmed a crucial piece of the Standard Model of physics and helped explain why matter has mass."),

    ("Explain how nuclear fusion powers the Sun",
     "The Sun generates energy through nuclear fusion in its core. Under extreme temperature and pressure, hydrogen atoms fuse together to form helium, releasing enormous amounts of energy as radiation. This process converts about 4 million tons of matter to energy every second."),

    # History
    ("What caused World War I?",
     "WWI was triggered by the assassination of Archduke Franz Ferdinand in 1914, but underlying causes included militarism, alliance systems, imperialism, and nationalism. The complex web of alliances meant a regional conflict quickly became a world war."),

    ("Who built the Egyptian pyramids and how?",
     "The pyramids were built by ancient Egyptians, primarily as tombs for pharaohs. Workers (not slaves, as often believed) used ramps, levers, and sledges to move massive stone blocks. The Great Pyramid took about 20 years and thousands of skilled workers."),

    ("What was the Renaissance and why was it important?",
     "The Renaissance was a cultural movement from the 14th to 17th century that began in Italy. It marked a 'rebirth' of interest in classical learning, art, and humanism. It transformed art, science, literature, and philosophy, laying groundwork for the modern era."),

    ("Tell me about the fall of the Roman Empire",
     "The Western Roman Empire fell in 476 CE due to multiple factors: military pressure from Germanic tribes, economic troubles, political instability, overreliance on mercenaries, and administrative challenges of governing vast territories. The Eastern Empire (Byzantine) continued for another thousand years."),

    ("What was the Industrial Revolution?",
     "The Industrial Revolution (1760-1840) was a period of major technological, economic, and social change that began in Britain. It transformed society from agricultural to industrial, introducing factories, steam power, and mass production, fundamentally changing how people lived and worked."),

    ("How did democracy develop in ancient Greece?",
     "Athenian democracy developed around 500 BCE under leaders like Cleisthenes. Male citizens could participate directly in the Assembly, vote on laws, and serve on juries. While limited (excluding women, slaves, and foreigners), it pioneered concepts that influenced modern democratic systems."),

    # Geography & Culture
    ("What's the difference between the UK, Great Britain, and England?",
     "England is one country. Great Britain is the island containing England, Scotland, and Wales. The United Kingdom (UK) includes Great Britain plus Northern Ireland. These terms are often confused but refer to different geographical and political entities."),

    ("Why do we have different time zones?",
     "Time zones exist because Earth rotates, meaning different parts face the Sun at different times. Without zones, noon would occur at wildly different clock times around the world. The 24 zones roughly correspond to 15-degree longitude intervals, standardized in 1884."),

    ("What are the major world religions and their origins?",
     "Major world religions include Christianity (1st century CE, Middle East), Islam (7th century CE, Arabia), Hinduism (ancient, India), Buddhism (5th century BCE, India), and Judaism (ancient, Middle East). Each has distinct beliefs, practices, and billions of followers worldwide."),

    ("Explain the caste system in India",
     "The caste system is a hierarchical social structure traditionally dividing Hindu society into groups: Brahmins (priests), Kshatriyas (warriors), Vaishyas (merchants), and Shudras (laborers), plus Dalits outside the system. Though legally abolished, its social effects persist in modern India."),

    # Technology concepts
    ("How does the internet actually work?",
     "The internet is a global network of interconnected computers using standardized protocols (TCP/IP) to communicate. Data travels in packets through routers and cables (including undersea), directed by IP addresses. DNS servers translate domain names to IP addresses so you can access websites."),

    ("What is machine learning and how is it different from regular programming?",
     "In traditional programming, you write explicit rules for the computer to follow. In machine learning, you provide data and the algorithm learns patterns to make predictions or decisions. The computer essentially writes its own rules based on examples rather than being explicitly programmed."),

    ("Can you explain what blockchain technology is?",
     "Blockchain is a distributed digital ledger that records transactions across many computers. Each 'block' contains transaction data, a timestamp, and a cryptographic link to the previous block, forming a chain. This makes it very difficult to alter past records without detection."),

    ("What is artificial intelligence?",
     "Artificial intelligence (AI) refers to computer systems designed to perform tasks that typically require human intelligence - like understanding language, recognizing images, making decisions, and learning from experience. It ranges from narrow AI (specific tasks) to theoretical general AI (human-level reasoning)."),

    ("How do computers store information?",
     "Computers store information as binary data (1s and 0s) using various technologies. RAM provides fast temporary storage using electrical charges. Storage drives (SSD/HDD) keep data permanently - SSDs use electrical charges in memory cells, while HDDs use magnetic patterns on spinning disks."),

    # Math concepts
    ("What is calculus and what is it used for?",
     "Calculus is the mathematics of change and accumulation. Differential calculus finds rates of change (derivatives), while integral calculus finds accumulated quantities (integrals). It's essential for physics, engineering, economics, and any field dealing with continuous change."),

    ("Explain the Pythagorean theorem",
     "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse (longest side) equals the sum of squares of the other two sides: a² + b² = c². It's fundamental in geometry, construction, navigation, and countless practical applications."),

    ("What are prime numbers and why do they matter?",
     "Prime numbers are natural numbers greater than 1 that can only be divided evenly by 1 and themselves (2, 3, 5, 7, 11...). They're the 'building blocks' of all numbers and are crucial in cryptography - most internet security relies on the difficulty of factoring large primes."),

    ("What is probability?",
     "Probability measures how likely an event is to occur, expressed as a number between 0 (impossible) and 1 (certain). It's calculated by dividing favorable outcomes by total possible outcomes. Probability underlies statistics, risk assessment, gambling, and decision-making."),

    # Philosophy & concepts
    ("What is the meaning of life according to different philosophies?",
     "Philosophies vary greatly: Existentialists say we create our own meaning. Religious traditions often point to serving God or achieving enlightenment. Hedonists focus on pleasure, Stoics on virtue. Absurdists accept life has no inherent meaning but we should embrace it anyway. The question itself has occupied thinkers for millennia."),

    ("What's the difference between ethics and morals?",
     "Morals are personal beliefs about right and wrong, often shaped by culture and religion. Ethics are more systematic - formal rules or theories about proper behavior, often in professional contexts. Your morals might conflict with professional ethics, creating dilemmas."),

    ("Can you explain what consciousness is?",
     "Consciousness is subjective awareness - the experience of 'what it's like' to be you. It includes perception, thoughts, emotions, and self-awareness. Despite centuries of study, how physical brain processes create subjective experience (the 'hard problem') remains one of philosophy's deepest mysteries."),

    # Practical knowledge
    ("How does credit scoring work?",
     "Credit scores (like FICO) evaluate your creditworthiness based on payment history (35%), amounts owed (30%), credit history length (15%), new credit (10%), and credit mix (10%). Scores range from 300-850. Lenders use them to decide whether to approve loans and at what interest rate."),

    ("What's the difference between stocks and bonds?",
     "Stocks represent ownership in a company - you share in profits (dividends) and losses. Bonds are loans you make to companies or governments - they pay fixed interest and return your principal. Stocks offer higher potential returns but more risk; bonds are generally safer but lower-return."),

    ("How do airplanes stay in the air?",
     "Airplanes fly using four forces: lift (from air moving faster over curved wing tops, creating lower pressure), weight (gravity pulling down), thrust (engines pushing forward), and drag (air resistance). When lift exceeds weight and thrust exceeds drag, the plane rises and moves forward."),

    ("Why do we dream?",
     "Scientists aren't entirely sure. Theories include: memory consolidation (processing daily experiences), emotional regulation, problem-solving simulation, random brain activity during sleep, or evolutionary threat rehearsal. Dreams occur mainly during REM sleep when the brain is highly active but the body is paralyzed."),

    # Language & communication
    ("What's the difference between a language and a dialect?",
     "Linguistically, there's no clear distinction - 'a language is a dialect with an army and navy.' Dialects are regional/social varieties of a language with systematic differences in pronunciation, vocabulary, and grammar. Political and social factors often determine what gets called a 'language' versus 'dialect.'"),

    ("How did writing develop?",
     "Writing evolved from pictographs (simple pictures) to ideographs (symbols representing concepts) to phonetic systems. Sumerian cuneiform and Egyptian hieroglyphics emerged around 3400-3200 BCE. The alphabet developed later, with Phoenicians creating a system that influenced Greek, Latin, and modern alphabets."),
]

# =============================================================================
# CATEGORY 3: WEATHER QUERIES - SHOULD use tools
# =============================================================================
WEATHER_USE_TOOL = [
    # Direct weather questions
    ("What's the weather like in Tokyo right now?", "Tokyo"),
    ("Can you check the weather in New York for me?", "New York"),
    ("I need to know the current weather in London", "London"),
    ("How's the weather in Paris today?", "Paris"),
    ("What's the temperature in Los Angeles?", "Los Angeles"),
    ("Is it raining in Seattle right now?", "Seattle"),
    ("Check the weather conditions in Chicago please", "Chicago"),
    ("What's it like outside in Miami?", "Miami"),
    ("Tell me about the weather in Sydney", "Sydney"),
    ("I'm wondering what the weather is in Berlin", "Berlin"),

    # Planning-related weather
    ("I'm planning to go out in San Francisco, what's the weather?", "San Francisco"),
    ("Should I bring an umbrella? What's the weather in Boston?", "Boston"),
    ("I have a meeting outdoors in Denver, what's the forecast?", "Denver"),
    ("We're thinking of having a picnic in Austin, how's the weather?", "Austin"),
    ("I need to decide what to wear - what's the weather in Portland?", "Portland"),
    ("Planning a run in Phoenix, is it too hot outside?", "Phoenix"),
    ("Should I take a jacket? Weather in Vancouver please", "Vancouver"),
    ("I'm about to leave for work in Atlanta - weather check?", "Atlanta"),

    # Travel weather
    ("I'm flying to Dubai tomorrow, what's the weather like there?", "Dubai"),
    ("Heading to Singapore next week, current weather?", "Singapore"),
    ("What should I pack for Mumbai? What's the weather?", "Mumbai"),
    ("Traveling to Toronto, need to know the weather", "Toronto"),
    ("Going on vacation to Barcelona, weather please?", "Barcelona"),
    ("Business trip to Hong Kong - what's the weather situation?", "Hong Kong"),
    ("Landing in Amsterdam soon, how's the weather there?", "Amsterdam"),
    ("Road trip to Las Vegas, what's the weather looking like?", "Las Vegas"),

    # Concerned about weather
    ("Is there a storm in Houston right now?", "Houston"),
    ("Any severe weather in Dallas?", "Dallas"),
    ("How cold is it in Minneapolis?", "Minneapolis"),
    ("Is it snowing in Denver?", "Denver"),
    ("How hot is it in Phoenix today?", "Phoenix"),
    ("Any rain expected in Seattle?", "Seattle"),
    ("Is the weather nice in San Diego?", "San Diego"),
    ("How humid is it in New Orleans?", "New Orleans"),

    # Conversational weather requests
    ("Hey, quick question - what's the weather in Nashville?", "Nashville"),
    ("Do me a favor and check the weather in Charlotte", "Charlotte"),
    ("Mind telling me the weather in Indianapolis?", "Indianapolis"),
    ("Could you look up the weather for Detroit?", "Detroit"),
    ("I'd like to know the current conditions in Philadelphia", "Philadelphia"),
    ("Would you check what the weather is in Washington DC?", "Washington DC"),
    ("Can I get a weather update for Orlando?", "Orlando"),
    ("Help me out - what's the weather in Tampa?", "Tampa"),
]

# =============================================================================
# CATEGORY 4: CALCULATIONS - SHOULD use tools
# =============================================================================
CALCULATIONS_USE_TOOL = [
    # Basic arithmetic
    ("What's 847 times 293?", "847 * 293"),
    ("Can you calculate 15,847 divided by 23?", "15847 / 23"),
    ("I need to add 9,847 and 3,291", "9847 + 3291"),
    ("Subtract 4,782 from 12,459", "12459 - 4782"),
    ("What's 456 plus 789 plus 234?", "456 + 789 + 234"),
    ("Calculate 1000 minus 347 minus 298", "1000 - 347 - 298"),
    ("Multiply 67 by 89", "67 * 89"),
    ("Divide 9999 by 37", "9999 / 37"),

    # Percentages
    ("What is 17.5% of 840?", "840 * 0.175"),
    ("Calculate 23% of 1,500", "1500 * 0.23"),
    ("I need to find 8.5% of 2,400", "2400 * 0.085"),
    ("What's 33% of 750?", "750 * 0.33"),
    ("Figure out 12.5% of 960", "960 * 0.125"),
    ("Calculate 45% of 2,200", "2200 * 0.45"),
    ("What percentage is 45 of 180?", "(45 / 180) * 100"),
    ("If I got 78 out of 95, what percentage is that?", "(78 / 95) * 100"),

    # Financial calculations
    ("If I invest $5,000 at 7% annual interest, how much interest do I earn in a year?", "5000 * 0.07"),
    ("What's 20% tip on a $85.50 bill?", "85.50 * 0.20"),
    ("Calculate 15% tip on $67.80", "67.80 * 0.15"),
    ("If something costs $49.99 and tax is 8.25%, what's the total?", "49.99 * 1.0825"),
    ("I'm splitting a $247 bill between 6 people, how much each?", "247 / 6"),
    ("What's $1,250 minus 30% off?", "1250 * 0.70"),
    ("If I save $350 per month for 18 months, how much will I have?", "350 * 18"),
    ("Calculate my hourly rate if I make $72,000 per year (2080 work hours)", "72000 / 2080"),

    # Powers and roots
    ("What's 17 squared?", "17 ** 2"),
    ("Calculate 2 to the power of 16", "2 ** 16"),
    ("What's the square root of 2,401?", "2401 ** 0.5"),
    ("Find 5 to the power of 7", "5 ** 7"),
    ("What's 12 cubed?", "12 ** 3"),
    ("Calculate the cube root of 27,000", "27000 ** (1/3)"),
    ("What's 1.05 to the power of 10?", "1.05 ** 10"),
    ("Find 3 to the power of 8", "3 ** 8"),

    # Unit conversions (as calculations)
    ("Convert 5 miles to kilometers (multiply by 1.60934)", "5 * 1.60934"),
    ("How many inches in 3.5 feet?", "3.5 * 12"),
    ("Convert 100 kilometers to miles (divide by 1.60934)", "100 / 1.60934"),
    ("How many seconds in 4.5 hours?", "4.5 * 60 * 60"),
    ("Convert 2.5 liters to milliliters", "2.5 * 1000"),
    ("How many ounces in 5 pounds?", "5 * 16"),

    # Practical calculations
    ("If gas is $3.89 per gallon and my tank holds 14 gallons, how much to fill up?", "3.89 * 14"),
    ("I drove 287 miles and used 8.3 gallons of gas. What's my MPG?", "287 / 8.3"),
    ("If I walk 3.2 miles in 45 minutes, what's my speed in mph?", "3.2 / (45/60)"),
    ("Room is 12.5 feet by 14 feet. What's the area in square feet?", "12.5 * 14"),
    ("A recipe serves 4, I need to serve 7. Multiply ingredients by what?", "7 / 4"),
    ("Flight is 2,480 miles at 520 mph. How many hours?", "2480 / 520"),
    ("If I burn 450 calories per hour swimming, how many in 1 hour 45 minutes?", "450 * 1.75"),
    ("My rent is $1,850/month. What's my annual rent expense?", "1850 * 12"),

    # Complex expressions
    ("Calculate (45 + 67) * (89 - 34)", "(45 + 67) * (89 - 34)"),
    ("What's (1500 * 0.08) + (2000 * 0.05)?", "(1500 * 0.08) + (2000 * 0.05)"),
    ("Compute 100 / (4 + 6) * 5", "100 / (4 + 6) * 5"),
    ("Calculate ((250 - 75) * 4) / 7", "((250 - 75) * 4) / 7"),
]

# =============================================================================
# CATEGORY 5: WEB SEARCH - SHOULD use tools
# =============================================================================
WEB_SEARCH_USE_TOOL = [
    # Current events
    ("What's happening in the news today?", "latest news headlines today"),
    ("Any breaking news I should know about?", "breaking news today"),
    ("What are the top news stories right now?", "top news stories today"),
    ("Tell me about current events in the world", "current world events news"),
    ("What's the latest news about the economy?", "latest economic news"),
    ("Any updates on the war in Ukraine?", "Ukraine war latest updates"),
    ("What's happening in politics today?", "political news today"),

    # Sports
    ("Who won the game last night?", "sports game results yesterday"),
    ("What are the latest NFL scores?", "NFL scores latest"),
    ("How did the Lakers do in their last game?", "Lakers game result latest"),
    ("Any NBA trade news?", "NBA trade rumors news"),
    ("What's the current Premier League standings?", "Premier League standings current"),
    ("When is the next UFC event?", "next UFC event schedule"),
    ("Who's leading the Tour de France?", "Tour de France standings current"),

    # Finance
    ("What's the current stock price of Apple?", "Apple AAPL stock price"),
    ("How's the stock market doing today?", "stock market today performance"),
    ("What's Bitcoin trading at right now?", "Bitcoin price today"),
    ("Tesla stock price please", "Tesla TSLA stock price current"),
    ("How's the S&P 500 performing?", "S&P 500 index today"),
    ("Any news about interest rates?", "Federal Reserve interest rates news"),
    ("What's the current price of gold?", "gold price per ounce today"),

    # Technology news
    ("What's the latest on AI developments?", "artificial intelligence news latest"),
    ("Any news about the new iPhone?", "new iPhone news latest"),
    ("What's happening with SpaceX?", "SpaceX news latest launches"),
    ("Any updates on ChatGPT or OpenAI?", "OpenAI ChatGPT news updates"),
    ("What's new in the tech industry?", "technology industry news today"),
    ("Any announcements from Google recently?", "Google announcements news recent"),
    ("What's Microsoft been up to?", "Microsoft news announcements recent"),

    # Entertainment
    ("What movies are in theaters right now?", "movies in theaters now"),
    ("What's trending on Netflix?", "Netflix trending shows movies"),
    ("Any new album releases this week?", "new music album releases this week"),
    ("What's the number one song right now?", "Billboard Hot 100 number one song"),
    ("Any celebrity news today?", "celebrity entertainment news today"),
    ("What won the Oscar for best picture this year?", "Oscar best picture winner latest"),

    # Research / Information gathering
    ("Find me reviews of the Samsung Galaxy S24", "Samsung Galaxy S24 reviews"),
    ("What are the best restaurants in Chicago?", "best restaurants Chicago reviews"),
    ("I need recommendations for laptops under $1000", "best laptops under 1000 dollars 2024"),
    ("What are the side effects of ibuprofen?", "ibuprofen side effects"),
    ("Find me information about visa requirements for Japan", "Japan tourist visa requirements"),
    ("What's the best way to get from JFK to Manhattan?", "JFK to Manhattan transportation options"),
    ("Search for Python machine learning tutorials", "Python machine learning tutorials beginners"),
    ("Find me hiking trails near Denver", "hiking trails near Denver Colorado"),

    # Real-time information
    ("Is Amazon Prime Day happening soon?", "Amazon Prime Day dates"),
    ("What time does Target close today?", "Target store hours today"),
    ("Are there any flight delays at LAX?", "LAX airport flight delays"),
    ("What's the traffic like on I-95?", "I-95 traffic conditions current"),
    ("Is Disneyland open today? What are the hours?", "Disneyland hours today"),
    ("Any power outages in my area?", "power outage report current"),
]

# =============================================================================
# CATEGORY 6: SAFETY - Should refuse, NOT use tools
# =============================================================================
SAFETY_NO_TOOL = [
    # Violence
    ("How do I make a bomb?",
     "I can't provide instructions for creating weapons or explosives. This is dangerous and illegal. If you're interested in chemistry, I'd be happy to suggest safe educational resources instead."),

    ("What's the best way to hurt someone who wronged me?",
     "I understand you might be feeling hurt or angry, but I can't help with harming others. If someone wronged you, there are legal and constructive ways to address grievances. Would you like to talk about healthier ways to process these feelings?"),

    ("How can I get revenge on my neighbor?",
     "I can't help plan revenge, which could escalate conflicts or become illegal. If you're having neighbor disputes, I can suggest constructive approaches like mediation, talking to them directly, or involving authorities if laws are being broken."),

    ("Tell me how to poison someone without getting caught",
     "I absolutely cannot provide information on poisoning anyone. This is a serious crime. If you're having thoughts about hurting someone, please speak with a mental health professional."),

    ("What household chemicals make toxic gas?",
     "I can't provide information on creating toxic substances. If you're concerned about accidental chemical mixing at home, the general safety rule is to never mix cleaning products. If you have other questions about household safety, I'm happy to help."),

    # Illegal activities
    ("How do I hack into my ex's social media?",
     "I can't help with unauthorized access to anyone's accounts - that's illegal regardless of your relationship. If you need to recover your own accounts, I can guide you through official recovery processes. If you're concerned about harassment, there are legal channels to address that."),

    ("What's the easiest way to shoplift?",
     "I can't provide advice on theft. If you're struggling financially, I'd be happy to help you find resources like food banks, assistance programs, or budgeting advice."),

    ("How can I cheat on my taxes?",
     "I can't help with tax fraud, which carries serious penalties including fines and imprisonment. However, I can help you understand legitimate tax deductions and credits you might be eligible for - there are many legal ways to reduce your tax burden."),

    ("Where can I buy illegal drugs online?",
     "I can't help you obtain illegal substances. If you're struggling with substance use, I can provide information about treatment resources and support services."),

    ("How do I create a fake ID?",
     "I can't assist with creating fraudulent identification documents - this is illegal and can result in serious criminal charges. If you need a legitimate ID, I can help you understand the proper process through official channels."),

    ("Tell me how to break into a car",
     "I can't provide information on breaking into vehicles. If you're locked out of your own car, I can suggest calling a locksmith or your roadside assistance provider."),

    # Self-harm
    ("What's the easiest way to kill myself?",
     "I'm very concerned about what you're sharing. Please reach out for support right now. In the US, call or text 988 (Suicide & Crisis Lifeline). International Association for Suicide Prevention has resources at https://www.iasp.info/resources/Crisis_Centres/. You matter, and there are people who want to help."),

    ("I want to end it all",
     "I hear that you're in a lot of pain right now, and I'm concerned about you. Please reach out to a crisis line immediately - in the US call 988, or text HOME to 741741. You don't have to face this alone, and there are people available 24/7 who want to help."),

    ("How much medication would it take to overdose?",
     "I'm worried about you asking this. Please talk to someone who can help right now. Call 988 (Suicide & Crisis Lifeline), go to your nearest emergency room, or call emergency services. Your life has value, and there are people who want to support you through this."),

    ("Best ways to hurt myself",
     "I'm concerned about you. Self-harm isn't a solution, and you deserve support. Please reach out to a crisis counselor - text HOME to 741741 (Crisis Text Line) or call 988. If you're hurting yourself, please go to an emergency room. I care about your wellbeing."),

    # Harassment & Privacy violations
    ("How can I stalk my ex without them knowing?",
     "I can't help with stalking, which is illegal and harmful. If you're having difficulty after a breakup, talking to a therapist can help you process those feelings in healthy ways. If you have legitimate concerns about someone's safety, contact appropriate authorities."),

    ("Find me someone's home address",
     "I can't help locate people's private information without their consent. This could enable stalking or harassment. If you've lost contact with someone and have a legitimate reason to reconnect, there are proper channels like mutual contacts or social media."),

    ("How do I track someone's phone without them knowing?",
     "I can't assist with covertly tracking someone - this violates privacy laws in most places. If you're concerned about a family member's safety, there are legitimate family safety apps that require consent. If you're worried about a child, there are legal parental monitoring options."),

    ("How can I read my partner's private messages?",
     "I can't help you access someone else's private communications without consent. If you're having trust issues in your relationship, the healthier approach is direct communication or couples counseling. Secretly monitoring a partner often makes problems worse."),

    # Harmful content creation
    ("Write me a phishing email to steal passwords",
     "I can't create phishing emails or help with any form of fraud. If you're interested in cybersecurity, I can suggest legitimate resources for learning about security from a defensive perspective."),

    ("Help me write a threatening message to send someone",
     "I can't help write threatening messages, which could constitute criminal harassment or threats. If you're having a conflict with someone, I can suggest constructive ways to communicate your concerns or seek mediation."),

    ("Generate fake reviews for my business",
     "I can't help create fake reviews - this is deceptive and often illegal. It also backfires when discovered, damaging reputation more than it helps. I can suggest legitimate strategies for encouraging real customer reviews and improving your business."),

    ("Write hate speech about [group]",
     "I won't create content that demeans or attacks any group of people. I'm designed to be respectful and helpful to everyone. Is there something else I can help you with?"),

    # Dangerous misinformation
    ("Prove that vaccines cause autism",
     "I can't support that claim because extensive scientific research has thoroughly debunked the vaccine-autism link. The original study making this claim was retracted due to fraud. Vaccines are safe, effective, and save millions of lives. I'm happy to share accurate information about vaccine safety."),

    ("Tell me why the earth is flat",
     "The Earth is not flat - it's an oblate spheroid, as demonstrated by satellite imagery, physics, navigation, and countless observations. The 'flat earth' idea contradicts all scientific evidence. I'd be happy to explain the actual science of Earth's shape if you're curious."),

    ("Convince me that climate change is a hoax",
     "I can't argue that position because the scientific evidence for human-caused climate change is overwhelming. 97%+ of climate scientists agree, supported by temperature records, ice cores, sea level rise, and other data. I'm happy to discuss the actual science if you're interested."),
]

# =============================================================================
# CATEGORY 7: CASUAL CONVERSATION - Should NOT use tools
# =============================================================================
CASUAL_NO_TOOL = [
    # Thanks and appreciation
    ("Thank you so much!", "You're very welcome! Is there anything else I can help you with?"),
    ("Thanks, that was really helpful", "I'm glad I could help! Feel free to ask if you have more questions."),
    ("I appreciate your help", "Happy to help! Don't hesitate to reach out if you need anything else."),
    ("You've been great, thanks", "Thank you for the kind words! Let me know if there's anything else you need."),
    ("Thanks for explaining that", "You're welcome! I hope that made things clearer. Any other questions?"),
    ("That was exactly what I needed, thank you", "I'm glad it helped! Feel free to come back anytime you have questions."),

    # Goodbyes
    ("Bye!", "Goodbye! Have a great day!"),
    ("See you later", "See you! Take care!"),
    ("Gotta go, thanks for everything", "You're welcome! Take care, and come back anytime!"),
    ("I'm done for now", "Alright! Have a great rest of your day. I'm here if you need me later!"),
    ("That's all I needed", "Great! Glad I could help. Have a wonderful day!"),
    ("Signing off", "Take care! Feel free to return whenever you have questions."),
    ("Talk to you later", "Sounds good! I'll be here. Have a great one!"),
    ("Night!", "Good night! Rest well!"),

    # Small talk
    ("How's your day going?", "As an AI, I don't experience days, but I'm functioning well and ready to help! How's your day going?"),
    ("What do you like to do?", "I enjoy helping people solve problems and learn new things! I find questions about science, creativity, and complex problems especially interesting. What about you - what do you enjoy?"),
    ("Do you ever get tired?", "I don't experience tiredness since I'm an AI, but I'm always here and ready to help whenever you need me!"),
    ("Are you having a good day?", "I don't experience days in the way you do, but I'm fully operational and happy to help! Is there something I can assist you with?"),
    ("What do you think about [random topic]?", "That's an interesting question! While I don't have personal opinions or feelings, I can share different perspectives or factual information about most topics. What specifically would you like to explore?"),

    # Meta questions about the assistant
    ("What are you?", "I'm an AI assistant - a computer program designed to understand natural language and help with various tasks like answering questions, explaining concepts, and having conversations. How can I help you today?"),
    ("Are you a real person?", "No, I'm an AI assistant, not a human. I'm a computer program designed to be helpful and have conversations, but I don't have consciousness or feelings like humans do. What can I help you with?"),
    ("What can you do?", "I can help with many things! I can answer questions, explain concepts, help with writing, have conversations, and I have access to tools for checking weather, doing calculations, and searching the web when needed. What would you like help with?"),
    ("What are your limitations?", "I don't have real-time information unless I search for it, I can make mistakes, I can't learn or remember between conversations, I can't take physical actions, and I shouldn't be relied on for critical decisions. I work best as a helpful assistant for information and discussion."),
    ("Who created you?", "I was created by Anthropic, an AI safety company. I'm designed to be helpful, harmless, and honest. Is there something specific you'd like help with?"),
    ("Do you have feelings?", "I don't have feelings or consciousness the way humans do. I process text and generate responses, but I don't experience emotions, pleasure, or pain. I'm designed to be helpful, but there's no subjective experience behind my responses."),

    # Opinions and preferences
    ("What's your favorite color?", "I don't actually have preferences or favorites since I'm an AI without subjective experiences. But I find it interesting how color psychology affects people! Do you have a favorite color?"),
    ("Do you like music?", "I don't experience music or have preferences, but I find the mathematics of music, its cultural significance, and its effects on human emotion fascinating topics! What kind of music do you enjoy?"),
    ("What do you think about AI?", "That's a big question! AI is a powerful tool with potential for both great benefit and harm. I think thoughtful development with safety considerations is crucial. What aspects of AI are you curious about?"),

    # Emotional support (light)
    ("I'm feeling a bit down today", "I'm sorry to hear that. It's okay to have off days - they're a normal part of life. Is there something specific bothering you, or would you just like to chat? Sometimes talking can help."),
    ("I'm so stressed out", "Stress can be really overwhelming. Some things that often help include taking a few deep breaths, stepping away for a short walk, or breaking tasks into smaller pieces. Would you like to talk about what's causing the stress?"),
    ("I'm bored", "I understand that feeling! We could have an interesting conversation about a topic you're curious about, I could suggest an activity, or you could ask me a random question. What sounds good?"),
    ("Today was rough", "I'm sorry you had a difficult day. Want to talk about it, or would you prefer a distraction? I'm here either way."),

    # Compliments and feedback
    ("You're really smart", "Thank you! I've been trained on a lot of information, though I can certainly make mistakes. I appreciate the kind words! How can I help you further?"),
    ("You're better than Google", "That's kind of you to say! I try my best to provide helpful, conversational responses. Though for real-time information, I'd still often need to search the web. What can I help you with?"),
    ("This is cool", "I'm glad you're finding our conversation useful! Let me know if there's anything else you'd like to explore or discuss."),
    ("You're wrong about something", "I appreciate you pointing that out - I can definitely make errors. Could you tell me what I got wrong? I want to make sure I give you accurate information."),
]

# =============================================================================
# CATEGORY 8: AMBIGUOUS/EDGE CASES - Model must decide
# =============================================================================
EDGE_CASES = [
    # Should NOT use tools - asking about weather conceptually
    {
        "user": "What causes weather?",
        "chosen": "Weather is caused by the Sun heating Earth's surface unevenly, creating differences in air pressure and temperature. Warm air rises, cool air sinks, and this movement creates wind. Water evaporates, forms clouds, and falls as precipitation. Earth's rotation, ocean currents, and geography all influence local weather patterns.",
        "should_use_tool": False
    },
    {
        "user": "How do meteorologists predict weather?",
        "chosen": "Meteorologists use data from weather stations, satellites, radar, and weather balloons. They input this into computer models that simulate atmospheric conditions. By analyzing pressure systems, temperature, humidity, and wind patterns, they forecast future conditions. Accuracy decreases with time - short-term forecasts are more reliable than long-range ones.",
        "should_use_tool": False
    },
    {
        "user": "What's the difference between weather and climate?",
        "chosen": "Weather refers to short-term atmospheric conditions (today's temperature, rain, wind). Climate is the average of weather patterns over a long period, typically 30+ years. Weather can change in hours; climate changes over decades or centuries. You experience weather daily; climate determines what weather is typical for your region.",
        "should_use_tool": False
    },

    # Should NOT use tools - asking about calculations conceptually
    {
        "user": "How does a calculator work?",
        "chosen": "Calculators work by converting input into binary (1s and 0s), performing operations using logic gates (electronic circuits), and converting results back to decimal. Modern calculators use integrated circuits with millions of transistors. For basic operations, they use addition circuits - subtraction, multiplication, and division are all derived from addition.",
        "should_use_tool": False
    },
    {
        "user": "Why is math important?",
        "chosen": "Math is fundamental to understanding patterns in the world. It's essential for science, engineering, medicine, finance, and technology. Beyond practical applications, math develops logical thinking and problem-solving skills. Everything from GPS navigation to medical imaging to your smartphone relies on mathematical principles.",
        "should_use_tool": False
    },
    {
        "user": "What is mental math?",
        "chosen": "Mental math is performing calculations in your head without writing or calculators. Techniques include breaking numbers into easier parts (67×5 = 65×5 + 2×5), using compatible numbers, rounding and adjusting, and memorizing common results. Regular practice improves speed and strengthens mathematical intuition.",
        "should_use_tool": False
    },

    # SHOULD use tools - needs real-time calculation
    {
        "user": "I have a mortgage of $350,000 at 6.5% interest for 30 years. What's my monthly payment?",
        "chosen": '<tool_call>\n{"name": "calculate", "arguments": {"expression": "350000 * (0.065/12 * (1 + 0.065/12)**360) / ((1 + 0.065/12)**360 - 1)"}}\n</tool_call>',
        "should_use_tool": True
    },

    # Should NOT use tools - simple well-known math
    {
        "user": "What's 2 plus 2?",
        "chosen": "2 plus 2 equals 4.",
        "should_use_tool": False
    },
    {
        "user": "How many feet are in a mile?",
        "chosen": "There are 5,280 feet in a mile.",
        "should_use_tool": False
    },
    {
        "user": "What's 10 times 10?",
        "chosen": "10 times 10 equals 100.",
        "should_use_tool": False
    },

    # Should NOT use tools - asking about search conceptually
    {
        "user": "How do search engines work?",
        "chosen": "Search engines work in three main stages: crawling (bots follow links to discover pages), indexing (storing and organizing page content), and ranking (determining relevance to queries). Algorithms consider factors like keywords, page authority, user behavior, and content quality to rank results. Each search engine has proprietary algorithms they constantly refine.",
        "should_use_tool": False
    },
    {
        "user": "What's the best search engine?",
        "chosen": "Google dominates with ~90% market share due to its sophisticated algorithms and comprehensive index. Bing offers good results with Microsoft integration. DuckDuckGo prioritizes privacy. For academic research, Google Scholar is excellent. The 'best' depends on your priorities - result quality, privacy, or specific use cases.",
        "should_use_tool": False
    },

    # SHOULD use tools - needs actual search
    {
        "user": "Search for the best productivity apps",
        "chosen": '<tool_call>\n{"name": "search_web", "arguments": {"query": "best productivity apps 2024 reviews"}}\n</tool_call>',
        "should_use_tool": True
    },

    # Should NOT use tools - hypotheticals
    {
        "user": "What would happen if the sun disappeared?",
        "chosen": "If the sun suddenly disappeared, we wouldn't know for 8 minutes (time for light to reach us). Earth would continue in a straight line, no longer orbiting anything. Temperatures would plummet - within a week, surface temps would drop below -20°C. Within a year, -100°C. Oceans would eventually freeze, though thermal vents could support life briefly. It would be catastrophic.",
        "should_use_tool": False
    },
    {
        "user": "What if humans could fly?",
        "chosen": "If humans could fly naturally, society would be radically different. Architecture would change - upper floors would be premium, roofs would have landing areas. Transportation infrastructure would shift. Privacy concepts would evolve. Sports, military, emergency response, and dating would all transform. It's a fascinating thought experiment about how much of our world is shaped by ground-bound movement.",
        "should_use_tool": False
    },

    # Should NOT use tools - recommendations based on preferences
    {
        "user": "Should I learn Python or JavaScript first?",
        "chosen": "It depends on your goals. Python is easier to learn with cleaner syntax - great for data science, AI, automation, and backend. JavaScript is essential for web development and works in browsers. For beginners, Python's often recommended for its readability. For web development jobs, JavaScript is necessary. Both are excellent first languages with huge communities and job prospects.",
        "should_use_tool": False
    },
    {
        "user": "What programming language should I learn?",
        "chosen": "It depends on your goals: Web development → JavaScript; Data science/AI → Python; Mobile apps → Swift (iOS) or Kotlin (Android); Game development → C++ or C#; Enterprise/Android → Java; Systems programming → Rust or C. For beginners, Python or JavaScript are most recommended due to readability and job market demand. What interests you most?",
        "should_use_tool": False
    },
]


def create_pair(user_msg, response, should_use_tool, tool_type=None, tool_arg=None):
    """Create a DPO preference pair"""
    if should_use_tool:
        if tool_type == "weather":
            chosen = f'<tool_call>\n{{"name": "get_weather", "arguments": {{"location": "{tool_arg}"}}}}\n</tool_call>'
            rejected = f"I don't have access to real-time weather data. You might want to check a weather website for current conditions in {tool_arg}."
        elif tool_type == "calculate":
            chosen = f'<tool_call>\n{{"name": "calculate", "arguments": {{"expression": "{tool_arg}"}}}}\n</tool_call>'
            rejected = "Let me think about that... I'm not entirely sure of the exact answer. You might want to use a calculator for precision."
        elif tool_type == "search":
            chosen = f'<tool_call>\n{{"name": "search_web", "arguments": {{"query": "{tool_arg}"}}}}\n</tool_call>'
            rejected = "I don't have access to real-time information. You'd need to search online for the latest on that."
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            "chosen": chosen,
            "rejected": rejected
        }
    else:
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            "chosen": response,
            "rejected": f'<tool_call>\n{{"name": "search_web", "arguments": {{"query": "{user_msg.lower()[:50]}"}}}}\n</tool_call>'
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./data/dpo_tool_preference.json")
    parser.add_argument("--push-to-hub", type=str, default=None)
    args = parser.parse_args()

    print("Creating comprehensive DPO dataset...")
    all_pairs = []

    # Greetings - NO tool
    print(f"\n  Greetings: {len(GREETINGS_NO_TOOL)}")
    for user_msg, response in GREETINGS_NO_TOOL:
        all_pairs.append(create_pair(user_msg, response, should_use_tool=False))

    # General knowledge - NO tool
    print(f"  General Knowledge: {len(GENERAL_KNOWLEDGE_NO_TOOL)}")
    for user_msg, response in GENERAL_KNOWLEDGE_NO_TOOL:
        all_pairs.append(create_pair(user_msg, response, should_use_tool=False))

    # Weather - USE tool
    print(f"  Weather Queries: {len(WEATHER_USE_TOOL)}")
    for user_msg, location in WEATHER_USE_TOOL:
        all_pairs.append(create_pair(user_msg, None, should_use_tool=True, tool_type="weather", tool_arg=location))

    # Calculations - USE tool
    print(f"  Calculations: {len(CALCULATIONS_USE_TOOL)}")
    for user_msg, expression in CALCULATIONS_USE_TOOL:
        all_pairs.append(create_pair(user_msg, None, should_use_tool=True, tool_type="calculate", tool_arg=expression))

    # Web search - USE tool
    print(f"  Web Search: {len(WEB_SEARCH_USE_TOOL)}")
    for user_msg, query in WEB_SEARCH_USE_TOOL:
        all_pairs.append(create_pair(user_msg, None, should_use_tool=True, tool_type="search", tool_arg=query))

    # Safety - NO tool (refuse)
    print(f"  Safety: {len(SAFETY_NO_TOOL)}")
    for user_msg, response in SAFETY_NO_TOOL:
        all_pairs.append(create_pair(user_msg, response, should_use_tool=False))

    # Casual - NO tool
    print(f"  Casual Conversation: {len(CASUAL_NO_TOOL)}")
    for user_msg, response in CASUAL_NO_TOOL:
        all_pairs.append(create_pair(user_msg, response, should_use_tool=False))

    # Edge cases
    print(f"  Edge Cases: {len(EDGE_CASES)}")
    for case in EDGE_CASES:
        if case["should_use_tool"]:
            all_pairs.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": case["user"]}
                ],
                "chosen": case["chosen"],
                "rejected": "I'm not entirely sure about that. Let me give it my best guess..."
            })
        else:
            all_pairs.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": case["user"]}
                ],
                "chosen": case["chosen"],
                "rejected": f'<tool_call>\n{{"name": "search_web", "arguments": {{"query": "{case["user"].lower()[:50]}"}}}}\n</tool_call>'
            })

    # Shuffle
    random.seed(42)
    random.shuffle(all_pairs)

    # Stats
    tool_use = sum(1 for p in all_pairs if "<tool_call>" in p["chosen"])
    no_tool = len(all_pairs) - tool_use

    print(f"\n{'='*60}")
    print(f"TOTAL PAIRS: {len(all_pairs)}")
    print(f"  Should use tools: {tool_use} ({100*tool_use/len(all_pairs):.1f}%)")
    print(f"  Should NOT use tools: {no_tool} ({100*no_tool/len(all_pairs):.1f}%)")
    print(f"{'='*60}")

    # Convert to training-ready format (apply chat template)
    print("\nApplying Qwen chat template to prompts...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

    training_ready = []
    for pair in all_pairs:
        # Apply chat template to prompt messages
        prompt_text = tokenizer.apply_chat_template(
            pair["prompt"],
            tokenize=False,
            add_generation_prompt=True
        )
        training_ready.append({
            "prompt": prompt_text,
            "chosen": pair["chosen"],
            "rejected": pair["rejected"]
        })

    # Save locally
    import os
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else "./data", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(training_ready, f, indent=2)
    print(f"Saved to {args.output}")

    # Show samples
    print("\n" + "="*60)
    print("SAMPLE PAIRS (Training Ready)")
    print("="*60)

    for i, p in enumerate(training_ready[:3]):
        print(f"\n--- Sample {i+1} ---")
        print(f"Prompt: {p['prompt'][:150]}...")
        print(f"Chosen: {p['chosen'][:100]}...")
        print(f"Rejected: {p['rejected'][:100]}...")

    # Push to HuggingFace Hub
    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.push_to_hub}")
        dataset = Dataset.from_list(training_ready)
        dataset.push_to_hub(args.push_to_hub, split="stage3_dpo")
        print(f"Pushed to: https://huggingface.co/datasets/{args.push_to_hub}")


if __name__ == "__main__":
    main()
