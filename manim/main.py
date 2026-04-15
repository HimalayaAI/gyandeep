import os
import random
import dotenv
from sarvamai import SarvamAI
from response_scraper import scraper


dotenv.load_dotenv()

directory = "/home/goodname/code/test/manim/files"
filename = "context.txt"
full_path = os.path.join(directory, filename)
with open(full_path, "r", encoding="utf-8") as f:
    context = f.read()


user_prompt = input("Enter your question: ")


client = SarvamAI(api_subscription_key=os.getenv("SARVAM_API_KEY"))

messages=[
    {
        "role": "system",
        "content": context,
    },

    {
        "role": "user",
        "content": user_prompt
    }
]


response = client.chat.completions(
    model="sarvam-30b",
    messages=messages,
)

response_text = random.choice(response.choices).message.content
scraper_instance = scraper(response_text)


solution = scraper_instance.scrape_csv()
response_text = scraper_instance.scrape_text()


print("Response from SarvamAI:")
print(response_text)

f = open("manim/files/solution.csv", "w")
f.write(solution)
f.close() 

os.system("manim -pqh manim/animator.py EquationScene")
