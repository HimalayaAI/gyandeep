import asyncio

from config import SARVAMAI_KEY, API_KEY_PLACEHOLDER
from sarvamai import SarvamAI

async def test():
    if not SARVAMAI_KEY or SARVAMAI_KEY == API_KEY_PLACEHOLDER:
        print("API Key not found or empty.")
        return

    client = SarvamAI(api_subscription_key=SARVAMAI_KEY)
    try:
        response = await asyncio.to_thread(
            client.chat.completions,
            model="sarvam-2B-chat",
            messages=[{'role': 'user', 'content': 'Hello!'}]
        )
        print("Success:", response)
    except Exception as e:
        print("Error with sarvam-2B-chat:", e)
        
    try:
        response = await asyncio.to_thread(
            client.chat.completions,
            model="sarvam-30b",
            messages=[{'role': 'user', 'content': 'Hello!'}]
        )
        print("Success:", response)
    except Exception as e:
        print("Error with sarvam-30b:", e)

if __name__ == "__main__":
    asyncio.run(test())
