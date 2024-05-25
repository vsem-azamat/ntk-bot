from openai import OpenAI
from config import cnfg


client = OpenAI(api_key=cnfg.OPENAI_API_KEY)


async def get_gpt_response(message: str) -> str | None:
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
        "role": "system",
        "content": cnfg.INSTRUCTIONS
        },
        {
        "role": "user",
        "content": [
            {
            "text": message,
            "type": "text"
            }
        ]
        }
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response.choices[0].message.content

