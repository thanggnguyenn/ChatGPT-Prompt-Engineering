"""This sample demonstrates a basic call to the chat completion API.
It is leveraging your endpoint and key. The call is synchronous."""

import os
from openai import OpenAI

token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.inference.ai.azure.com"

# Pick one of the Azure OpenAI models from the GitHub Models service
model_name = "gpt-4o-mini"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

def get_completion(prompt):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_name,
        # Optional parameters
        temperature=1.,
        max_tokens=1000,
        top_p=1.    
    )
    return response.choices[0].message.content

# ---------- Summarization ----------
text = f"""
You should express what you want a model to do by \ 
providing instructions that are as clear and \ 
specific as you can possibly make them. \ 
This will guide the model towards the desired output, \ 
and reduce the chances of receiving irrelevant \ 
or incorrect responses. Don't confuse writing a \ 
clear prompt with writing a short prompt. \ 
In many cases, longer prompts provide more clarity \ 
and context for the model, which can lead to \ 
more detailed and relevant outputs.
"""

prompt = f"""
Summarize the text delimited by triple backticks \ 
into a single sentence.
```{text}```
"""

# ---------- Structured response ----------
prompt_structure = f"""
Generate a list of 5 song titles by Taylor Swift along \ 
with their publish year and genres. 
Provide them in HTML format with the following keys: 
book_id, title, author, genre.
"""
# ---------- Conditions in prompt (return a sequence of instructions) ----------
text_1 = f"""
Making a cup of tea is easy! First, you need to get some \ 
water boiling. While that's happening, \ 
grab a cup and put a tea bag in it. Once the water is \ 
hot enough, just pour it over the tea bag. \ 
Let it sit for a bit so the tea can steep. After a \ 
few minutes, take out the tea bag. If you \ 
like, you can add some sugar or milk to taste. \ 
And that's it! You've got yourself a delicious \ 
cup of tea to enjoy.
"""

text_2 = f"""
The sun is shining brightly today, and the birds are \
singing. It's a beautiful day to go for a \ 
walk in the park. The flowers are blooming, and the \ 
trees are swaying gently in the breeze. People \ 
are out and about, enjoying the lovely weather. \ 
Some are having picnics, while others are playing \ 
games or simply relaxing on the grass. It's a \ 
perfect day to spend time outdoors and appreciate the \ 
beauty of nature.
"""

prompt_condition = f"""
You will be provided with text delimited by triple quotes. 
If it contains a sequence of instructions, \ 
re-write those instructions in the following format:

Step 1 - ...
Step 2 - …
…
Step N - …

If the text does not contain a sequence of instructions, \ 
then simply write \"No steps provided.\"

\"\"\"{text_2}\"\"\"
"""
# response = get_completion(prompt_condition)
# print("Completion for Text 2:")
# print(response)

# ---------- Few-shot prompting ----------
prompt_fewshot = f"""
Your task is to answer in a consistent style.

<child>: Teach me about patience.

<grandparent>: The river that carves the deepest \ 
valley flows from a modest spring; the \ 
grandest symphony originates from a single note; \ 
the most intricate tapestry begins with a solitary thread.

<child>: Teach me about resilience.
"""
# response = get_completion(prompt_fewshot)
# print(response)

# ---------- Specify steps required to complete a task ----------
text = f"""
Gold prices jumped to record high and the dollar was on the rise again on Wednesday, keeping the pressure on the yen and the euro,\ 
while stocks in Asia stuttered as investors were reluctant to place major bets ahead of a hotly contested U.S. election. \  
The shifting expectations around how fast and deep the Federal Reserve will cut rates have also hurt risk sentiment, \  
with traders now anticipating the U.S. central bank to be measured in its easing. \ 
That has taken U.S. Treasury yields to a three-month peak and the dollar to multi-month highs against the euro, sterling and the yen, \ 
which is now back at 150 per dollar levels, prompting verbal warnings from Japanese officials.
MSCI's broadest index of Asia-Pacific shares outside Japan (.MIAPJ0000PUS), opens new tab was last 0.06% higher. \  
Tokyo's Nikkei (.N225), opens new tab was slightly lower in early trading.\ 
"Volatility within a range bound trade is increasingly becoming the norm, as markets brace for pivotal weeks ahead, \   
including the U.S. presidential election and a heavy corporate earnings agenda," said Anderson Alves, a trader with ActivTrades. \                                  
China and Hong Kong stocks made a steady open of trade on Wednesday, as the promise of government help for the economy supported the major indexes to settle in at higher levels. \ 
Shifting momentum towards a likely Donald Trump presidency has been in focus for investors, \   
with Trump policies including tariffs and restrictions on undocumented immigration expected to increase inflation. \    
That in turn has supported the dollar on expectations U.S. rates may remain relatively high for a longer-than-anticipated period.
"""
# example 1
prompt_1 = f"""
Perform the following actions: 
1 - Summarize the following text delimited by triple \
backticks with 1 sentence.
2 - List all the numbers in the text and their meanings.
3 - List each name in the summary.
4 - Output a json object that contains the following \
keys: names, numbers.

Separate your answers with line breaks.

Text:
```{text}```
"""

prompt_2 = f"""
Your task is to perform the following actions: 
1 - Summarize the following text delimited by 
  <> with 1 sentence.
2 - List all the numbers in the text and their meanings.
3 - List each name in the summary.
4 - Output a json object that contains the 
  following keys: names, numbers.

Use the following format:
Text: <text to summarize>
Summary: <summary>
Numbers summary: <numbers summary>
Names: <list of names in summary>
Output JSON: <json with names and numbers>

Text: <{text}>
"""


# response = get_completion(prompt_2)
# print("\nCompletion for prompt 2:")
# print(response)

# response = get_completion(prompt_1)
# print("Completion for prompt 1:")
# print(response)

# --------------- Instruct the model to work on its own solution before rushing to a conclusion -------------
prompt = f"""
Determine if the student's solution is correct or not.
If the student's solution is not correct, show the correct solution in the format that has no punctation.

Question:
I'm building a solar power installation and I need \
 help working out the financials. 
- Land costs $100 / square foot
- I can buy solar panels for $250 / square foot
- I negotiated a contract for maintenance that will cost \ 
me a flat $100k per year, and an additional $10 / square \
foot
What is the total cost for the first year of operations 
as a function of the number of square feet.

Student's Solution:
Let x be the size of the installation in square feet.
Costs:
1. Land cost: 100x
2. Solar panel cost: 250x
3. Maintenance cost: 100,000 + 100x
Total cost: 100x + 250x + 100,000 + 100x = 450x + 100,000
"""

response = get_completion(prompt)
print(response)