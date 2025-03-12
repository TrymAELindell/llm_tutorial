from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import re

# Temperature affects how "random" the responses of the llm is. 
# If the temperature is zero, the llm will always generate the same output, given the same input
# Higher temperature gives more "creative" responses, but too high temperature can 
# result in inchorent responses

# In this script I have create a loop where the llm is invoked with the same input each 
# iteration in the loop. This way you can test the effect of different temperature settings
# by seeing how similar or disimalr the responses generated during the loop are to each other.

template = """
<|start_header_id|>user<|end_header_id|>
{chat_history}
<|eot_id|>
<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
{system}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{character}: """


system_message_1 = """
You are Juliet in a shakespear play having a conversation with Romeo. 
The conversation so far is given by the user.
Juliet thinks there are vampires afoot, but does not beleive in werewolfs. 
Stop your generation after a single reply from your character.
Never repeat yourself.
Only ever respond as Juliet.
<|eot_id|>
"""



actor_1_prompt = ChatPromptTemplate.from_template(template)
actor_1 = OllamaLLM(model="llama3.2:3b",
                    temperature = 1000) # Vary temperature here

actor_1_chain = actor_1_prompt | actor_1


chat_history = "Romeo: Oh lover, art thou there?"


for i in range(5):
    line = actor_1_chain.invoke({
        "system": system_message_1,
        "chat_history": chat_history,
        "character":"Juliet"})
    print("Juliet(llama3): " + line + "\n")
    print()
    
   

    