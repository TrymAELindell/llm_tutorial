from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

################# some tutorials to read ###################################
# LangChain on local LLMs
# https://python.langchain.com/docs/how_to/local_llms/
# Note in particular the section on quantization. 
# Ollama models can be downloaded with different parameter numbers and quantization
# Larger paramter number (xb) = more GB, larger quantization (q/fp) = more GB

# Basic get up and running with Ollama in LangChain
# https://python.langchain.com/docs/integrations/llms/ollama/

# langchain_ollama.llms.OllamaLM API reference
# https://python.langchain.com/api_reference/ollama/llms/langchain_ollama.llms.OllamaLLM.html#langchain_ollama.llms.OllamaLLM
# Mirostat paper: https://openreview.net/pdf?id=W1G1JZEIy5_

# llama3.2 on Ollama
# https://ollama.com/library/llama3.2
# For formatting of prompt check template: https://ollama.com/library/llama3.2/blobs/966de95ca8a6
# For tags check: https://ollama.com/library/llama3.2/tags 


##################################### setting up the llms ##################################################
## prompt templates
# LLMs have a limited context window. The context window is the maximum amount of tokens (sort of like words)
# that an llm can use as input. As a conversation becomes longer and longer this means that the beginning
# of the conversation will at some point drop off the edge of the context window. When this happens the llm
# will no longer remember this part of the conversation. This becomes important when you create a prompt template.

# A prompt template is a template which you can fill in with information that will shape the response of the llms.
# Each model will often have it's own template format (you can look at the default template in ollama for each model)
# Since this is often used during training the llm may perform better if you use the template format it was trained on.
# This can for example involve header naming, end of message and the specific names for the headers.
# Some special names that are often used are
# System - this is generally the instruction for the model concerning how it should act and what it's role is
# This can for example be: you are a helpfull ai. Only respond with short sentences and use positive language
# user - this would be were the user prompt would go
# assistant - This would be where the response from the llm would start

# Since the system message provides the instructions for how the llm should behave it's important that it 
# always stays in the context window. Therefore it should be placed after the chat history so that it always stays
# at the end of the template string and thus stays in the context window.

# The template below does this for the llama3.2 model 

llama_template = """
<|start_header_id|>user<|end_header_id|>
{chat_history}
<|eot_id|>
<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
{system}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{character}: """

# The template will use a dictionary to fill in the location where 
# your see keywords in curlybrackets in the template above.
# The below system message would for example replace {system} in the template
llama_system_message = """
You are Juliet in a shakespear play having a conversation with Romeo. 
The conversation so far is given by the user.
Juliet thinks there are vampires afoot, but does not beleive in werewolfs. 
Stop your generation after a single reply from your character.
Only give long responses
Never repeat yourself.
Only ever respond as Juliet.
<|eot_id|>
"""

# Template itself is created using the string template text and ChatPromptTemplate class from langchain
llama_prompt_template = ChatPromptTemplate.from_template(llama_template)


## Instantiating the model
# The model itself is instantited using OllamaLLM
# To do this you must have Ollama installed and have fetched the model you wish to use.
# This is done in the terminal/command line using: ollama fetch name_of_model
# The name of the model can have many different tags. The first bit is just the model name
# ex: llama3.2 
# This is followed by :xb where x is replaced by some number. This number indicated how many parameters
# or the model has. This will mostly be the weights in the model. The more parameters the better the model
# however it will require more RAM and be slower the more parameters it has
# The model can also have additional tags. The quantization level is one of the more important ones.
# Quantazation means that the model weight are converted to a more efficient numeric format. 
# Usually llms are trained with float32 weights, but this can be reduced to more efficent types, like float16 or even smaller
# Q4 for example means that the weights are stored as 4bit numbers (I think)

# The OllamaLLM class also has a range of parameters that can effect the model's performance.
# To read more about this check the documentation linked at the start of this script.
# Here I have only set the temperature parameter which controls how random the model will respond
# temperature = 0 means it is determinstic, while larger and larger numbers will give more and more inchoerent responses
# See the testing_the_effect_of_temperature.py script for more details
llama_actor = OllamaLLM(model="llama3.2:3b",
                    temperature = 8)

# To ensure that the input to the model is passed through the template before it reaches the model
# The template and the model is chained using the below code
llama_actor_chain = llama_prompt_template | llama_actor

## Actor 2
# For fun I'll have two different models play the different characters so that 
# we can see if they behave differently to each other. Note that you don't actually 
# need two different llms to create something like a dialogue. You can just as well
# Have a single llm produce the full dialogue. However, telling an llm to do too many things
# can hamper performance. An alternative that still only uses one llm is to create 
# different system messages and use them to represent the different "actors"

# Here I repeat the same steps as for the llama model to instantiate qwen
# Note that the template for qwen Is slightly different than llamas
qwen_template = """

{chat_history}

<|im_start|>system
{system}
<|im_end|>
<|im_start|>assistant
{character}: """


qwen_system_message = """
You are Romeo in a shakespear play having a conversation with Juliet. 
Romeo is afraid there are werewolfs afoot, but does not beleive in vampires. 
The conversation so far is given by the above.
Stop your generation after a single reply from your character.
Only respond with short responses. 
Never repeat yourself.
Only ever respond as Romeo.
"""


qwen_prompt_template = ChatPromptTemplate.from_template(qwen_template)
qwen_actor = OllamaLLM(model="qwen2.5:3b")

qwen_actor_chain = qwen_prompt_template | qwen_actor


######## Create the dialogue #########################################
# I start the chat history with some content. This way I can control where 
# the scene start. Otherwise the two bots would just start on some random
# story that takes into account the system prompts.

chat_history = "Romeo: Oh lover, art thou there?"
print(chat_history)

for i in range(20): # There will be 20 back and forth lines between the models
    
    # Getting a response from the model is simply done using the .invoke method
    # The input to this method should be a dictionary with key value pairs corresponding
    # to the template 
    line = llama_actor_chain.invoke({
        "system": llama_system_message,
        "chat_history": chat_history,
        "character":"Juliet"})
    
    print("Juliet(llama3): " + line + "\n")

    # To progress the story we need to update the chat history with the output from the llm
    # Here I have simply done this by appending the output from the llm to the chat history string
    chat_history = chat_history + "\n Juliet: " + line
   
    # Since the response from llama model now was added to the chat_history it will be part of the input
    # to the qwen model. The qwen model will thus respond to what the llama model said
    line = qwen_actor_chain.invoke({
        "system": qwen_system_message,
        "chat_history": chat_history,
        "character":"Romeo"})
    chat_history = chat_history + "\n Romeo: " + line
    print("Romeo(qwen2.5): " + line + "\n")
