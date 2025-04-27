from openai import OpenAI
import re
import numpy as np
from LLM.local_LLM import get_model
import json
import importlib
import LLM.server_LLM_pool as model_server

#client = OpenAI()
model_launched = False
class Chain_of_Thought():
  def __init__(self, config):
    # Initialize client
    #self.client = OpenAI()
    global model_launched
    if not model_launched:
      model_server.start_server()
      model_server.wait_for_server()
      model_launched = True
    self.client = model_server.query_model

    self.system_message = config.system_message
    self.tool_schemas = config.tool_schemas
    self.functions_module = importlib.import_module(config.module_name)
    # Initialize messages for chat
    self.cot_prompts = config.cot_prompts
    self.messages = []

  def __call__(self):
    """ 
    Perform Chain of Thought Reasoning.
    Inputs:  None, but the completion will come from current messages
    outputs: An unformatted action selection
    """

    # Assume system_message has already been appended in interaction...
    # Iterate through chain of thought
    for cot_prompt in self.cot_prompts[:-1]: 
      # cot_prompt format: {"role": "user", "content": "chain of thought string"}
      self.messages.append(cot_prompt)
      # Prompt the client based on the current set of messages
      #completion = self._query()
      completion = self._call_with_tools()
      self.messages.append(completion)
    
    self.messages.append(self.cot_prompts[-1])
    # WE DO NOT USE CHECK ACTION SELECTION. THIS WAS PREVIOUSLY USED TO VERIFY OUTPUT.
    # self.messages NOW INCLUDES THE FULL CHAIN OF THOUGHT PROCESS FOR THE AGENT.

    # action = self._check_action_selection(self.messages[-1]["content"])

    # NEW FOR TOOL CALLING: HAVE THE AGENT CALL THE TOOL
    tool_call = self._call_with_tools(self.tool_schemas)

    print(f" last message: {self.messages[-1]}")
    print(tool_call)

    fn_name = tool_call[0].function.name
    fn_args = json.loads(tool_call[0].function.arguments)

    print(f" last message: {self.messages[-1]}")
    print(tool_call)

    # Run the function
    method = getattr(self.functions_module, fn_name, None)
    if method and callable(method):
      try:
        action = method(**fn_args)
      except Exception as e:
        print(f"Error: {e}")
    else:
      print(f"Error: function not callable")      


    # Clear messages from training step
    self.messages = []
    return action

  def _query(self):
    # JUST USE _call_with_tools WITH OR WITHOUT TOOLS
    #completion = self.client.chat.completions.create(
    #  model="gpt-4o",
    #  messages = self.messages
    #)
    
    completion = self.client(self.messages)
    #return completion
    return {"role": "assistant", "content": completion[0].outputs[0].text}
  
  def _call_with_tools(self, tools=None):
    kwargs = {
        "model": "deepseek-ai/deepseek-llm-7b-chat",
        "messages": self.messages,
    }
    if tools is not None:
        kwargs["tools"] = self.tool_schemas
    try:
        #completion = self.client.chat.completions.create(**kwargs)
        #return completion.choices[0].message.tool_calls
        completion = self.client(**kwargs)
        print(completion)
        if tools is None:
            return {"role": "assistant", "content": completion[0].outputs[0].text}
        return {"role": "assistant",
                "tool_calls": completion[0].choices[0].message.tool_calls}

    except Exception as e:
        print(f"Exception: {e}")
     

 # WE DON'T USE THIS ANYMORE

  # def _check_action_selection(self, action_selection):
  #     # Normalize input
  #     action = action_selection.strip().lower()

  #     # Define action mappings
  #     action_map = {
  #         "0": 0, "nothing": 0, "no action": 0, "idle": 0,
  #         "1": 1, "left": 1, "fire left": 1, "fire left engine": 1, "rotate right": 1,
  #         "2": 2, "main": 2, "fire main": 2, "fire main engine": 2, "thrust": 2,
  #         "3": 3, "right": 3, "fire right": 3, "fire right engine": 3, "rotate left": 3
  #     }

  #     # Check for direct mappings
  #     if action in action_map:
  #         return action_map[action]

  #     # Try to extract numbers from input (e.g., "press 2" → 2)
  #     match = re.search(r"\b[0-3]\b", action)
  #     if match:
  #         return int(match.group(0))

  #     # Handle invalid input - return random output for now
  #     print(f"Unrecognized action: {action}")
  #     return np.random.randint(0, 3)
