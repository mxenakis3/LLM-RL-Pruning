from openai import OpenAI
import re
import numpy as np
from LLM.local_LLM import get_model

#client = OpenAI()

class Chain_of_Thought():
  def __init__(self, config):
    # Initialize client
    #self.client = OpenAI()
    self.client = get_model()

    self.system_message = config.system_message

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
    for cot_prompt in self.cot_prompts: 

      # cot_prompt format: {"role": "user", "content": "chain of thought string"}
      self.messages.append(cot_prompt)
      # Prompt the client based on the current set of messages
      completion = self._execute()
      self.messages.append(completion)
    
    # Input the last action chosen
    #action = self._check_action_selection(self.messages[-1].choices[0].message.content)
    action = self._check_action_selection(self.messages[-1]["content"])
  
    # Clear messages from training step
    self.messages = []
    return action

  def _execute(self):
    #completion = self.client.chat.completions.create(
    #  model="gpt-4o",
    #  messages = self.messages
    #)
    completion = self.client(self.messages)
    #return completion
    return {"role": "assistant", "content": completion[0].outputs[0].text}

  def _check_action_selection(self, action_selection):
      # Normalize input
      action = action_selection.strip().lower()

      # Define action mappings
      action_map = {
          "0": 0, "nothing": 0, "no action": 0, "idle": 0,
          "1": 1, "left": 1, "fire left": 1, "fire left engine": 1, "rotate right": 1,
          "2": 2, "main": 2, "fire main": 2, "fire main engine": 2, "thrust": 2,
          "3": 3, "right": 3, "fire right": 3, "fire right engine": 3, "rotate left": 3
      }

      # Check for direct mappings
      if action in action_map:
          return action_map[action]

      # Try to extract numbers from input (e.g., "press 2" → 2)
      match = re.search(r"\b[0-3]\b", action)
      if match:
          return int(match.group(0))

      # Handle invalid input - return random output for now
      print(f"Unrecognized action: {action}")
      return np.random.randint(0, 3)
