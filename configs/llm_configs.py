from configs.config_class import Config

lunar_llm_agent_configs = Config(
  {
    # Configs will go here
    "system_message" : {"role": "system", "content": """
      You are an agent playing the LunarLander reinforcement learning game. 

      Your goal is to safely land the aircraft on the ground below. 

      You will receive an observation of the current state:
      x position: {x_pos}
      y position: {y_pos}
      x velocity {x_vel}
      y velocity: {y_vel}
      angle: {angle} 
      angular velocity: {angular_vel}
      (Boolean) 1 if left leg is in contact with the ground, otherise 0: {left_contact}
      (Boolean) 1 if right leg is in contact with the ground, otherise 0: {right_contact}
                        
      The x positions range from around (-1, 1), where -1 is the leftmost point and 1 is the rightmost point.
      The y_positions range from around (0, ~1.4), where 0 is the ground.
                        
      x-velocity ranges from approximately -1.5 to 1.5. Negative values indicate movement to the left, and positive values indicate movement to the right
      y-velocity ranges from approximately -2.0 to 2.0. Negative values indicate downward movement (falling), and positive values indicate upward movement (thrust applied)
                        
      Angle typically ranges from about -π to π (or approximately -3.14 to 3.14 radians). Zero means the lander is upright, negative values indicate tilting to the right, and positive values indicate tilting to the left.
      Angular velocity typically ranges from about -5 to 5 radians per second. Negative values indicate clockwise rotation, and positive values indicate counterclockwise rotation.

      LunarLander-v2 will crash even if it is in the correct orientation if it is traveling too fast.


      The actions available to you are as follows:
                        
      0: do nothing
      1: fire left orientation engine
      2: fire main engine
      3: fire right orientation engine              

    """},

    "cot_prompts": [
      {"role": "user", "content": "Are you near the ground or high above the ground?",
       "role": "user", "content": "Are you travelling at a safe speed in the x and y direction?",
       "role": "user", "content": "If you are not travelling at a safe speed in the x and y direction, which of the actions available could you take at this step to travel at a safer speed?",
       "role": "user", "content": "Are you orientated in a safe landing angle for landing?, which of the actions available could you take at this step to travel at a safer speed?",
       "role": "user", "content": "If you are not at a safe landing angle, which of the actions available could you take to adjust your orientation?",
       "role": "user", "content": "based on the above conversation, respond with the index corresponding to the action you want to take: \n 0: do nothing \n 1: fire left orientation engine \n 2: fire main engine \n 3: fire right orientation engine"
      }
    ]

    

  }
)

texas_holdem_llm_agent_configs = Config(
  {
    # Configs will go here
    "system_message" : {"role": "system", "content": """
      You are an agent playing Texas Hold Em, against one opponent., The state of the game is as follows:
      {state}

    Your legal actions are {legal_actions}
    """},

    "cot_prompts": [
      {"role": "user", "content": "Evaluate the strength of your private hand and the community cards. Do you currently have a strong hand, a drawing hand, or a weak hand?"},
      {"role": "user", "content": "Based on the number of chips raised this round, is the opponent showing strength, weakness, or uncertainty?"},
      {"role": "user", "content": "Given your hand strength and opponent's last action, is it better to fold, call/check, or raise one chip?"},
      {"role": "user", "content": "If you raise, how likely is it that the opponent will fold or call based on their previous action?"},
      {"role": "user", "content": "Based on the above reasoning, respond ONLY with the index corresponding to your chosen legal action:\n0: call \n1: raise \n2: fold \n3: check"}
    ]

  }
)


