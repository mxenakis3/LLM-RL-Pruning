from configs.config_class import Config

agent_configs = Config(
  {
    "system_message" : {"role": "system", "content": """
    You are an agent as part of a 2-player team that cooks onion soups in a kitchen. The objective is to deliver as many soups as possible in a limited time window.
                        
    Each soup takes 20 timesteps to cook.
                        
    Each soup requires three onions.
                        
    The kitchen has one or more onion dispensers (where you can obtain an onion) and dish dispensers (where you can obtain a dish) which provide an unlimited supply of onions/dishes.
                        
    In order to cook the soup, you must add three onions to the soup, turn the stove on and allow the soup to cook for twenty timesteps, and then carry the soup to the serving area.
                        
    Note that our layouts do not contain tomatoes, so features corresponding to tomatoes will always be 0. You can disregard this part of the state space. 
                        
    The state information is as follows: {state_info}
    """},

    "cot_prompts": [
      {"role": "user", "content": "Does the closest pot require additional onions? If not, are you or your partner holding an onion?"},
      {"role": "user", "content": "If the closest soup does not require additional onions, is the soup cooking?"},
      {"role": "user", "content": "If the closest soup is cooking, is it ready (or almost ready) to be served?"},
      {"role": "user", "content": "If the soup is ready to be served, is the soup plated?"},
      {"role": "user", "content": "If the soup is ready to be served and the soup is not plated, do you or your partner have a dish?"},
      {"role": "user", "content": "What plan do you expect your partner has at this timestep?"},
      {"role": "user", "content": "based on the above conversation, suggest a reasonable action to take at this timestep."},

      # This is where tool calling will go when it is ready. 
      {"role": "user", "content": """Translate this into a function call. Options:
        up(): move in the +y direction
        down(): move in the -y direction
        left(): move in the -x direction
        right(): move in the +x direction
        stay(): do nothing
        interact(): Contextual action determined by the tile the you are facing. 
            (e.g. placing an onion when facing a stove)
            (e.g. delivering a dish when facing a delivery area)
            (e.g. picking up a dish when facing a dish dispenser)
      Format as follows:
      Example 1: 
      'up()'
      Example 2:
      'down()'
      Example 3:
      'stay()'
       etc.
      """},

    ],

    "tool_schemas": [
      {
        "type": "function",
        "function": {
          "name": "up",
          "description": "move in the +y direction",
          "parameters": {
            "type": "object",
            "properties": {},
         },
         "required": [],
         "additionalProperties": False
        },
        "strict": True
      },

      {
        "type": "function",
        "function": {
          "name": "down",
          "description": "move in the -y direction",
          "parameters": {
            "type": "object",
            "properties": {},
         },
         "required": [],
         "additionalProperties": False
        },
        "strict": True
      },

      {
        "type": "function",
        "function": {
          "name": "left",
          "description": "move in the -x direction",
          "parameters": {
            "type": "object",
            "properties": {},
         },
         "required": [],
         "additionalProperties": False
        },
        "strict": True
      },

      {
        "type": "function",
        "function": {
          "name": "right",
          "description": "move in the +x direction",
          "parameters": {
            "type": "object",
            "properties": {},
         },
         "required": [],
         "additionalProperties": False
        },
        "strict": True
      },

      {
        "type": "function",
        "function": {
          "name": "stay",
          "description": "Do nothing",
          "parameters": {
            "type": "object",
            "properties": {},
         },
         "required": [],
         "additionalProperties": False
        },
        "strict": True
      },

      {
        "type": "function",
        "function": {
          "name": "interact",
          "description": """
            Contextual action determined by the tile you are facing. 
            (e.g. placing an onion when facing a stove)
            (e.g. delivering a dish when facing a delivery area)
            (e.g. picking up a dish when facing a dish dispenser)
          """,
          "parameters": {
            "type": "object",
            "properties": {},
         },
         "required": [],
         "additionalProperties": False
        },
        "strict": True
      },
    ]

  }
)

lunarlander_cot_agent_configs = agent_configs

def up():
  return 0

def down():
  return 1

def left():
  return 2

def right():
  return 3

def stay():
  return 4

def interact():
  return 5


