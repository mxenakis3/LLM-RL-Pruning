# CLASS FOR CONFIGURATION
# 1) Attribute style access: Instead of config["eps_start"], we can use "config.eps_start" which is cleaner
# 2) We can expand the class later to include validation, default values, etc.

class Config:
  def __init__(self, config_dict):
    self.__dict__.update(config_dict)