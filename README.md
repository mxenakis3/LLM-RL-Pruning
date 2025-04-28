<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#instructions"> Instructions </a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#running-the-code">Running the Code</a></li>
      </ul>
    </li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## Instructions
CS7643 Project: Accelerating Deep Multi-Agent Reinforcement Learning with LLM-Agents
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Instructions -->
## Getting Started
To get a local copy up and running follow these simple example steps.

### Installation
1. Clone the repo
   ```sh
   git clone https://github.com/mxenakis3/LLM-RL-Pruning.git
   ```
3. Install python packages 
   ```sh
   pip install -r requirements.txt
   ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Running the code
The code to generate results is available under /notebooks, and configurations can be changed under /configs.  Running the notebooks sequentially will generate similar results as what was presented in the report, with results stored in pickle files under the same directory.  

Specifically, 
<ol>
    <li>/notebooks/ppo_example_rl_Card and /notebooks/ppo_example_rl_Card_heuristic will generate results similar to those presented in section 5.2.2</li>
</ol>