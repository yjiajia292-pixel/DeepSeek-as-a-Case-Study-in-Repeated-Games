DeepSeek in Repeated Games: Micro-Level Analysis of Decision Trajectories
This repository contains the complete experimental code used in our study on DeepSeek-V3.1’s decision-making behavior in repeated games, with a particular focus on micro-level action trajectories and reasoning processes in the Iterated Prisoner’s Dilemma (IPD).

1. Experimental Settings
   
1.1 Prompting Modes

We consider two prompting settings that define the information provided to the LLM at each round:

  m&o: The prompt includes both the opponent’s previous action and the model’s own previous action.
  
  o: The prompt includes only the opponent’s previous action.
  
1.2 Memory Length
Prompt memory length controls how much historical information is provided:
  1: Only the previous round’s action(s) are provided (memory-one).
  all: All past rounds’ actions are provided (full memory).

3. Experiments Overview
All experiments are conducted in the repeated IPD framework with noisy action transmission.
2.1 Experiment 1: Noise Effects under Memory-One
This experiment examines how external noise affects LLM decision-making when memory length is fixed to one.
  Memory length: 1
  Prompting mode: m&o or o
  Focus: Sensitivity of LLM cooperation to noise under minimal historical context
2.2 Experiment 2: Memory Length Effects
This experiment investigates how different memory lengths influence LLM behavior under noise.
  Memory length: 1 vs. all
  Prompting mode: fixed
  Focus: How long-term history alters cooperation stability and strategy formation
2.3 Experiment 3: Prompting Information Scope
This experiment compares how different prompting information scopes affect LLM behavior under memory-one settings.
  Memory length: 1
  Prompting scope: m&o (opponent and self actions) vs. o (opponent actions only)
  Focus: how the dimensionality of observed interaction signals influences cooperation decisions under noise

4. Code Structure and Naming Convention

All experimental scripts follow a consistent naming scheme: single_<noise>_<memory>_<prompt>.py
For example:
  single_0.01_1_m&o.py
    → Noise level = 0.01
    → Memory length = 1
    → Prompt includes both model and opponent history

The repository includes all code files used in the paper, covering:
  Two-stage sampling procedure
  Noise injection
  Classic opponent strategies (ALLD, TFT, WSLS, Extortion, Majority, CURE, etc.)
  Round-level logging of actions and outcomes

4. Outputs
Each run produces:
  Detailed round-level logs (true actions vs. noisy actions)
  Summary statistics including cooperation rates, average payoffs, and confidence intervals
These outputs correspond directly to the empirical results reported in the paper.
