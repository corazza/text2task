# text2task

This repo contains code for my master's thesis,
*Synthesis of reward functions from natural language descriptions*.

[![YouTube video](https://img.youtube.com/vi/xhaLgkVK9lI/0.jpg)](https://www.youtube.com/watch?v=xhaLgkVK9lI)

The core task it accomplishes is captured by this workflow:

1. The user describes the task in English, for example, **"patrol the forest and the factory, but avoid traps"**.
2. The language model is invoked to translate the user's description into a reward machine formalism.
3. QRM and the reward machine from the previous step are used to train the agent to perform the task described by the user in the first step.
4. The results are visually demonstrated to the user in a small number of episodes.
