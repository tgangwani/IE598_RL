In this assignment, you'll use policy gradient reinforcement learning for a continuous control task. The environment is [here](https://gym.openai.com/envs/LunarLanderContinuous-v2/) (think SpaceX rocket booster landing; a cool [video](https://www.youtube.com/watch?v=RfWAUXAJm-o)). Read the description and understand what is the count, range and meaning of actions in this environment. The problem statement is in ```assets/main.pdf```.

``NetworkVP.py`` is the only file that should require modification for this assignment, although you are free to change other files. It is recommended that you keep the parameters in ``Config.py`` unchanged. In ``NetworkVP.py``, you'll add the policy network, value-function network, log-likelihood and entropy. Search for the string "YOUR CODE HERE" in the file.

### Visualization
[Disclaimer: This has not been tested, but should work]
If you'd like to visualize the performance of the policy after training, you could use the model save/restore functionality provided in ``NetworkVP.py``. Save the model after training, restore the model and then run by setting Config.PLAY\_MODE to True. This renders the environment. You can do the restore and rendering on your local machine if it doesn't work on BlueWaters.
