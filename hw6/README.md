In this assignment, you'll experiment with a policy gradient algorithm for RL. We'll play with an open-source A3C implementation from Nvidia [(code)](https://github.com/NVlabs/GA3C). The description of the problem statement and a brief background is provided in ```assets/main.pdf```. 

### Required Reading
1. [A3C](https://arxiv.org/pdf/1602.01783.pdf) -- Helps you grasp the objective function, network architecture, idea of parallel agents etc. Definitely look at the pseudocode for the algorithm (Supplementary Algorithm S3).
2. [GA3C](https://openreview.net/pdf?id=r1VGvBcxl) -- Will go a great deal in helping you understand how this code from Nvidia works.

### Code Structure

##### GA3C.py
Entry point of your program. You'll run your code with a command like ```python3.4 GA3C.py```. Check the ```sample_pbs``` file to see how you can provide other command line options, and also how to set up some aprun knobs when submitting a batch job. 

If the game runs fine, the output should look like below:

*...
[Time: 33] [Episode: 26 Score: -19.0000] [RScore: -20.5000 RPPS: 822] [PPS: 823 TPS: 183] [NT: 2 NP: 2 NA: 32]
[Time: 33] [Episode: 27 Score: -20.0000] [RScore: -20.4815 RPPS: 855] [PPS: 856 TPS: 183] [NT: 2 NP: 2 NA: 32]
[Time: 35] [Episode: 28 Score: -20.0000] [RScore: -20.4643 RPPS: 854] [PPS: 855 TPS: 185] [NT: 2 NP: 2 NA: 32]
[Time: 35] [Episode: 29 Score: -19.0000] [RScore: -20.4138 RPPS: 877] [PPS: 878 TPS: 185] [NT: 2 NP: 2 NA: 32]
[Time: 36] [Episode: 30 Score: -20.0000] [RScore: -20.4000 RPPS: 899] [PPS: 900 TPS: 186] [NT: 2 NP: 2 NA: 32]
...*

**PPS** (predictions per second) demonstrates the speed of processing frames, while **Score** shows the achieved score. **RPPS** and **RScore** are the rolling average of the above values.

##### NetworkVP.py
Builds the TF computational graph for policy and value networks. The architecture is almost the same as the A3C paper. There are convolutional layers to extract features, followed by different heads to make the policy-predictions and value-function-predictions. Make sure to understand the ```_create_graph(self)``` function completely. You can skip over some portions which are conditional on configuration setting, e.g. ```Config.DUAL_RMSPROP``` is always False for us (see Config.py below). 

```
def _create_graph(self):
        self.x = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.img_channels], name='X')
        self.y_r = tf.placeholder(tf.float32, [None], name='Yr')
        self.advantages = tf.placeholder(tf.float32, [None], name='advantages')
        ...
```

##### ProcessAgent.py
This, hopefully, is the only file that needs modification for this assignment and therefore it's crucial to understand it well . It contains the code that is run by each of the parallel game playing agents. The agents use ```prediction_q``` and ```training_q``` to interact with predictor and trainer threads, as should be clear from the GA3C paper. For part-2, you'll modify the ``` _accumulate_rewards()``` function to use generalized advantage estimation (GAE). Make yourself familiar with the current advantage estimation in place:

```
reward_sum = discount_factor * reward_sum + r
experiences[t].reward = reward_sum
experiences[t].advantage = reward_sum - experiences[t].value
```

To calculate GAE, one possible approach is to calculate the TD-residual using the information already available in the ```experiences``` data structure, store it as ```experiences[t].delta```, and then calculate ```experiences[t].advantage``` by looping over ```experiences``` in a fashion very similar to existing code. Note that there might be other legitimate ways of doing this as well.

##### Config.py
Contains all the relevant configuration parameters. You should only be required to modify ```ATARI_GAME```, ```GAE_LAMBDA```, ```BETA_START``` and ```BETA_END```. You should set the last two to the same value since we'll not perform annealing for this assignment.  ```AGENTS```, ```PREDICTORS``` and ```TRAINERS``` control the number of threads running on the CPU, so you can tweak those based on your CPU resources. 

##### Environment/GameManager.py
These files handle the game environment and provide a clean API to interact with it. It probably won't hurt that much to skip major sections of these files, but look at the ```step()``` function that actually takes a step in the environment with the action of your choosing and returns the next observation.

##### Other files
Server, ThreadPredictor, ThreadTrainer and other classes are mainly for efficient management of the asynchronous execution and transfer of data between threads/processes. Look at them on a need-to basis.

### BlueWaters Hours (critical)
RL is typically heavy on computation, and model-free first-order policy-gradient methods more so. **Your final plots for Seaquest (Part-1) and Pong (Part-2) should be run for 12 hours each**. When doing development, be very judicious in using hours - DO NOT start a 8-10 hour run unless you are fairly confident that your code modification makes sense.
