# Artificial Intelligence and Its Role in Robotics

Before the emerge of AI, robots had to be programmed line by line. If you wanted to learn the robotic, each part had to be programmed. That works fine for simple and well defined tasks, but it quickly falls apart when things get messy or unpredictable. That’s where AI comes in. Artificial Intelligence, or AI, is the idea that computers can be made intelligent. The core of AI is that machine/algorithms can do things which typically require human intelligence. That includes stuff like:

- Making decisions
- Solving problems
- Learning from experience
- Understanding language
- Recognizing what’s in an image

In its most ambitious form, this is called Artificial General Intelligence (AGI). That’s the kind of AI that could learn and adapt completely on its own, without needing help from humans. It’s like giving a machine a brain that works like ours, which means that it does not explicitly need to be programmed on how the learn things. It can perceive and learn things by itself.

But let’s be honest, most of the AI we see today isn’t that advanced. What we actually have is called narrow AI or weak AI. It’s designed to do one specific thing really well, but it doesn’t handle anything outside the task it has been programmed/learned for. 

For example,  ChatGPT is great at working with text. However, especially in the beginning it could not handle even simpler things like `1 + 1`, it was easily tricked. The reason is that the AI does not have consciousness, but makes predictions on what is the most likely next word based upon context [[1](https://arxiv.org/abs/2406.02356)]. It is lacking the logic reasoning that we as humans have.  

## AI as a Field

Now, AI isn’t just one thing. It’s a whole field made up of different techniques and approaches. One of the most used areas within AI right now is machine learning. In this subfield, algorithms/computers are learned based upon patterns from data instead of being directly programmed. The data from which is learned can come from:
- Experiments
- Sensors
- Simulations
- Artificially generated data

It is important to keep in mind that also for artificial intelligence the rule of garbage in, garbage out applies. In other words, the algorithm is only as good as the data. If the data does not makes sense, it is nearly impossible to create a properply trained model. 


## Types of Machine Learning

The field of machine learning can be divided in three smaller groups:

### 1. Supervised Learning
Supervised learning is probably the most straightforward type. You give the computer a bunch of examples where the correct answer is already known. The model tries to learn the relationship between the input and the output. During training, it makes predictions and checks if they’re right. If not, it adjusts itself to do better next time. This kind of learning is great for tasks like classifying images or predicting values. The downside? You need a lot of labeled data, and labeling can be time-consuming and expensive. On top of that, this labelling is still done by humans [[2](https://theconversation.com/long-hours-and-low-wages-the-human-labour-powering-ais-development-217038)].


### 2. Unsupervised Learning
In unsupervised learning the data doesn’t come with labels. The model has to figure things out on its own. It looks for patterns, groups similar items together, or tries to simplify the data by reducing its dimensions. This is useful for things like clustering customers based on behavior or finding hidden structures in data. Since there’s no “correct” answer, it’s more about exploration than precision. In the end during training and also testing it is important that the engineer/developer also reflects the perforamcne of the system towards its own expectation, because there is no clear right or wrong. 


### 3. Reinforcement Learning
Finally, we have reinforcement learning. This one’s a bit different compared to the other two types of learning. In this type t. The model learns by interacting with its environment. It tries things out, gets feedback in the form of rewards or penalties, and uses that to improve its behavior over time. Think of it like training a dog, you give it treats when it does something right. After some point in time, the dog will start to listen as it assumes it will get a treat.  But here’s the catch: if the feedback isn’t clear, the model might learn the wrong thing. That’s why designing the reward system is super important. 

## Recap
Artificial Intelligence has transformed robotics from rigid, line-by-line programming into systems that can learn, adapt, and improve over time. While most AI today is still narrow and task-specific, techniques like supervised, unsupervised, and reinforcement learning allow robots to handle more complex and dynamic environments than ever before. 

However, AI is not magic—it depends heavily on the quality of data and the design of learning systems. The principle of Garbage In, Garbage Out reminds us that poor input leads to poor results. As research advances, the ultimate goal remains creating robots that can operate reliably in unpredictable situations without constant human intervention.

Next, we will explore how these AI techniques are applied in real-world robotics and what challenges still need to be solved.   
