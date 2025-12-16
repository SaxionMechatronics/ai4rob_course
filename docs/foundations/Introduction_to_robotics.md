


Before exploring how artificial intelligence (AI) can enhance robotics, it’s important to understand what robotics actually is. The definition isn’t as straightforward as it seems because robotics is a broad field.



## What Is Robotics?

Robotics is about designing, building, and using robots. It combines knowledge from mechanical, electrical and software engineering. It is a multidisciplenary field, which makes it also difficult to explicitly define what a robot is. There’s no universally agreed definition of a robot. In general, it’s a machine that can do tasks on its own or with just a little help from people. So, something like a drill doesn’t count as a robot, because you still have to use it yourself. It’s just a tool. But a robot vacuum cleaner, like a Roomba, is considered a robot, because it can clean your floor without you doing anything.

## Common Types of Robots

When people hear the word “robot,” they often think of human-like machines or robotic arms, but the world of robotics is much bigger. There are lots of different types of robots, each built for specific jobs, and each with its own set of challenges. Robotics includes many types of robots, each designed for specific tasks and facing unique challenges. 


### 1. Mobile Ground Robots
These are robots that move around on wheels or legs and use sensors like GPS, cameras, and gyroscopes to figure out where they are and what’s around them. They’re often used for inspections or surveillance. For example, in the CHARISMA project, a four-legged robot was used to detect gas leaks on the street. These robots do well in places that don’t dynamically change much, like homes or warehouses. Once they’ve mapped the area, they can move around pretty easily. But things get tricky when the environment is unpredictable. In a forest or a busy street, the robot might run into unexpected obstacles, moving objects, or changes in lighting. For example, a human suddenly jumping in front of the robot.  These things can confuse the sensors or mess up the robot’s navigation. Most of these robots rely on a map they made earlier, and if something changes, they don’t always know how to react.



### 2. Aerial Robots (Drones)
Aerial robots, like drones, face similar problems but with extra layers of difficulty. Flying is harder than driving, and it’s easier to lose control. If a drone’s battery dies, the drone will just fall out of the sky, which can be dangerous. Drones are used for inspections and search-and-rescue missions, but they struggle when something unexpected happens. For example, if the GPS signal drops, the drone might lose track of where it is. Wind, birds, or sudden changes in terrain can also throw it off. And because drones are often far from the operator, fixing problems on the fly isn’t easy.

### 3. Underwater Robots
Underwater robots have their own set of challenges. They work below the surface, where everything has to be waterproof and visibility is poor. Cameras don’t work as well underwater, and sensors can get distorted by water pressure or particles. Communication is also harder, radio signals don’t travel well underwater, so robots often have to rely on cables or very limited acoustic signals. All of this makes underwater robotics a tough area to work in. Even simple tasks like identifying objects or navigating through narrow spaces become complicated.


### 4. Manipulators
Manipulators are robots that don’t move around but are built to handle objects. Think of robotic arms in factories. It is good to know that there is a distinction baetween robots and cobots. Cobots are used to collaborate with humans and are designed to reduce the risk of clamping, but also have torque/force detection. This detection ensures thats cobot stops moving when the force/torque reaches a threshold. These are pretty advanced and are used a lot in manufacturing, where everything is controlled and predictable. For example, in a car factory, the same parts are put together in the same way over and over again. That’s perfect for a robot. But if you throw in some randomness, like a bin full of mixed-up objects, things get a lot more complicated. The robot has to figure out what the object is, how big it is, what angle it’s sitting at, and how to pick it up without dropping it. These are things humans do without thinking, but for robots, it’s a real challenge. It is important 



### 5. Humanoid Robots
Humanoid robots are designed to look and move like people. They’ve been around for a while, but they haven’t really made it into everyday life. Only recently, thanks to better learning techniques and control systems, have they started to show more promise. But they’re still mostly used in research or special projects. They’re expensive, complex, and hard to control. Walking, balancing, and interacting with objects in a human-like way is incredibly difficult. And because they’re meant to operate in human environments, they need to deal with all the unpredictability that comes with it.



## The Big Challenge in Robotics

If you look at all these types of robots, you’ll notice they all run into the same kind of problem. They’re great when things are predictable, but they struggle when something unexpected happens. Whether it’s a new obstacle, a change in the environment, or a different kind of task, robots often don’t know what to do. They’re not great at adapting.

And that’s the big challenge in robotics right now. How do we build robots that can deal with new, unfamiliar situations without needing someone to reprogram them every time? That leads to something which is simply impossible, because for each task it should be reprogrammed. This is not a feasible solution as there are infinitely many tasks that exists.

This is where artificial intelligence might be able to help. But before we get into that, it’s important to understand what AI actually is, and how it connects to things like machine learning, deep learning, and reinforcement learning. These terms get thrown around a lot, and it’s easy to mix them up. So before we talk about how AI could make a difference in robotics, let’s first break down what these ideas really mean.

[Continue to  introduction to artificial intelligence →](introduction_to_ai.md){ .md-button .md-button--primary }
