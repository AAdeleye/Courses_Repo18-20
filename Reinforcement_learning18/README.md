Hello friends.

#Docker items of interest:

This has classic control, Atari, and MuJoCo environments (gym=0.9.something, mujoco-py=0.5.something)
mcsit/ece267chw4:latestAtari

This has classic control, Atari, MuJoco, and Robotics (gym=0.10.something, mujoco=1.5.something)
mcsit/ece267chw4:latestRobotics

To run this:

Run python testEnv.py from the github. It will give you an error message indicating that you have to add something to your .bashrc. Copy the export LD_LIBRARY etc etc etc text and run it in your terminal. Then run python testEnv.py again and the environments should work.

If this doesn't work, check if you need to update gym and mujoco to the versions indicated above. In the cluster terminal run:

pip install gym --upgrade
pip install mujoco-py --upgrade

If the Atari env doesn't work, follow the pip install gym[atari] command that the exception provides.
