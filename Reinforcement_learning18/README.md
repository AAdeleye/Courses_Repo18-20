Hello friends.

# Dockers of interest

This has Classic Control, Atari, and MuJoCo environments (gym=0.9.something, mujoco-py=0.5.something)
**mcsit/ece267chw4:latestAtari_base**

This has Classic Control, Atari, MuJoco, and Robotics (gym=0.10.5, mujoco-py=1.50.1.56)
**mcsit/ece267chw4:latestRobotics**

# To run the Dockers

In ./Courses_Repo18-20/ run
```
python testEnv.py
```
It will give you an error message indicating that you have to add something to your .bashrc. Copy the export LD_LIBRARY etc etc etc text and run it in your terminal. Then run python testEnv.py again and the environments should work.

If this doesn't work, check if you need to update gym and mujoco to the versions indicated above. In the cluster terminal run:

```
pip install gym --upgrade
pip install mujoco-py --upgrade
```

If the Atari env doesn't work, follow the pip install gym[atari] command that the exception provides.


### When you want to run the jupyter notebook to get all the environments, run these three commands
1. conda env list  (to get the name of your env)
2. source activate myenv
3. python -m ipykernel install --user --name myenv --display-name "Python (myenv)"


# To run a jupyter notebook in a local Docker
```
sudo docker run -p 8888:8888 -it docker/fileThing:tag /bin/bash
jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
```

Do not use the link they provide you with. Do note that they provide you with a token though. Open a new web browser and write:

```
localhost:8888 (or whatever port number you used above)
```

Put the token text from the command line into the browser.
