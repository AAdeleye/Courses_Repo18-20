Hello friends.

# Dockers of interest

This has Classic Control, Atari, and MuJoCo environments (gym=0.9.something, mujoco-py=0.5.something)
**mcsit/ece267chw4:latestAtari_base**

This has Classic Control, Atari, MuJoco, and Robotics (gym=0.10.5, mujoco-py=1.50.1.56)
**mcsit/ece267chw4:latestRobotics**

To run the program locally, we have to make a number of changes to accomodate. This has Classic Control, Atari, MuJoCo, and Robotics
**mcsit/ece267chw4:localRunRobotics**

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


### Update your cluster with the new gym and mujoco environments
After testEnv.py runs without errors, you need to update your Jupyter Notebooks so that it can use the updated Python with the exported LD_LIBRARY and so on.
```
conda env list  (to get the name of your myenv)
source activate myenv
python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
```

Typically conda env list returns base so the code will be
```
source activate base
python -m ipykernel install --user --name base --display-name "Python base"
```

# Running a Jupyter Notebook in a local Docker
```
sudo docker run -p 8888:8888 -it docker/fileThing:tag /bin/bash
jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
```

Do not use the link they provide you with. Do note that they provide you with a token though. Open a new web browser and write:

```
localhost:8888 (or whatever port number you used above)
```

Put the token text from the command line into the browser.

# Running localRunRobotics Docker locally
If you encounter errors with the localRunRobotics Docker first install python-opengl
```
apt-get install python-opengl
```

Install the following (source: https://stackoverflow.com/questions/21665914/installing-and-configuring-xvfb)
```
apt-get install python-pip
apt-get install xvfb xserver-xephyr vnc4server
pip install pyvirtualdisplay
```

Set-up the rest of the packages (source: https://stackoverflow.com/questions/40195740/how-to-run-openai-gym-render-over-a-server)
If on a linux server, open jupyter with
```
$ xvfb-run -s "-screen 0 1400x900x24" jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
```

In Jupyter
```
import matplotlib.pyplot as plt
%matplotlib inline
from IPython import display
```
The notebook already has from IPython.display import something, but I added the last line in for good measure.
