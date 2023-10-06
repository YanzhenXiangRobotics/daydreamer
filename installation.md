- Create a conda env for Isaac Orbit if you aims to work on Daydreamer w/ Orbit. The following command automatically create a conda environment of **python 3.7.16** compatible w/ Orbit
  
  ``orbit --conda ENV_NAME``
- Install dependencies for Daydreamer
  
  ``pip install -r daydreamer/requirements.txt``
- Install dependencies for XArm

  ``git clone git@github.com:xArm-Developer/xArm-Python-SDK.git ANOTHER_PATH``

  ``cd xArm-Python-SDK && python setup.py install``
- Test if the code runs

  ``cd PATH_DAYDREAMER``

  ``rm -rf ~/logdir/run1``
  
  ``CUDA_VISIBLE_DEVICES=0 python embodied/agents/dreamerv2plus/train.py --configs xarm --run learning --task xarm_dummy --tf.platform cpu --logdir ~/logdir/run1``
  
- TODO: resolve TF w/ GPU 
  
