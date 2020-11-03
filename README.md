# initial-drone-training
Supervised pretraining for 'learning-drones' to initialize drone controllers and accelerate learning.


# Start Training

Execute
  
  python nn/train.py
  
For training.

All training (hyper-)parameters can be changed in the same file.

The module controller is responsible for setting up PID controllers as a stand-in for a "drone-brain" (on board computer)

The module create_dataset creates PID ground truth thrust vectors based on different random drone states.
