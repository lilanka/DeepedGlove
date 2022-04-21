# AI based energy optimization solution
Inputs -> System -> outputs

## System
The optimization system should,  
1. Understand the recurring patterns amoung the optimization parameters to better schedual the functioning system.
2. Build a simulation model of the system and assess different control strategies on the model to build most effiencint one.

The system outputs how much to change each controllable parameters. 
1. Temperature of each section
2. Flow control of each section
3. Level control of each section
4. Motor control of each section

## Reward function
Reward function is contained following parameters (assuming there will be feedback for everybatch)
* The quality of gloves produced (Q)
* Energy consumption of the plant (EC)
* Number of gloves manufactured in a given day (GN)
* Temperature inside the plant/human factor (HF)

Reward (R) = (w1 * Q) + (w2 * GN) - (w3 * EC) - (w4 * HF)

Since the quality varies with bath to batch, the feedback should be given everytime there is a batch output. 
The Energy consumption (EC) is captured in terms of electricity consumption and CO2 emission.

## Input
* Temperature values (t1-t2) <- floats
* Motor control values (m1-m2) <- floats
* Level control values (l1-l2) <- floats
* Flow control values (f1-f2) <- floats
* Glove type (string)
temize}
flow control valeus (F1-F2) <- float
