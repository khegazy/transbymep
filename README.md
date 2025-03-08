# Popcornn
Path Optimization with a Continuous Representation Neural Network for reaction path with machine learning potentials

## Installation and Dependencies
```
python3 -m pip install popcornn
```

## Optimization
You can find several run files inside the `example` directory that rely on the implemented modules in the Popcornn library. The run scripts need to be accompanied with a yaml configuration file. You can run an example optimization script with the following command:
```
python run.py --config configs/6445.yaml
```
You can also call the optimization as a function, which takes a list of ASE Atoms as an input and returns a list of ASE Atoms for the reaction pathway and the transition state ASE atoms as outputs. You can run an example script with
```
python func_run.py
```
Since the reaction path is continuous, the input and output sizes can in general be different. The input list must contain at least the two endpoints. The endpoints are used as-is and will not be (re-)aligned or (re-)mapped.
