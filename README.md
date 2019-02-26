# ki-simulator v.0.1

This repository regroup the Python version of the scripts that constitute the Ki Simulator. 

## Structure
The current iteration is structured as follows: 

```
|-- ki-simulator 
|    |
|    |--reward : the dynamic reward simulator 
|    |   |--data : the simulation data 
|    |   |   |--btc : bitcoin data
|    |   |   |--eth : ethereum data
|    |   |
|    |   |--res : the simulation results
|    |   |   |--<forder_by_date> : simulation results per run date
|    |   |
|    |   |--src : the simulation scripts
|    |   
|    |--reputation the PoR simulator 
|
|--README.md : this readme 

``` 

## Getting started
To use the simulator, start by cloning the repository and navigating to your simulator of choice :

``` 
git clone https://github.com/GetGenki/ki-simulator/tree/master/reward && cd <reward|reputation>
``` 

The simulator are configurable using the .env file contained in the _ki-simulator/reward_ and _ki-simulator/reputation_ folders. Details on the configuration process are given in the next sections.  

## Reward simulator 
The reward simulator contains 4 components and a configuration file. Following are the description of these files:
 
 __The data generator__ - _reward/src/generator.py_ : 
* Allows to load and format transaction data from a data file.
* Allows to generate transaction data from a preset distribution configuration.
* Allows to generate a set of validators with their relative stakes from a preset distribution configuration.
* Allows to distribute the validation spots over the generated validators based on a preset strategy.

__The scripts launcher__ - _reward/src/launcher.py_ :
* Allows to launch different reward simulation scenarios.

__The forecasting module__ - _reward/src/predictor.py_ : 
* Allows to make predictions on the number of transaction based on historical data.

__The reward calculator__ - _reward/src/rewarder.py_ : 
*

__The configuration file__ - _reward/.env_ 
*

## PoR simulator
coming soon...

