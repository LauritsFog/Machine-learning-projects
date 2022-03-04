***************************************************************************
***				            Alg_FFire									***
***************************************************************************
***************************************************************************
This data set is change compared to the origional dataset.
The changes has been made to make it easier to handle the data in matlab.
The following changes has been made:
	* We have added an ID field to every observation s.t. we can easily distinguis between the observations
	* The dataset was "split" in two different regions, we have concatenated the data and know that at line 123(in the csv file) the other region begins.

Explanation of attributes:
Link to data: (https://archive.ics.uci.edu/ml/datasets/Algerian+Forest+Fires+Dataset++)

Our data has 15 columns. They each describe the following:

1) ID 
2) Day of month
3) month (in number)
4) year
5) Temperature in Celcius
6) Relative Humidity in procent (RH)
7) Wing Speed in km/h (Ws)
8) Total rain that that in mm (Rain)
9) Fine Fuel Moisture Code index (FFMC)
10) Duff Moisture Code index (DMC)
11) Drought Code index (DC)
12) Initial speed index (ISI)
13) Buildup Index (BUI)
14) Fire Weather Index (FWI)
15) Classes: "fire" and "not fire"