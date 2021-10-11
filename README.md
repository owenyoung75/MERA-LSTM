## MERA-LSTM

### Experimental Implementation and Explanation Material

This is an public repository for the work _Entanglement-Structured LSTM Boosts Chaotic TimeSeries Forecasting_. 
Mainly two sets of information are included here:    
1. [Code implementation for experiments](https://github.com/owenyoung75/MERA-LSTM/tree/main/code);    
2. [Related instructional/explanational materials](https://github.com/owenyoung75/MERA-LSTM/tree/main/materials).    


- The [material folder]((https://github.com/owenyoung75/MERA-LSTM/tree/main/materials)) contains mainly slides used previously in off-line seminars/workshops by authors.
More informaton about original ideas and motivations could be found in slides, which are more detailed compared with the manuscript.
<p float="left" align="center">
  <img src="https://user-images.githubusercontent.com/16418655/136801384-6c37d557-a010-4606-9d97-111246c74afb.png" width="400" />
  <img src="https://user-images.githubusercontent.com/16418655/136801169-0eae8ab8-9930-4518-85f5-0764411cada9.png" width="400" /> 
</p>

- For the [code implementation](https://github.com/owenyoung75/MERA-LSTM/tree/main/code) part, to make the result and experiments more transparent and reproducible, instead of a package of codes in a script language only, we implement the newly proposed model using a [Mathematica](https://www.wolfram.com/mathematica/) notebook, which contains all params and result numbers&figures, making results directly visible.
- <p float="left" align="center">
  <img src="https://user-images.githubusercontent.com/16418655/136801543-3173a985-6333-4fa5-a77d-0014cd8b60bb.png" width="400" />
  <img src="https://user-images.githubusercontent.com/16418655/136801600-e55b2621-faa7-43c7-85cb-b358513e8714.png" width="400" /> 
</p>


### Data Source

The work uses two types of data:
1. Data from simulation: all training-testing data used for various dynamical systems are generated using standard methods (iteration method for discrete dynamics, and standard ODE, e.g. Runge-Kutta, for continuous dynamics). All parameters used for each system in simulation has been listed in details in the Appendix of the manuscript.
2. Data from external sources: we use external data in the experiments on [weather datasets](https://reference.wolfram.com/language/ref/WeatherData.html), which are provided by [Wolfram Research](https://www.wolfram.com/).
As documented on the [official site](https://reference.wolfram.com/language/note/WeatherDataSourceInformation.html), [WeatherData](https://reference.wolfram.com/language/ref/WeatherData.html) is based on a wide range of sources, with enhancement at the Wolfram Research Companies by both human and algorithmic processing. [WeatherData](https://reference.wolfram.com/language/ref/WeatherData.html) is continually maintained with the latest available information, with automatic updating when WeatherData is used inside the Wolfram Language. Among current principal sources for WeatherData are:
- Citizen Weather Observer Program. "Citizen Weather Observer Program (CWOP)." [»](http://www.wxqa.com/)
- Gladstone, P. "Locations of the Weather Stations of the World." [»](https://weather.gladstonefamily.net/cgi-bin/location.pl/pjsg_all_location.csv?csv=1)
- National Oceanic and Atmospheric Administration. "National Weather Service." [»](https://www.weather.gov/)
- United States National Climatic Data Center. "Global Surface Summary of Day." [»](ftp://ftp.ncdc.noaa.gov/pub/data/gsod/)
- United States National Climatic Data Center. "Integrated Surface Database." [»](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database)
