# Formula-1 score prediction using analytics and machine learning

This project explores the Formula 1 data analysis using Oracle Autonomous Data Warehouse (ADW) and Oracle Analytics Cloud (OAC). It delves into the factors that contribute to the race experience for fans using machine learning, leveraging historical race data from 1950 to 2021 and fan score data ranging from (2008 - 2020) from racefans.net. This project also aims to predict the fan score for 2021.

The tasks involve the following:
- **Data Loading**: Setting up the ADW environment and loading Grand Prix data.
- **Basic Exploration**: Identifying potential factors influencing fan appreciation, like weather, circuit, and overtakes, through data visualization.
- **Advanced Exploration**: Uncovering hidden patterns by crafting features like "Come back score" and "RankVersusPosition".
- **Machine Learning Training & Prediction**: Using AutoML to build a model predicting fan scores and evaluating its accuracy.


The first step is to set up a user-schema that is essential for ingesting different types of race data such as (Races data, Lap Times data, Safety Car data, Pit Stop data, Race Results data, and Driver Ranking data), all of which are in .csv format and each file has 1000s of rows and columns containing all the related info regarding the race. This is also useful for ramping up the speed at which the data is processed.



**Basic Data exploration:**
After analyzing the Races data, and finding which season has highest and lowest scores respectively, now it is important to visualize the data by position, laps, and driver_ID.

![Position changes](https://github.com/pnsudhanva/Formula-1-score-prediction-using-analytics-and-machine-learning/assets/14261453/912994fc-be0f-4ab0-83fc-10d2b18a2e8a)

Here, we can see that a high scoring race has many more position changes than the low scoring race. This suggests that the number of take overs are an important factor that influence the race score.
