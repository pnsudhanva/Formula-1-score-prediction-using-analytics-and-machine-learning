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

![Position changes](https://github.com/pnsudhanva/Formula-1-score-prediction-using-analytics-and-machine-learning/assets/14261453/f68a5005-6024-4599-b1b9-c91b6806ae9a)


Here, we can see that a high scoring race has many more position changes than the low scoring race. This suggests that the number of take overs are an important factor that influence the race score.


Now, digging into the position changes chart, we can see that drivers from last position will come to a higher position at a specific lap in the race. This is shown by the below image.


![Top position changes](https://github.com/pnsudhanva/Formula-1-score-prediction-using-analytics-and-machine-learning/assets/14261453/c78ef6ae-dac4-40c5-a1f4-eebe8d516f14)


I can conclude that the Brazilian GP doesn't only have more take overs in general, it also has more take overs in the leading positions, especially in the top positions- are an important factor that influence the score that fans give to a race.


Furthermore, it is imperative that certain drivers facing engine failures, or crashes in the middle of the race, counted as DNFs (did not finish). By viewing closely, we can see that certain drivers like Lewis Hamilton failed to complete the race at the 55th lap, and there are many more.

![dnfs](https://github.com/pnsudhanva/Formula-1-score-prediction-using-analytics-and-machine-learning/assets/14261453/3ac666c2-42ec-4131-817a-02e786fe8bb4)


To further verify this, I had utilized a scatter plot to check how fan scores are affected by overtaken positions.


![scatter plot for position changes](https://github.com/pnsudhanva/Formula-1-score-prediction-using-analytics-and-machine-learning/assets/14261453/1cfcc206-216a-45ad-ad71-09ecc11f8292)


And finally, weather plays a major role in an any F1 race as it might affect the traction of the cars on the road by making it slippery, and also, blinding the drivers visibility by 60%. To check this, I had to utilize a box plot where the median corresponds to the fan score.

![using box plot for weather changes](https://github.com/pnsudhanva/Formula-1-score-prediction-using-analytics-and-machine-learning/assets/14261453/beb48966-b92b-4b06-b583-d6816e166d5e)
