# Formula-1 score prediction using analytics and machine learning

This project explores Formula 1 data analysis using Oracle Autonomous Data Warehouse (ADW) and Oracle Analytics Cloud (OAC). It delves into the factors that contribute to the race experience for fans using machine learning, leveraging historical race data from 1950 to 2021 and fan score data ranging from (2008 to 2020) from racefans.net. This project also aims to predict the fan score for 2021.

The tasks involve the following:
- **Data Loading**: Setting up the ADW environment and loading Grand Prix data.
- **Basic Exploration**: Identifying potential factors influencing fan appreciation, like weather, circuit, and overtakes, through data visualization.
- **Advanced Exploration**: Uncovering hidden patterns by crafting features like "Come back score" and "RankVersusPosition".
- **Machine Learning Training & Prediction**: Using AutoML to build a model predicting fan scores and evaluating its accuracy.


The first step is to set up a user schema that is essential for ingesting different types of race data such as (Races data, Lap Times data, Safety Car data, Pit Stop data, Race Results data, and Driver Ranking data), all of which are in .csv format and each file has 1000s of rows and columns containing all the related info regarding the race. This is also useful for ramping up the speed at which the data is processed.



**Basic Data exploration:**
After analyzing the Races data, and finding which season has the highest and lowest scores respectively, now it is important to visualize the data by position, laps, and driver_ID.

![Position changes](https://github.com/pnsudhanva/Formula-1-score-prediction-using-analytics-and-machine-learning/assets/14261453/f68a5005-6024-4599-b1b9-c91b6806ae9a)


Here, we can see that a high-scoring race has many more position changes than a low-scoring race. This suggests that the number of takeovers is an important factor that influences the race score.


Now, digging into the position changes chart, we can see that drivers from the last position will come to a higher position at a specific lap in the race. This is shown in the below image.


![Top position changes](https://github.com/pnsudhanva/Formula-1-score-prediction-using-analytics-and-machine-learning/assets/14261453/c78ef6ae-dac4-40c5-a1f4-eebe8d516f14)


I can conclude that the Brazilian GP doesn't only have more takeovers in general, it also has more takeovers in the leading positions, especially in the top positions, an important factor that influences the score that fans give to a race.


Furthermore, it is imperative that certain drivers facing engine failures, or crashes in the middle of the race, are counted as DNFs (did not finish). By viewing closely, we can see that certain drivers like Lewis Hamilton failed to complete the race at the 55th lap, and there are many more.

![dnfs](https://github.com/pnsudhanva/Formula-1-score-prediction-using-analytics-and-machine-learning/assets/14261453/3ac666c2-42ec-4131-817a-02e786fe8bb4)


To further verify this, I utilized a scatter plot to check how fan scores are affected by overtaken positions.


![scatter plot for position changes](https://github.com/pnsudhanva/Formula-1-score-prediction-using-analytics-and-machine-learning/assets/14261453/1cfcc206-216a-45ad-ad71-09ecc11f8292)


And finally, weather plays a major role in any F1 race as it might affect the traction of the cars on the road by making it slippery, and also, blinding the driver's visibility by 60%. To check this, I had to utilize a box plot where the median corresponds to the fan score.

![using box plot for weather changes](https://github.com/pnsudhanva/Formula-1-score-prediction-using-analytics-and-machine-learning/assets/14261453/beb48966-b92b-4b06-b583-d6816e166d5e)


**Advanced Data Exploration**

Now that the basic factors are identified which was pretty much simple to identify and guess, it's time to analyze a bit more into the race season and gather further details that might have contributed to higher fan scores.

Usually, when a driver strives hard and tries to overtake other drivers who are in the front, such instances are termed "comebacks", and it might be a good starting point to guess.

![vettel-comeback](https://github.com/pnsudhanva/Formula-1-score-prediction-using-analytics-and-machine-learning/assets/14261453/fca87dc2-edb6-46dc-9ec6-55ab46e47791)

In this graph, the violet line (reference for "Sebastian Vettel"), is making its way from the 23rd position to the 5th position and back to the 18th position and then finally staying in the 5th position. Overall, his position jumped from (23rd to 5th position), which signifies his large come-back statistics.


During a race, it is imperative that a driver has to make a pit stop to change his car's tires because of wear and tear, damage, etc. Because of this, they might lose their existing position to someone who overtakes them from behind. 

In the below graph, we can see that many drivers getting overtaken (represented by circle dots), and making a pit stop (represented by square box).

![drag-pit-stop](https://github.com/pnsudhanva/Formula-1-score-prediction-using-analytics-and-machine-learning/assets/14261453/83a06199-7e37-4575-a53c-d80e6509e032)


But the caveat here is that there is a huge difference in overtaken due to a pit stop and being overtaken on the actual race, which might affect the fan scores.

![Pit stops](https://github.com/pnsudhanva/Formula-1-score-prediction-using-analytics-and-machine-learning/assets/14261453/83cb5456-3f6a-4b0a-9957-c13dd20e400f)


Finally, during a race, if some driver crashes, or has some issues with the car, they are unable to race at that point. Because of the obstruction or car debris on the track, all the remaining drivers are supposed to slow down wherever they are and are supposed to follow behind a special car (known as a safety car) until the debris is cleared. When this happens, all the remaining drivers will come very close to each other in terms of the actual distance which we can see from the below graph.

![safety-car situation](https://github.com/pnsudhanva/Formula-1-score-prediction-using-analytics-and-machine-learning/assets/14261453/43ceb69a-3257-434a-85f5-8d0c64796846)


**Machine Learning Training & Prediction**

After gathering all the data points that are affecting the fan scores, we're gonna test this by running an ML model on Oracle's Analytics Cloud (OAC), which yields us a prediction graph.

The first part is generating the new features using SQL, i.e,
- Calculate the number of laps that have a change of driver in the leading position.
- Calculate the number of laps that have a change of driver in the top 5 positions.
- Calculate the total number of overtakes in two ways: semi-overtakes due to a pit stop and actual overtakes on the track.
- Calculate the "come back score", defined as the maximum number of positions that a driver recovers from a position in the back of the field.
- Retrieve the number of safety cars in a race.

<img width="953" alt="generating nw features" src="https://github.com/pnsudhanva/Formula-1-score-prediction-using-analytics-and-machine-learning/assets/14261453/49f0ab0a-7686-4774-89be-b25f3da8243d">

Running this, the RACES table will be added with the above-mentioned features.

Now is the testing phase, where I run 5 different models to fetch their accuracy namely, (Generalized Linear Model, Generalized Linear Model Ridge Regression, Neural Network, Support Vector Machine Gaussian, and Support Vector Machine Linear)

![Traning using Auto ML](https://github.com/pnsudhanva/Formula-1-score-prediction-using-analytics-and-machine-learning/assets/14261453/b43a1ce6-23e5-44bb-b5bc-b5efaa718555)

Since we had only fan scores from 2008 to 2020 and historical race data from 1960 to 2021, now, the task lies in predicting the accuracy of the fan scores for the 2020 season by training the model with the previous data through the data flow.

![Predicted scores](https://github.com/pnsudhanva/Formula-1-score-prediction-using-analytics-and-machine-learning/assets/14261453/bee01a01-2368-4f16-9b55-f3358834496b)

Here, we can see that the orange line represents the predicted score and the blue line represents the actual score. Which highlights a great level of accuracy.

Just on a side note, if I want to check the difference between the two, I can add a column below the graph to represent the difference which we can see below.

![difference between actual and predicted scores](https://github.com/pnsudhanva/Formula-1-score-prediction-using-analytics-and-machine-learning/assets/14261453/d19cb29b-a42e-4372-ae27-438e34338247)

Finally, my job is to predict the scores for the 2021 season, because I only had fan scores from 2008 to 2020. This is just to prove my prediction for the future fan scores for the 2021 season. 

![Predicting future scores based on past data](https://github.com/pnsudhanva/Formula-1-score-prediction-using-analytics-and-machine-learning/assets/14261453/4eba3ad1-d0f5-48cd-b140-be8a038f9437)


The results seem to be almost similar, except for the final race at Abu Dhabi where there was a huge drama/rule change in the end leading to Max Verstappen overtaking Lewis Hamilton in the race, ultimately leading to him winning the championship. :'))

**Future Improvements**
- the actual distance between the drivers, if they are close to each other, that might influence the fan scores
- Weather: only considering the races that are not canceled due to bad weather
- takeover situation on the top-most positions

