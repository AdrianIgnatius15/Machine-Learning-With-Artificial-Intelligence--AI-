# Linear Regression

The first most basic machine learning model to create is called "Linear Regression". This supervised learning algorithm is used when there are data that the computer needs to predict next based on two variables/arguments that when placed in a graph, it can create a straight line (hence it's called "Linear Regression").

Linear Regression line will be drawn which tries to be as close with all the data variables/arguments when plotted. It works this way in the formula we learned in school for maths:

                                                    ```math
                                                        $y = mx + b$
                                                    ```

This formular which we learnt in school, creates the line which is straight and best fit. Now, for `y` it is called "Dependent variable" hence the graph has y-axis and `x` becomes in the "Dependent variable".

#### Goal of the Best-Fit Line

![Goal of the best-fit line](https://media.geeksforgeeks.org/wp-content/uploads/20260112155359063476/observed_value.webp)

The goal of linear regression is to find a straight line that minimizes the error (the difference) between the observed data points and the predicted values. This line helps us predict the dependent variable for new, unseen data.

Where:

    1. y is the predicted value (dependent variable)
    2. x is the input (independent variable)
    3. m is the slope of the line (how much y changes when x changes)
    4. b is the intercept (the value of y when x = 0)

#### Minimizing the Error: The Least Squares Method

To find the best-fit line, we use a method called Least Squares. The idea behind this method is to minimize the sum of squared differences between the actual values (data points) and the predicted values from the line. These differences are called residuals.

                                                    ```math
                                                        Residual = $$\sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$
                                                    ```
