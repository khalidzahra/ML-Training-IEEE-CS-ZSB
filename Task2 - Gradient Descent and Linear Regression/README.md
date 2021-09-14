<h1>Implementation of Linear Regression and Gradient Descent Without External Libraries</h1>
<table>
    <tr>
        <td>
            <h2>Overview</h2>
            <p>In this task, I have implemented linear regression and gradient descent in an iterative manner in order to do it without the help of external libraries. The algorithm functions properly and returns the expected results. However, it is obviously better to implement this in a vectorized way. This was just to make sure I compeltely grasp the concept.</p>
        </td>
    </tr>
    <tr>
        <td>
            <h2>Results</h2>
            <p>The algorithm starts out with initial parameters <strong>theta = [0, 0]</strong>. After running gradient descent 50 times for an alpha value of <strong>alpha = 0.0003</strong>, a minimum is found at parameters <strong>theta = [012455766443793956, 2.2592011590562677]</strong></p>
            <p>This is a plot showing the hypothesis best fit line through the data set:
            </p>    
            <img src="./img/best_fit_img.png">
            <p>This is a plot showing the variance of the cost with the number of repetitions of gradient descent:
            </p>    
            <img src="./img/cost_function_img.png">
        </td>
    </tr>
</table>