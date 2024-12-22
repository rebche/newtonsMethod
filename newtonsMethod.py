import matplotlib.pyplot as plt

class findRoot:
    """
    A class for finding the root of a function.

    Attributes:
        function (callable): A function whose root value is being calculated for.

        derivative (callable): The derivative of the function.
    """
    
    def __init__(self, function, derivative) -> None:
        """
        Initializes a findRoot object.

        Parameters:
            function (callable): The function whose root value is being calculated for.

            derivative (callable): The derivative of the function.
        """
        self.function = function

        if derivative == None:
            raise ValueError("Newton's method cannot be performed without a derivative!")

        self.derivative = derivative
        self.approximations = [] # essentially stores the values approximations ("guesses")

    def newtonsMethod(self, initial_guess, tolerance = 1e-9):
        """
        Uses Newton's Method to calculate the root of a function, and stores all approximations
        leading up to the final computed value to be stored in a list.

        Parameters:
            initial_guess (float): The starting point for performing the approximations to calculate
            for the root of the function.

            tolerance (float): an optional value that acts as the convergence threshold, stopping
            the method when the absolute difference between the successive approximations is less
            than this value. If not provided by user, defaults to 1e-9.

        Returns:
            x_n (int): the value of the approximate root of the function
        """

        self.approximations = [initial_guess]
        x = initial_guess # f(x) = f(0) at this point

        while True:
            # function at time t
            function_x = self.function(x)
            # derivative at time t
            derivative_x = self.derivative(x)

            if derivative_x == 0:
                print("Newton's Method cannot be performed when the derivative (slope) is equal to zero.")
                break

            x_n = x - function_x / derivative_x
            self.approximations.append(x_n)

            if abs(x_n - x) < tolerance:
                # root found
                break

            x = x_n
            
        return x_n
        
    def plot(self, x_min, x_max):
        """
        Creates the visual x-y plot displaying the calculation process with each approximation from
        Newton's Method and then the final calculated root value.

        Parameters:
            x_min (float): The minimum x-value for the plot range

            x_max (float): The maximum x-value for the plot range

        Returns:
            A visual x-y graph where the function is symbolized by a solid blue line, the intermediate
            approximation as cyan points with their respective tangent lines as red dashed lines, and 
            final computed root marked with a black star.
        """
        def axis_steps(first_num, last_num, number_of_steps):
            """
            This is a helper function used to create a list of evenly spaced values between two numbers.
            This will be used to generate points on the axes of the plots.
            Parameters:
                first_num (float): The starting value of the range.

                last_num (float): The last value of the range.

                number_of_steps (int): The total number of evenly spaced values.
            
            Returns:
                This helper returns a list of floats that represent the evenly spaced values between the
                starting value and last value of the range.
            """
            
            if number_of_steps == 1:
                return [first_num]
                
            step = (last_num - first_num) / (number_of_steps - 1)
            return [first_num + i*step for i in range(number_of_steps)]
            
        x_axis = axis_steps(x_min, x_max, 400)
        y_axis = [self.function(x) for x in x_axis]

        plt.figure(figsize=(10, 5))
        plt.plot(x_axis, y_axis, 'b-', label = 'f(x)')
        plt.axhline(0, color = 'black', linewidth = 1)
        plt.axvline(0, color = 'black', linewidth = 1)

        for i in range(len(self.approximations) - 1):
            current_x = self.approximations[i]
            current_y = self.function(current_x)

            # mark the point of the current approximation
            plt.plot(current_x, current_y, 'c.', markersize=10)

            # draw the tangent line at the current approximation
            slope = self.derivative(current_x)
            x_tangent = axis_steps(current_x - 1, current_x + 1, 400)
            y_tangent = [current_y + slope * (i - current_x)for i in x_tangent]
            plt.plot(x_tangent, y_tangent, 'r--', alpha = 0.7)

        final_x = self.approximations[-1]
        final_y = self.function(final_x)
        plt.plot(final_x, final_y, 'k*', markersize = 10, label = 'Root (x = 0)')

        plt.title("Approximation of the Root of f(x)")
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid(True)
        caption = "This the approximation of the root of the function using the Newton's Method."
        plt.figtext(0.5, 0.005, caption, horizontalalignment = 'center', fontsize = 10)
        plt.show()


