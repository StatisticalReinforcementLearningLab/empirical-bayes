# Sample R code to test your VS Code setup

# Print a message
print("Hello, VS Code!")

# Perform a simple calculation
result <- 2 + 2
print(paste("2 + 2 =", result))

# Create a simple plot
x <- 1:10
y <- x^2
plot(x, y,
    type = "b", col = "blue",
    main = "Simple Plot", xlab = "X-axis", ylab = "Y-axis"
)

# Load a built-in dataset and display summary
data(mtcars)
summary(mtcars)
