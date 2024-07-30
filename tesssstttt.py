import numpy as np

# Example values (you should replace these with actual values from your problem)
gamma = np.array([1.0])  # Example value, replace with actual value
QH_g = 10.0  # Example value, replace with actual value from your function
betas = [1.0]  # Example list, replace with actual values
QH_s = 5.0  # Example value, replace with actual value from your function

for i in range(len(betas)):
    exponent_value = -gamma * (QH_g + betas[i] * QH_s)
    exp_result = np.exp(exponent_value)
    
    # Print the intermediate values
    print(f"gamma: {gamma}")
    print(f"QH_g: {QH_g}")
    print(f"betas[{i}]: {betas[i]}")
    print(f"QH_s: {QH_s}")
    print(f"exponent_value: {exponent_value}")
    print(f"exp_result: {exp_result}")
