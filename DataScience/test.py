import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y1 = [2, 3, 4, 5]
y2 = [1, 4, 2, 5]

plt.plot(x, y1, label="linear")
plt.plot(x, y2, label="nonlinear")

plt.legend()
plt.title("linear vs nonlinear")
plt.show()