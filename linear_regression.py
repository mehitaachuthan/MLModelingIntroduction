import numpy

def predict_image(black_density, weights, biases):
  return weights*black_density + biases

def cost_function(black_density, decision_variable, weights, biases):
  num_pixels = len(black_density)
  total_error = 0.0
  for i in range(num_pixels):
    total_error += (decision_variable[i] - (weights*black_density[i] + biases))**2
  return total_error/num_pixels

def update_weights(black_density, decision_variable, weights, biases, learning_rate):
  weight_derive = 0
  bias_derive = 0
  num_pixels = len(black_density)

  for i in range(num_pixels):
    weight_derive += -2*(black_density[i])*(decision_variable[i] - ((weights*(black_density[i])) + biases))
    bias_derive += -2*(decision_variable[i] - ((weights*black_density[i]) + biases))
  
  weights -= ((weight_derive / num_pixels) * learning_rate)
  biases -= (weight_derive / num_pixels) * learning_rate

  return weights, biases

def train(black_density, decision_variable, weights, biases, learning_rate, iters):
  cost_history = []

  for i in range(iters):
    weights, biases = update_weights(black_density, decision_variable, weights, biases, learning_rate)

    cost = cost_function(black_density, decision_variable, weights, biases)
    cost_history.append(cost)

    if(i % 10 == 0):
      print(i)
      print(cost[i])

  return weights, biases, cost_history

def driver():
  black_density = numpy.array([37.8, 39.3, 45.9, 41.3])
  decision_variable = numpy.array([22.1, 10.4, 18.3, 18.5])
  weights = numpy.array([1.0, 1.0, 1.0, 1.0])
  biases = numpy.array([0.0, 0.0, 0.0, 0.0])
  learning_rate = 0.0002
  iters = 10
  train(black_density, decision_variable, weights, biases, learning_rate, iters)
  print(predict_image(black_density, weights, biases))

if __name__ == '__main__':
  driver()