import csv
from random import randrange, randint

def random_color_in_range(start = 0, end = 255):
  r = randrange(start, end)
  g = randrange(start, end)
  b = randrange(start, end)
  return r,g,b

def random_light_or_dark_color():
  choice = randint(0, 1)

  if choice == 0:
    return (random_color_in_range(0, 127), 'dark')
  else:
    return (random_color_in_range(127, 255), 'light')

def generate_csv(filename='rgb_train.csv'):
  with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(['red', 'green', 'blue', 'label'])

    for _ in range (20000):
      result = random_light_or_dark_color()
      color = result[0]
      writer.writerow([color[0], color[1], color[2], result[1]])

generate_csv('rgb_train.csv')
generate_csv('rgb_test.csv')