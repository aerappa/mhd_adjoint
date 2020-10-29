import pickle
import pickler
import sys

exp = sys.argv[1]
run = int(sys.argv[2])
filename = exp + '.p'
file = open('pickles/' + filename, 'rb')
my_data = pickle.load(file)
print(my_data.lo_B)
# Get to the run we want
for i in range(run+1):
    terms = pickle.load(file)
for i in range(len(terms)):
    print('T{0}={1:.5e}'.format(i, terms[i]))
T = terms
result = T[0] + T[1] + T[2] + T[4]
#result = T[4]
print('result={}'.format(result))
