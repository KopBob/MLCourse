import random



with open('featuresX.dat', 'w') as f:
	for x in xrange(1,100):
		f.write("%d %d\n" % (random.randint(1000, 2800),
								random.randint(1, 9))) 


with open('proceY.dat', 'w') as f:
	for x in xrange(1,100):
		f.write("%d\n" % (random.randint(100, 5000))) 