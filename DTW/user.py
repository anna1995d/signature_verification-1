import numpy as np
from Data import extract_sample,extract_user
from DTW2 import DTW, euclidean, manhattan

if __name__ == "__main__":
	A = extract_sample(15,25)
	# C = extract_sample(1,20)
	# D = extract_sample(1,30)
	# E = extract_sample(1,40)
	# print(DTW(A,B,2,4,euclidean))
	# print(DTW(A,C,2,4,euclidean))
	# print(DTW(A,D,2,4,euclidean))
	# print(DTW(A,E,2,4,euclidean))
	for i in range(40):
		for j in range(40):
			B = extract_sample(i+1,j+1)
			print(DTW(A,B,2,4,euclidean))
exit(0)