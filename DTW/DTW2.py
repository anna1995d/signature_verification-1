import numpy as np
import math

"""
A, B are vectors 
n: number of attributes for each point
w is window size -> bounds: A = B +/- w 
"""
def DTW (A, B, n, w, dist):
	na = len(A)
	nb = len(B)

	w = max(abs(na-nb), w)
	D = np.full((na, nb), math.inf)
	d = np.full((na, nb), -1)

	d[0][0] = dist(A[0],B[0],n)
	D[0][0] = d[0][0]
	for i in range(1, min(w,max(na,nb))):
		if(i<na):
			if d[i][0] == -1:
				d[i][0] = dist(A[i],B[0],n)
			D[i][0]=D[i-1][0] + d[i][0]
		if(i<nb):
			if d[0][i] == -1:
				d[0][i] = dist(A[0],B[i],n)
			D[0][i]=D[0][i-1] + d[0][i]

	
	for i in range(1, na):
		for j in range(max(1, i-w), min(nb, i+w)):
			if d[i][j]==-1:
				d[i][j] = dist(A[i], B[j], n)
			D[i][j] = min(D[i-1][j-1], D[i][j-1], D[i-1][j])  + d[i][j]

	print(D)
	return D[na-1][nb-1]

def euclidean(A, B, n):
	sum = 0
	for i in range(n):
		if n==1:
			sum += pow(A-B, 2)
		else:	
			sum += pow(A[i]-B[i], 2)
	return math.sqrt(sum)

def manhattan(A, B, n):
	sum = 0
	for i in range(n):
		sum += abs(A[i]-B[i])
	return sum

# a = [[1,1],[2,1],[3,1],[4,1],[5,1],[6,1]]
# b = [[1,2],[2,4],[3,3],[4,3],[5,1],[6,2]]
# DTW(a,b,2,5,euclidean)
# DTW([0,1,1,2,3,2,1],[1,1,2,3,2,0],1,,euclidean)
# c = [[-.6,-.46],[-.65,-.62],[-.71,-.68],
# 	[-.58,-.63],[-.17,-.32],[.77,.74],[1.94,1.97]]
# d = [[-.87,-.88],[-.84,-.91],[-.85,-.84],[-.23,-.24],
# 	[1.95,1.92],[1.36,1.41],[.6,.51],[0,.03],[-.29,-.18]]
# DTW(c,d,2,10,euclidean)
DTW([0,1,1,2,3,2,],[1,1,2],1,1,euclidean)