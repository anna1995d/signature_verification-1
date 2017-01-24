import numpy as np

"""	Returns UxSy.txt in an array
	user_no = x, sample_no = y """
def extract_sample(user_no, sample_no):
	file_name = "../SVC2004/Task1/U"+ str(user_no) + "S" + str(sample_no)+".txt"
	file = open(file_name)
	n = int(file.readline()) #number of points
	s = []
	for i in range(n):
		l = file.readline().split()
		s.append([float(l[0]), float(l[1])])

	a = np.array(s)
	return a

def extract_user(user_no):
	d = []
	for i in range(40):
		d.append(extract_sample(user_no, i+1))
	return np.array(d)

#print(extract_sample(1,1))
