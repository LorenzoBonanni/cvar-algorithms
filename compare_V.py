import pickle
from scipy.io import loadmat

matlabValues = -loadmat('value.mat')['im']
world, value_py, policy = pickle.load(open('vi_test.pkl', 'rb'))
print("matlabValues", matlabValues)