import pickle
from scipy.io import loadmat

matlabValues = -loadmat('value.mat')['im']
world, value_py = pickle.load(open('vi_test.pkl', 'rb'))
print("matlabValues", matlabValues)