import pickle
from scipy.io import loadmat

# matlabValues = -loadmat('value.mat')['im']
value_py, policy = pickle.load(open('standard_vi.pkl', 'rb'))
cvar_value_py, cvar_policy = pickle.load(open('cvar_vi.pkl', 'rb'))
cvar_value_py_hand = pickle.load(open('cvar_vi_hand.pkl', 'rb'))

print(cvar_value_py)