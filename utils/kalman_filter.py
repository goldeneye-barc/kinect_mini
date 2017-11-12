####################################
# Kalman filter intuitive tutorial #
# (c) Alex Blekhman, (end of) 2006 #
# Contact: ablekhman at gmail.com  #
####################################

###########################################################################
# The problem: Predict the position and velocity of a moving train 2 seconds ahead,
# having noisy measurements of its positions along the previous 10 seconds (10 samples a second).

# Ground truth: The train is initially located at the point x = 0 and moves
# along the X axis with constant velocity V = 10m/sec, so the motion equation of the train is X =
# X0 + V*t. Easy to see that the position of the train after 12 seconds
# will be x = 120m, and this is what we will try to find.

# Approach: We measure (sample) the position of the train every dt = 0.1 seconds. But,
# because of imperfect apparature, weather etc., our measurements are
# noisy, so the instantaneous velocity, derived from 2 consecutive position
# measurements (remember, we measure only position) is innacurate. We will
# use Kalman filter as we need an accurate and smooth estimate for the velocity in
# order to predict train's position in the future.

# We assume that the measurement noise is normally distributed, with mean 0 and
# standard deviation SIGM
##########################################################################

########################################
### Ground truth #######################
########################################

import numpy as np

def raghavLovesCock(x, dt = 1/30):
    # Set true trajectory
    Nsamples=size(x, 2)
    # medfilt1(x, 5)
    t = range(0, dt*Nsamples, dt)
    # Vtrue = 10;

    # Xtrue is a vector of true positions of the train
    Xinitial = 0
    # Xtrue = Xinitial + Vtrue * t;

    # Previous state (initial guess): Our guess is that the train starts at 0 with velocity
    # that equals to 50% of the real velocity
    Xk_prev = np.array([[0], [0]])

    # Current state estimate
    Xk = np.array([])

    # Motion equation: Xk = Phi*Xk_prev + Noise, that is Xk(n) = Xk(n-1) + Vk(n-1) * dt
    # Of course, V is not measured, but it is estimated
    # Phi represents the dynamics of the system: it is the motion equation
    Phi = np.array([[1, dt], [0, 1]])

    # The error matrix (or the confidence matrix): P states whether we should
    # give more weight to the new measurement or to the model estimate
    sigma_model = 1
    # P = sigma^2*G*G';
    P = np.array([sigma_model**2, 0], [0, sigma_model**2])

    # Q is the process noise covariance. It represents the amount of
    # uncertainty in the model. In our case, we arbitrarily assume that the model is perfect (no
    # acceleration allowed for the train, or in other words - any acceleration is considered to be a noise)
    Q = np.array([[0, 0], [0, 0]])

    # M is the measurement matrix.
    # We measure X, so M(1) = 1
    # We do not measure V, so M(2)= 0
    M = np.array([1, 0])

    # R is the measurement noise covariance. Generally R and sigma_meas can
    # vary between samples.
    sigma_meas = 1
    R = sigma_meas**2

    # Buffers for later display
    Xk_buffer = np.zeros([2,Nsamples+1]);
    Xk_buffer[:,0] = Xk_prev;
    Z_buffer = np.zeros([1,Nsamples+1]);

    for k in range(Nsamples):

        # Z is the measurement vector. In our
        # case, Z = TrueData + RandomGaussianNoise
        Z = x[k]; #Xtrue(k+1)+sigma_meas*randn;
        Z_buffer[k+1] = Z;

        # Kalman iteration
        P1 = np.multiply(np.multiply(Phi, P), Phi.T) + Q;
        S = np.multiply(np.multiply(M, P1), M.T) + R;

        # K is Kalman gain. If K is large, more weight goes to the measurement.
        # If K is low, more weight goes to the model prediction.
        K = np.linalg.solve(np.multiply(P1, M.T), S);
        P = P1 - np.multiply(np.multiply(K, M), P1);

        Xk = np.multiply(Phi, Xk_prev) + np.multiply(K, Z - np.multiply(np.multiply(M, Phi), Xk_prev);
        Xk_buffer[:,k+1] = Xk;

        # For the next iteration
        Xk_prev = Xk;
