"""The physical arm: actuation, marker geometry, and the fixed base policy.

    motors   Dynamixel bus driver (three cable motors)
    frames   marker frames -- reference-frame construction and the
             change-of-frame every metric in the paper is computed in
    policy   the LSTM inverse-dynamics policy, trained once on babbling data
             and held fixed while the online learners adapt
"""
