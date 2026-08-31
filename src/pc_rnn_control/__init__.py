"""Predictive-coding RNN online learning for a cable-driven soft arm.

Three halves, deliberately separate:

    pc_rnn_control.robot     the physical arm -- motors, marker frames, and the
                             fixed LSTM inverse-dynamics policy every method
                             sits on top of
    pc_rnn_control.methods   the learners -- PC-RNN-I and every baseline it is
                             compared against, behind one interface
    pc_rnn_control.mocap     OptiTrack streaming, needed only on the robot

Nothing is re-exported here on purpose: importing the package must not pull in
torch, a motor bus or a mocap socket. Import what you need directly.
"""
