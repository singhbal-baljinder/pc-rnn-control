"""The learners compared in the paper, behind one interface.

    online_learning     the interface: construct any method by name, then
                        predict / update. Everything below is reached through it
    predictive_coding   PC-RNN-I, the proposed method
    cerebnet            the CerebNet baseline
    rnn/                the nine online RNN learning algorithms
    gp/                 the LoG-GP baseline
    dataset             babbling-data loader
    normalization       per-channel scaling of positions and motor commands
"""
