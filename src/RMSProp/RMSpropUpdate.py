

beta = 0.9;
def rmsPropUpdateParams(mW, mb, dW, db):
    mW = beta * mW + (1 - beta) * dW * dW;
    mb = beta * mb + (1 - beta) * db * db;
    return mW, mb;
