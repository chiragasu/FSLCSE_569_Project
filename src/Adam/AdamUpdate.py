###need 2 more variables vw and vb

beta = 0.9;
beta2 = 0.999;
def adamUpdateParams(mW, mb, dW, db):
    mW = beta * mW + (1 - beta) * dW;
    mb = beta * mb + (1 - beta) * db;
    vW = beta2 * vW + (1 - beta2) * dW * dW;
    vb = beta2 * vb + (1 - beta2) * db * db;
    return mW, mb;
