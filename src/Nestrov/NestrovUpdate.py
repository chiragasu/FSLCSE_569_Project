



beta =0.9;

def NestrovUpdateParams(mW, mb, dW, db):
    mW = beta * mW + (1 - beta) * dW;
    mb = beta * mb + (1 - beta) * db;
    return mW, mb;