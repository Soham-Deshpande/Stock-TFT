#attention block here

class SelfAttention(nn):
    def __int__(self, emb, heads=8, mask=False):
        super().__init__()

