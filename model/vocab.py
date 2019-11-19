class Vocab:
    def __init__(self, sf):
        self.w2i = {}
        self.i2w = []
        self.sf = sf
        self.supports = []

    def add_tokens(self, tokens, supports=None):
        for tidx, token in enumerate(tokens):
            self.add_token(token, support=None if not supports else supports[tidx])

    def add_token(self, token, support=None):
        if token not in self.w2i:
            self.w2i[token] = len(self.i2w)
            self.i2w.append(token)
            self.supports.append(support)

    def get_id(self, token):
        if token in self.w2i:
            return self.w2i[token]
        return -1

    def get_support_by_token(self, token):
        return self.get_support_by_id(self.get_id(token))

    def get_support_by_id(self, id):
        return self.supports[id]

    def get_token(self, id):
        return self.i2w[id]

    def size(self):
        return len(self.i2w)
