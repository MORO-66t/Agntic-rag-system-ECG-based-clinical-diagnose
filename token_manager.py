from config import HF_TOKENS


class TokenManager:

    def __init__(self):

        self.tokens = HF_TOKENS
        self.current = 0

    def get_token(self):

        return self.tokens[self.current]

    def rotate(self):

        self.current += 1

        if self.current >= len(self.tokens):

            self.current = 0

        return self.tokens[self.current]