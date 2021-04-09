from random import randrange


class DataValidator:
    @staticmethod
    def check_data(inputs):
        rnd = randrange(0, inputs.shape[1])
        # Mean should be zero and standard deviation
        # should be 1. However, due to some challenges
        # relationg to floating point positions and rounding,
        # the values should be very close to these numbers.
        # For details, see:
        # https://stackoverflow.com/a/40405912/947889
        # Hence, we assert the rounded values.
        print(inputs[:, rnd].std())
        print(inputs[:, rnd].mean())
