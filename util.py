def clamp(x, _min, _max):
    if x < _min:
        return _min
    elif x > _max:
        return _max
    else:
        return x

def mix(x, y, a):
    # https://www.opengl.org/sdk/docs/man/html/mix.xhtml
    assert a <= 1 or a >= 0
    return x*(1 - a) + y*a