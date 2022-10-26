def extend_until(xs, until, f):
    for i in range(until - len(xs)):
        xs.append(f())
