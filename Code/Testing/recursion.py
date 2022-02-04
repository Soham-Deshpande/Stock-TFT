def f(n):
    if n == 1:
        # print(n)
        return n
    else:
        # print(n*f(n-1))
        return n * f(n - 1)


n = 50000
print(f(n))