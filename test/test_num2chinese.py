from text.numbers import normalize_numbers

if __name__ == '__main__':
    num = '123456789.1234'

    ch = normalize_numbers(num)
    print(ch)
