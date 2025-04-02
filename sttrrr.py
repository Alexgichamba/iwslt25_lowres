import string

print(string.punctuation)
translator = str.maketrans('', '', string.punctuation)
print(translator)
lines = ['Hello, world!', 'Python is great. 123']
lines = [line.translate(translator) for line in lines]
print(lines)