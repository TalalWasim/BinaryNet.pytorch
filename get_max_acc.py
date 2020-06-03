lines = []

with open('log.txt') as f:
    our_lines = f.readlines()
    for l in our_lines:
        if 'Training Loss' in l:
            lines.append(l)
            

accs = [float(x.strip().split(" ")[-4]) for x in lines]
print("Max Accuracy =", max(accs))
y = input("Press any key to exit")
