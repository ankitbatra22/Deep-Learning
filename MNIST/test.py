n = int(input())
correct = str(input())

adrian = "ABC" * n
bruno = "BABC" * n
goran = "CCAABB" * n
totals = [0,0,0]

for i in range(n):
  if adrian[i] == correct[i]:
    totals[0] += 1
  if bruno[i] == correct[i]:
    totals[1] += 1
  if goran[i] == correct[i]:
    totals[i] += 1


