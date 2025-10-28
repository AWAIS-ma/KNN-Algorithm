from KNN import knn

animal = input("eneter animal(goat, sheep, cow, buffelo) : ")
sym1 = input("SYM 1: ")
sym2 = input("SYM 2: ")
sym3 = input("SYM 3: ")

result = knn(animal,sym1,sym2,sym3)

print(f"disease : {result['Predicted Disease']}")
print(f"Confidence: {result['Confidence']}%")