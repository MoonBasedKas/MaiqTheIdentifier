import seaborn as sb
import matplotlib.pyplot as mlp
import pandas as pd

sum = 0
gordon = pd.read_csv("testgordon.csv")
sum = gordon[" predictions"].sum()
print(sum)
gordon[" predictions"] = gordon[" predictions"]/sum
gordon["test"] = ["gordon", "gordon","gordon"]

guy = pd.read_csv("testguy.csv")
guy[" predictions"] = guy[" predictions"]/guy[" predictions"].sum()
guy["test"] = ["guy", "guy","guy"]

scott = pd.read_csv("testscott.csv")

scott[" predictions"] = scott[" predictions"]/scott[" predictions"].sum()
scott["test"] = ["scott", "scott","scott"]
data = pd.concat([scott, guy, gordon])

print(data)
sb.set_palette(palette="Dark2")
sb.barplot(data, x="test", y=" predictions", hue="label")
mlp.xlabel("Test Name")
mlp.ylabel("Percentage of Guesses")
mlp.title("Guesses for each person for each test.")
mlp.show()