from itertools import product
import pandas as pd
from mip import BINARY, Model, maximize, xsum
from more_itertools import pairwise, windowed
import streamlit as st


shifts = ['day', 'night', 'off']
request = st.sidebar.file_uploader("Shift request csv")
dfws = pd.read_csv(request or "shift-request.csv")
days = dfws.columns[1:]
dffx = dfws.melt("Name", days, "Day", "Shift").dropna()

d = product(dfws.Name, days, shifts)
df = pd.DataFrame(d, columns=dffx.columns)


m = Model()
x = m.add_var_tensor((len(df),), "x", var_type=BINARY)
df["Var"] = x
m.objective = maximize(xsum(dffx.merge(df).Var))

for _, gr in df.groupby(["Name", "Day"]):
    m += xsum(gr.Var) == 1
for _, gr in df.groupby("Day"):
    m += xsum(gr[gr.Shift == "day"].Var) >= 2
    m += xsum(gr[gr.Shift == "night"].Var) >= 1
q1 = "(Day == @d1 & Shift == 'night') | "
q2 = "(Day == @d2 & Shift != 'off')"
q3 = "Day in @dd & Shift == 'off'"
for _, gr in df.groupby("Name"):
    m += xsum(gr[gr.Shift == "day"].Var) <= 4
    m += xsum(gr[gr.Shift == "night"].Var) <= 2 
    for d1, d2 in pairwise(days):
        m += xsum(gr.query(q1 + q2).Var) <= 1
    for dd in windowed(days, 4):
        m += xsum(gr.query(q3).Var) >= 1
m.optimize()
df["Val"] = df.Var.astype(float)
res = df[df.Val > 0]
res = res.pivot_table("Shift", "Name", "Day", "first")

f"""
# 👩‍⚕️NURSE SHIFT SCHDULE👨‍⚕️
## RESULT
- 📊STATUS : {m.status}
- ⭐️NUMBER OF REQUEST APPROVED : {m.objective.x}
"""  # (a3)
f = lambda s: f"color: {'red' * (s == 'off')}"
st.dataframe(res.style.applymap(f))
