import streamlit as st

def new_line(n=1, a = st):
    for _ in range(n):
        a.write("")
