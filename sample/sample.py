import streamlit as st
import matplotlib.pyplot as plt

# Example Data
values = [10, 20, 30]
labels = ['A', 'B', 'C']

# Create plot
fig, ax = plt.subplots()
ax.bar(labels, values)
ax.set_title("Sample Bar Plot")

#st.title('plotly graph')
# Display in Streamlit
st.pyplot(fig)
