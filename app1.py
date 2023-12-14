#!/usr/bin/env python
# coding: utf-8

# In[9]:


import streamlit as st
import sys
directory_path = 'D:/'
sys.path.append(directory_path)
import Reddy_Purushotham_606_final as fp


# In[10]:


st.write("Merged Data:")
st.write(f"Accuracy: {fp.accuracy:.4f}")
st.write(f"Precision: {fp.precision:.4f}")
st.write(f"Recall: {fp.recall:.4f}")


# In[ ]:




