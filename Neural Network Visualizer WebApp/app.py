
import streamlit as st
import json, requests
import matplotlib.pyplot as plt
import numpy as np

URL = 'http://127.0.0.1:5000'

st.title('Neural Network Visualizer')
st.sidebar.markdown('## Input Image:')

if st.button('Get Random Prediction'):
    resp = requests.post(URL, data={})
    resp = json.loads(resp.text)
    preds = resp.get('prediction')
    image = resp.get('image')
    image = np.reshape(image, (28,28))
    
    st.sidebar.image(image, width=150)
    
    for layer, p in enumerate(preds):
        numbers = np.squeeze(np.array(p))
        plt.figure(figsize=(32,4))
        row, col = (1, 10) if layer == 2 else (2, 16)
        for i, number in  enumerate(numbers):
            plt.subplot(row, col, i+1)
            plt.imshow(number * np.ones((8,8,3)).astype('float32'))
            plt.xticks([]); plt.yticks([])
            if layer == 2:
                plt.xlabel(str(i), fontsize=40)
        
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.tight_layout()
        st.text(f'Layer {layer+1}')
        st.pyplot()
