
# Streamlit EDA for Clean/Dirty Road Image Classification
import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def run():

    st.title("🛣️ Road Cleanliness Image Classification - EDA")
    st.markdown("""
    In the EDA section, we’ll dive deep into the analysis of our initial datasets. We’ll use visualisation and statistics to find insights, solve problems, and identify potential challenges for modelling through Image Characteristic analysis.
    """)

    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSBOtq0ZnBAUQscpWd7i-s4-knfps_J49QGvg&s", caption="Road Cleanliness Image Classification - EDA", use_container_width=True)

    st.header("1. About the Dataset")
    st.markdown("""
    - **Source:** Kaggle - Clean/Dirty Road Classification Dataset  
    - **Classes:** 2 (Clean, Dirty)  
    - **Images:** Each image is labeled as either a clean or dirty road.  
    """)

    st.header("2. Preview Images")

    image = Image.open('deployment/Preview Dirty Class.png')
    st.image(image, caption='Dirty Road Example')

    st.markdown("""
    As you can see, dirty roads in this dataset are characterized by visible piles of trash, scattered plastics, paper, bottles, and sometimes even large accumulations of waste. These visual cues are what the model will use to distinguish between clean and dirty roads.
    """)

    image = Image.open('deployment/Preview Clean Class.png')
    st.image(image, caption='Clean Road Example')

    st.markdown("""
    As you can see, clean roads in this dataset are characterized by well-maintained pavement, clear roadlines, and a lack of visible trash or debris. These visual cues are what the model will use to distinguish between clean and dirty roads.
    """)


    st.header("3. What Makes a Road Clean or Dirty?")
    st.markdown("""
    To begin, we will first define what are each classes means.

    > **Clean :** *we can define as the road that are free from significant litter, debris, or accumulated waste on the pavement, shoulder, or immediate roadside. The road surface appears relatively clear, and road markings (lines) are often visible. The focus is on the absence of visible trash or large amounts of debris.*

    > **Dirty :** *we can define as the presence of visible litter, trash, garbage bags, or accumulated waste on the road surface, shoulder, or immediate roadside. The term “dirty” refers to human-generated waste, not mud, water, or cracks in the pavement.*



    From that definition, then we can visually inspect there are differencet charactheristic for both classes as follows:

    **Clean**

    - Well maintained pavement
    - Clear roadlines
    - Clean road shoulder
    - Natural surrounding(sky,trees,landscapes)
    - Clean infrastrctures (building, road marking)
    - No trash visible

    **Dirty**

    - Scattered Trash (plastics, bottles, packagings)
    - Piles of garbages, mountain of garbages
    - Road debris and dirts on the roadside or spilling to the road
    - Poorly maintained infrastucture
    - Unpaved Road
    """)



    st.header("4. Class Distribution")
    st.markdown("The dataset is generally balanced between clean and dirty road images, which is good for training a model.")
    st.write("#### Class Distribution")

    image = Image.open('deployment/Distribution Class.png')
    st.image(image, caption='Distribution of Clean and Dirty Classes')
    st.markdown("""
    The class distribution is balanced, with approximately equal numbers of clean and dirty road images. This balance is beneficial for training a model, as it helps prevent bias towards one class over the other.
    """)

    st.header("5. Color Channel")
    st.markdown("""            
    After checked the cell above, and it confirms that our dataset images are loading in RGB format (3 color channels).

    This is important because color information is crucial for detecting and distinguishing objects like dirt, debris, or piles of garbage, which often have striking colors.
    """)

    st.header("6. Image Resizing")
    st.markdown("""            
    All images are resized to: 128x128
                
    Image Height: 128
                
    Image Width: 128

    Above we can see the original image sizes were varied between all the datasets and we unify the size of image to define as the target size.

    The target size after resizing is 128x128, it was done in the previous step using `ImageDataGenerator`, this is necessary steps to provide uniform input to the CNN model.
                
    """)

    st.header("6. Image Characteristics")

    image = Image.open('deployment/Image Charactheristic.png')
    st.image(image, caption='Image Characteristics')

    st.markdown("""


    **Observations on Image Characteristics:**

    1. **Typical Angles/Perspective:** Angles and prespective are vary including views looking straight down the road (often highways or rural areas), angled views from the side (common in urban or dirty scenes), and slightly elevated perspectives. Not consistent across all images.


    2. **Road Surface Types:** Primarily Ashpalt or concrete roads. Some images in the class `dirty` shows dirt road, unpaved and gravel roads.


    3. **Presence of Other Objects:** Other objects such as vehicle (Cars, buses, truck and motorcylce) are common. People doing activities can also be spotted across images. There are buildings, walls, trees and grass alongside the road. The most significant presence especially in class `dirty` notably large garbage bins, piles of garbages.


    4. **Lighting Conditions:** Vary across images due to different camera or capture qualities. However, we can see images in bright sunny conditions, overcast skies, cloudy skies, and overexposed pictures (midday light).

    5. **Dirtiness**: This charactheristic exclusive to the `dirty` class, as we can see on the images it is predominantly scattered litter such as plastic bags in form of piles in the side of the road. It is not a natural dirt but mostly rather a human generated trash.

    6. **Image Quality:** The images are mostly good, but some are a bit blurry or out of focus. However, we tried to unify them by setting the image size.

    """)

    st.header("7. EDA Insight")
    st.markdown("""            
    Based on our exploratory data analysis, the litter or garbage would be our primary focus, especially in the `class 1` or `dirty` category. We want the model to be able to detect this later on, not just the general condition of the road.

    There are also variations in the characteristics of both classes, such as the presence of other objects like humans, buildings, and background scenes.

    We definitely expect the model to be able to identify other patterns beyond our sample observations. There will likely be a new type of trash pattern that occurs throughout the dataset, and the small amount of images (dataset) will be a significant challenge in training the model.
    """)

if __name__ == "__main__":
    run()