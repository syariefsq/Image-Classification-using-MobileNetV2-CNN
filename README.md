# 🛣️ Clean/Dirty Road Image Classification

## ✨ Project Overview

Hi, I’m Syarief! This project is a hands-on exploration of how deep learning can make city life cleaner and smarter. I built an end-to-end system that classifies road images as “Clean” or “Dirty” using Convolutional Neural Networks (CNNs), with a focus on practical deployment and real-world impact. To push things further, I integrated an AI-powered strategy generator to help city planners and cleaning teams take action based on the data.

## ⚠️ Why This Project?

Manual road inspections are slow, inconsistent, and expensive. As cities grow, keeping streets clean becomes a bigger challenge. My goal was to automate this process—making it faster, more objective, and data-driven. The result is a system that can help municipalities, cleaning services, and urban planners monitor street cleanliness at scale and respond proactively.

## 🚀 Real-World Applications

- **📹 Camera Integration:** Plug into city CCTV or mobile cameras for real-time monitoring and alerts.
- **📅 Automated Scheduling:** Use prediction data to optimize cleaning routes and times.
- **📊 Policy Support:** Generate evidence-based reports for city officials.
- **📍 Route Optimization:** Prioritize dirty areas for cleaning teams.
- **💡 AI-Driven Recommendations:** The built-in Gemini-powered assistant suggests tailored cleaning strategies for different scenarios.

## 📊 Dataset

- **Source:** [Kaggle - Clean/Dirty Road Classification Dataset](https://www.kaggle.com/datasets/faizalkarim/cleandirty-road-classification/data)
- **Task:** Binary classification (Clean vs. Dirty)
- **Classes:** `Clean` and `Dirty`

## 🎯 Project Objectives

1. Prepare and preprocess the dataset for robust modeling.
2. Build and benchmark baseline CNN models.
3. Apply transfer learning (MobileNetV2, VGG16, ResNet50) to boost performance.
4. Evaluate and compare models with clear metrics and visualizations.
5. Deploy the best model as an interactive web app (HuggingFace Spaces + Streamlit).
6. Integrate Gemini API for dynamic, AI-generated cleaning strategies.

## 🌐 Live Demo

Check out the deployed app here:  
[CVRoadCleanlinessClassifier](https://huggingface.co/spaces/syariefsq/ComputerVisionRoadCleanlinessClassifier)

## 📁 Project Structure

- `P2G7_syarief_qayum.ipynb`: Main notebook for model building, training, evaluation, and analysis.
- `P2G7_syarief_suaib_inference.ipynb`: Notebook for inference and deployment prep.
- `road-cleanliness-infographic.html`: Interactive infographic with data visualizations and the AI strategy generator.

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras** — Deep learning & transfer learning
- **Scikit-learn** — Data splitting, metrics, and evaluation
- **Matplotlib & Seaborn** — Data visualization
- **Pandas & NumPy** — Data wrangling
- **Streamlit** — Web app deployment
- **Kaggle API** — Dataset access
- **HTML, CSS (Tailwind CSS), JavaScript, Chart.js** — Custom UI and interactive infographic
- **Google Gemini API** — LLM-powered strategy generation

## 🧠 Model Architecture

My models use robust CNN architectures, with transfer learning for better generalization:

- **Input:** Images resized to (128, 128, 3)
- **Convolutional & Pooling Layers:** Feature extraction
- **Flatten & Dense Layers:** Final classification with sigmoid activation
- **Callbacks:** EarlyStopping, ModelCheckpoint, ReduceLROnPlateau for efficient training

## 🔭 What’s Next?

- Expand dataset for broader generalization
- Experiment with more advanced architectures or ensembles
- Real-time video stream analysis
- Geospatial mapping of cleanliness for city dashboards
- Edge deployment for on-device inference
- Smarter LLM integrations for predictive insights and summaries

## 👋 About Me

**Syarief Qayum Suaib**  
syarif.qayyum@gmail.com
