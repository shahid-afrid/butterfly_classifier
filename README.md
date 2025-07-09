# 🦋 Enchanted Wings: Marvels of Butterfly Species

This project is developed as part of the **APSCHE Short-Term Virtual Internship Program** in collaboration with **SmartBridge**. The goal is to build an image classification system that accurately identifies butterfly species from images using deep learning techniques.

## 📁 Project Structure

- `app.py` – Flask web application for image upload and prediction
- `templates/` – HTML templates for the UI
- `static/` – Uploaded image previews
- `requirements.txt` – Python dependencies
- `model/` – Placeholder directory for the trained model (not included in repo due to file size)
- `notebooks/` – Jupyter Notebooks for training and evaluation
- `data/` – Training and testing datasets (stored externally)

## 👨‍💻 Team Details

- **Team ID:** LTVIP2025TMID45420  
- **Team Leader:** Shaik Mahammad Shahid Afrid  
- **Team Member:** Meda Dhruva Teja  
- **Team Member:** B Hruday  
- **Team Member:** C Ajay Kumar  

## 📦 Dataset

We used the **Butterfly Species Classification Dataset** available on [Kaggle](https://www.kaggle.com/datasets) which contains thousands of labeled butterfly images across various species. The dataset was preprocessed and split into training and testing sets using `ImageDataGenerator`.

## 🧠 Model Architecture

A **VGG16-based CNN model** was used, trained on augmented image data.  
The final model was saved as:

```bash
model/vgg16_model1.h5

🔗 Google Drive (Model, Images, and Videos)

Link:https://drive.google.com/drive/folders/1MzDBRmy4Xaw7L5sSTqSOZ11oxBWuCXW9?usp=drive_link



# Clone the repo
git clone https://github.com/your-username/butterfly-classifier.git
cd butterfly-classifier

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py
